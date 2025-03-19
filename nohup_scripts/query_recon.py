#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
from pathlib import Path
import pycolmap
import torch
import torchvision.transforms as transforms
import numpy as np
from hloc import extract_features, match_features, reconstruction, pairs_from_retrieval
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster

"""
  This script replicates the logic for reference and query image localization
  based on the process demonstrated in the Jupyter notebook.

  It uses:
    - NetVLAD for global descriptor extraction (retrieval_conf)
    - SuperPoint for feature extraction (feature_conf)
    - LightGlue for feature matching (matcher_conf)
    - colmap-based reconstruction in pycolmap
    - hloc pipelines for feature extraction, matching, pair retrieval, and pose estimation
"""

device = "cuda" if torch.cuda.is_available() else "cpu"

# Global configuration for extraction and matching
retrieval_conf = extract_features.confs["netvlad"]
retrieval_conf["device"] = "cuda"  # Force GPU for NetVLAD

feature_conf = extract_features.confs["superpoint_aachen"]
feature_conf["device"] = "cuda"    # Force GPU for SuperPoint

matcher_conf = match_features.confs["superpoint+lightglue"]
matcher_conf["device"] = "cuda"    # Force GPU for LightGlue

def filter_query_images(ref_register_list):
    """
    Filters out images from 'rgb7', 'rgb8', 'rgb9' for references to avoid query overlap.
    """
    skip_list = ['rgb7', 'rgb8', 'rgb9']
    filtered = []
    for name in ref_register_list:
        prefix = name.split('/')[0]
        if prefix not in skip_list:
            filtered.append(name)
    return filtered

def parse_retrieval(path):
    """
    Reads the pairs from the NetVLAD retrieval output file (pairs-loc.txt).
    Returns a list of reference image names that are retrieved for each query.
    """
    retrieval = []
    with open(path, "r") as f:
        for line in f.read().splitlines():
            if not line.strip():
                continue
            q, r = line.split()
            retrieval.append(r)
    return retrieval

def query_position(
    query_list, model, references_registered, model_index, 
    feature_conf, retrieval_conf, matcher_conf, images, outputs
):
    """
    Localizes a list of queries against a given model by:
      - extracting features,
      - global descriptors,
      - retrieving reference pairs,
      - matching local features,
      - and finally computing the pose with pose_from_cluster.
    
    Writes results in a JSON file.
    """
    count = 0
    time_total = 0.0
    output_file_path = f"{outputs.resolve()}_{model_index}_query_results.json"
    number_of_match = 10

    with open(output_file_path, "w") as json_file:
        json_file.write("[\n")  # begin JSON array
        for query in query_list:
            count += 1
            print(f"progress: {count / len(query_list)} | query: {query}")
            time_start = time.time()

            # Local and global feature extraction for the query
            extract_features.main(
                feature_conf, images, image_list=[query], 
                feature_path=outputs / "features.h5", 
                overwrite=True
            )
            extract_features.main(
                retrieval_conf, images, image_list=[query], 
                feature_path=outputs / "features_retrieval.h5"
            )

            # Retrieve top-K references for the query and store in pairs-loc.txt
            pairs_from_retrieval.main(
                outputs / "features_retrieval.h5", 
                outputs / "pairs-loc.txt", 
                num_matched=number_of_match, 
                db_list=references_registered, 
                query_list=[query]
            )

            # Match local features
            match_features.main(
                matcher_conf, 
                outputs / "pairs-loc.txt", 
                features=outputs / "features.h5", 
                matches=outputs / "matches.h5", 
                overwrite=True
            )

            # Parse netvlad retrieval results
            retrieval_images = parse_retrieval(outputs / "pairs-loc.txt")
            camera = pycolmap.infer_camera_from_image(images / query)

            # Gather 3D references from retrieval output
            ref_ids = []
            for n in references_registered:
                if n in retrieval_images:
                    img_id = model.find_image_with_name(n).image_id
                    ref_ids.append(img_id)

            conf = {
                "estimation": {"ransac": {"max_error": 12}},
                "refinement": {
                    "refine_focal_length": True, 
                    "refine_extra_params": True
                },
            }
            localizer = QueryLocalizer(model, conf)

            try:
                # Attempt to localize the query
                ret, log = pose_from_cluster(
                    localizer, query, camera, ref_ids, 
                    outputs / "features.h5", outputs / "matches.h5"
                )
                time_end = time.time()
                inference_time = time_end - time_start
                time_total += inference_time

                result = {
                    "query": query,
                    "rotation": ret["cam_from_world"].rotation.quat.tolist(),
                    "translation": ret["cam_from_world"].translation.tolist(),
                    "inference_time": inference_time
                }
                json.dump(result, json_file)
                if count < len(query_list):
                    json_file.write(",\n")
                else:
                    json_file.write("\n")
            except Exception as e:
                print("="*60)
                print(f"Error localizing {query} with model {model_index}: {e}")
                print("="*60)
        json_file.write("]\n")  # end JSON array

    print(f"avg inference time: {time_total / len(query_list):.4f} seconds")

def main():
    """
    Main entry point for reference reconstruction and query localization.
    """
    # Adjust paths according to your setup
    images = Path("/home/siyanhu/Gits/Tramway/Hierarchical-Localization/datasets/sacre_coeur/mapping")
    queries_dir  = Path("/home/siyanhu/Gits/Tramway/Hierarchical-Localization/datasets/sacre_coeur/mapping")
    outputs_ref = Path("/home/siyanhu/Gits/Tramway/Hierarchical-Localization/datasets/sacre_coeur/hloc")
    outputs_query = Path("/home/siyanhu/Gits/Tramway/Hierarchical-Localization/datasets/sacre_coeur/query")
    outputs_query.mkdir(parents=True, exist_ok=True)
    sfm_dir = outputs_ref / "sfm"

    # Number of pre-computed sub-models in sfm_dir/models/
    total_model_num = 1
    models = []
    ref_registers = {}

    # Gather the colmap reconstructions and filter out queries from references
    for model_index in range(total_model_num):
        # Each sub-reconstruction is stored in: sfm_dir/models/<model_index>
        model_path = sfm_dir / "models" / str(model_index)
        model = pycolmap.Reconstruction(model_path)

        # ID of images that are already registered
        references_registered_init = [model.images[i].name for i in model.reg_image_ids()]
        references_registered = filter_query_images(references_registered_init)

        models.append(model)
        ref_registers[model_index] = references_registered

    # Suppose you have some query list already determined (e.g., 'rgb7', 'rgb8', 'rgb9' images)
    # In a real setup, you'd collect these from your dataset or command-line arguments
    query_paths = []
    if any(queries_dir.iterdir()):
        subdirs = [p for p in queries_dir.iterdir() if p.is_dir()]
        if subdirs:  # If subdirectories exist
            for subdir in subdirs:
                for img_path in subdir.glob('*.jpg'):  # Search for .jpg files in subdirectories
                    query_paths.append(str(img_path.relative_to(queries_dir)))
        else:  # If no subdirectories, look directly in the images folder
            for img_path in queries_dir.glob('*.jpg'):  # Search for .jpg files in images folder
                query_paths.append(str(img_path.relative_to(queries_dir)))

    # Localize the queries against each sub-model
    for idx, one_model in enumerate(models):
        references = ref_registers[idx]
        query_position(
            query_paths,
            one_model,
            references,
            idx,
            feature_conf,
            retrieval_conf,
            matcher_conf,
            images,
            outputs_query
        )

if __name__ == "__main__":
    main()