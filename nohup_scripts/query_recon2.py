#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
from pathlib import Path
import pycolmap
import torch
from tqdm import tqdm
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

def filter_query_images(ref_register_list, skip_seq_list=None):
    if skip_seq_list is None:
        return ref_register_list
    filtered = []
    for name in ref_register_list:
        prefix = name.split(os.sep)[0]
        if prefix not in skip_seq_list:
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
        sfm_model, model_index,
        query_list, query_dir, outputs_query,
        references_registered, ref_dir, outputs_ref,
        feature_conf, retrieval_conf, matcher_conf,
        number_of_matches = 20
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
    features = outputs_query / 'features.h5'
    matches = outputs_query / 'matches.h5'
    retrieval_features = outputs_query/'features_retrieval.h5'
    loc_pairs = outputs_query / 'pairs-loc.txt'

    db_features = outputs_ref / 'features.h5'
    db_retrieval_features = outputs_ref / 'features_retrieval.h5'

    count = 0
    time_total = 0.0
    output_file_path = outputs_query / "query_results.json"

    conf = {
                'estimation': {'ransac': {'max_error': 12}},
                'refinement': {'refine_focal_length': True, 'refine_extra_params': True},
            }
    localizer = QueryLocalizer(sfm_model, conf)

    with open(output_file_path, "w") as json_file:
        json_file.write("[\n")  # begin JSON array
        for query in tqdm(query_list, desc="Processing queries"):
            count += 1
            print(f"progress: {count / len(query_list)} | query: {query}")
            time_start = time.time()
            extract_features.main(feature_conf, query_dir, image_list=[query], feature_path=features, overwrite=True, query_mode=True)
            global_descriptors = extract_features.main(retrieval_conf, query_dir, image_list=[query], feature_path=retrieval_features, query_mode=True)
            pairs_from_retrieval.query_main(
                descriptors=global_descriptors,
                db_descriptors=db_retrieval_features,
                output=loc_pairs, 
                num_matched=number_of_matches, 
                db_list=references_registered, 
                query_list=[query]
            )
            match_features.main(
                conf=matcher_conf, 
                pairs=loc_pairs, 
                features=features, 
                matches=matches, 
                features_ref=db_features,
                overwrite=True
            )

            camera = pycolmap.infer_camera_from_image(query_dir / query)
            retrieval_images = parse_retrieval(loc_pairs)
            ref_ids = []
            for n in references_registered:
                if n in retrieval_images:
                    ref_ids.append(sfm_model.find_image_with_name(n).image_id)

            try:
                ret, log = pose_from_cluster(localizer, query, camera, ref_ids, features, matches)
                time_end = time.time()
                inference_time = time_end - time_start
                time_total += inference_time
                
                # Create the result dictionary
                result = {
                    "query": query,
                    "rotation": ret['cam_from_world'].rotation.quat.tolist(),
                    "translation": ret['cam_from_world'].translation.tolist(),
                    "inference_time": inference_time
                }
                
                # Write the result to the file
                json.dump(result, json_file)
                
                # Add a comma if this is not the last query
                if count < len(query_list):
                    json_file.write(',\n')
                else:
                    json_file.write('\n')  # Final entry, no comma
            except Exception as e:
                print("============================================")
                print(e)
                print("============================================")
                pass
        json_file.write(']\n')  # End the JSON array
    print('avg inference time: ', time_total / len(query_list))

            
def main():
    """
    Main entry point for reference reconstruction and query localization.
    """
    # Adjust paths according to your setup
    ref_dir = Path("/media/siyanhu/Changkun/Siyan/Tramway/process/lidar_rgb5/frames")
    queries_dir  = Path("/media/siyanhu/Changkun/Siyan/Tramway/process/lidar_rgb5/frames/2024")
    skip_query_seq_from_ref = ["2024"]
    outputs_ref = Path("/media/siyanhu/Changkun/Siyan/Tramway/process/lidar_rgb5/hloc")
    
    outputs_query = outputs_ref / "queries_2024"
    outputs_query.mkdir(parents=True, exist_ok=True)
    sfm_dir = outputs_ref / "sfm"

    # Number of pre-computed sub-models in sfm_dir/models/
    total_model_num = 100
    models = []
    ref_registers = {}

    for model_index in range(total_model_num):
        # Each sub-reconstruction is stored in: sfm_dir/models/<model_index>
        model_path = sfm_dir / "models" / str(model_index)
        if not model_path.exists():
            break
        model = pycolmap.Reconstruction(model_path)
        references_registered_init = [model.images[i].name for i in model.reg_image_ids()]
        references_registered = filter_query_images(references_registered_init, skip_seq_list=skip_query_seq_from_ref)
        models.append(model)
        ref_registers[model_index] = references_registered

    # Suppose you have some query list already determined (e.g., 'rgb7', 'rgb8', 'rgb9' images)
    ref_paths = []
    if any(ref_dir.iterdir()):
        subdirs = [p for p in ref_dir.iterdir() if p.is_dir()]
        if subdirs:  # If subdirectories exist
            for subdir in subdirs:
                for img_path in subdir.glob('*.jpg'):  # Search for .jpg files in subdirectories
                    ref_paths.append(str(img_path.relative_to(ref_dir)))
        else:  # If no subdirectories, look directly in the images folder
            for img_path in ref_dir.glob('*.jpg'):  # Search for .jpg files in images folder
                ref_paths.append(str(img_path.relative_to(ref_dir)))

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

    print(ref_paths[5:10], ref_paths[-10:-5])
    print(query_paths[5:10], query_paths[-10:-5])

    # Localize the queries against each sub-model
    for idx, one_model in enumerate(models):
        references = ref_registers[idx]
        query_position(
            one_model, idx,
            query_paths, queries_dir, outputs_query,
            references, ref_dir, outputs_ref,
            feature_conf, retrieval_conf, matcher_conf
        )

if __name__ == "__main__":
    main()