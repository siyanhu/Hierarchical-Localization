#!/usr/bin/env python3

"""
Hierarchical Localization Pipeline for Tramway Dataset (hloc==1.5)

This script consolidates the core steps from the Jupyter notebook:
1. Copy or gather all relevant images into a single dataset directory.
2. Configuration for retrieval, local feature extraction, and matcher.
3. Run the hloc pipeline: extract features, generate pairs, match features,
   create COLMAP database, perform geometric verification, and reconstruct.

Adjust paths and parameters for your environment.
"""

import os
import struct
import tqdm
from pathlib import Path
import numpy as np
import collections

# If you have installed hloc v1.5, import:
# pip install git+https://github.com/cvg/Hierarchical-Localization.git@v1.5
from hloc import extract_features, match_features, reconstruction, pairs_from_retrieval

import torch
import torchvision.transforms as transforms

def main():
    # Set up device: CPU or GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --------------------------------------------------------------
    # Step 1: Define paths and copy images (adjust to your needs)
    # --------------------------------------------------------------
    # Example commands from the original notebook (not automatically run here):
    #
    # os.system("cp -r /media/.../images /media/.../lidarcam/rgb7")
    # os.system("cp -r /media/.../images /media/.../lidarcam/rgb8")
    # os.system("cp -r /media/.../images /media/.../lidarcam/rgb9")
    #
    # The above commands are provided as a reference. You can adapt or remove them.

    # --------------------------------------------------------------
    # Step 2: Retrieval, feature, and matcher configurations
    # --------------------------------------------------------------
    retrieval_conf = extract_features.confs['netvlad'].copy()
    feature_conf   = extract_features.confs['superpoint_aachen'].copy()
    matcher_conf   = match_features.confs['superpoint+lightglue'].copy()

    # Force GPU usage if available
    retrieval_conf['device'] = 'cuda'
    feature_conf['device']   = 'cuda'
    matcher_conf['device']   = 'cuda'

    # --------------------------------------------------------------
    # Step 3: Path setup for images, output, and intermediate files
    # --------------------------------------------------------------
    images_dir  = Path("/media/siyanhu/T71/hloc/datasets")
    outputs_dir = Path("/media/siyanhu/T71/hloc/dji_recon")

    sfm_pairs_path = outputs_dir / "pairs-netvlad.txt"
    loc_pairs_path = outputs_dir / "pairs-loc.txt"

    sfm_dir          = outputs_dir / "sfm"
    features_path    = outputs_dir / "features.h5"
    matches_path     = outputs_dir / "matches.h5"
    retrieval_path_h5 = outputs_dir / "features_retrieval.h5"

    # List of image subfolders or sequences
    # Example:
    #   ref_seqs = [7, 8, 9]
    #   We only pick images from those sequences to form the reference paths.

    ref_seqs = [7, 8, 9]
    ref_paths = []
    for seq_num in range(1, 10):
        if seq_num in ref_seqs:
            subfolder = images_dir / f"rgb{seq_num}"
            for p in subfolder.iterdir():
                if p.suffix.lower() == ".jpg":
                    rel_path = str(p.relative_to(images_dir))
                    ref_paths.append(rel_path)

    # Show an example slice
    print("Example subset of reference paths:", ref_paths[:10])

    # --------------------------------------------------------------
    # Step 4: Global feature extraction + image pairs from retrieval
    # --------------------------------------------------------------
    print("\n[Step 4] Global retrieval feature extraction + pairs_from_retrieval")
    retrieval_path = extract_features.main(
        retrieval_conf,
        images_dir,
        image_list=ref_paths,
        feature_path=retrieval_path_h5
    )
    pairs_from_retrieval.main(retrieval_path, sfm_pairs_path, num_matched=20)
    print(f"Retrieval features output: {retrieval_path}")

    # --------------------------------------------------------------
    # Step 5: Local feature extraction & matching
    # --------------------------------------------------------------
    print("\n[Step 5] Local feature extraction & matcher")
    extract_features.main(
        feature_conf,
        images_dir,
        image_list=ref_paths,
        feature_path=features_path
    )
    match_features.main(
        matcher_conf,
        sfm_pairs_path,
        features=features_path,
        matches=matches_path
    )

    # --------------------------------------------------------------
    # Step 6: 3D reconstruction with COLMAP
    # --------------------------------------------------------------
    print("\n[Step 6] 3D reconstruction using COLMAP (via hloc.reconstruction.run_colmap)")
    # This step uses the hloc reconstruction module
    # and it merges the matches, features, etc. into a COLMAP database
    # then runs the incremental pipeline. The complete process can take a while.

    reconstruction.run_colmap(
        images_dir,
        sfm_dir,
        sfm_pairs_path,
        features_path,
        matches_path,
        skip_geometric_verification=False,  # set True if you already did it
        image_list=ref_paths
    )

    print("Reconstruction complete!")

if __name__ == "__main__":
    main()