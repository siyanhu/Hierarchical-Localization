#!/bin/bash

# Run the pipeline
python nohup_scripts/load_modules.py
python nohup_scripts/load_image_sequences.py
python nohup_scripts/extract_features_and_match.py
python nohup_scripts/reconstruction.py