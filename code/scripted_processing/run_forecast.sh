#!/bin/bash
python3 preprocess.py
python3 generate_stitch.py
python3 extract_features.py
python3 forecast_metrics_kml.py
python3 predict.py