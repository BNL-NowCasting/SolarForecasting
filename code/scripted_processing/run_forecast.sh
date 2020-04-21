#!/bin/bash
python3 preprocess.py
python3 generate_stitchv3.py
python3 extract_featuresv2.py
python3 forecast_metrics_kml.py
python3 predictv3.py