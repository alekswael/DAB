#!/bin/bash

source venv/bin activate

python3 src/data_handling.py
python3 src/pre_annotate.py

deactivate