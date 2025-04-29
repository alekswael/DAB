#!/bin/bash

# Activate venv
source venv/bin/activate

# Start label studio and open
label-studio start
xdg-open http://localhost:8080

# deactivate venv
deactivate