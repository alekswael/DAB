#!/bin/bash

source venv/bin/activate

label-studio start
xdg-open http://localhost:8080

deactivate
