#!/usr/bin/env bash

# Force Render to use pip instead of Poetry
echo "Disabling Poetry and using pip + requirements.txt"
pip install -r requirements.txt
