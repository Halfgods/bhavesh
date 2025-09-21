#!/bin/bash

# Install missing system libraries
apt-get update
apt-get install -y libgl1 libglib2.0-0

# Start your Python app (change main:app if needed)
python3 main.py
