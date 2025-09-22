#!/bin/bash

# Navigate to the aitlas project directory
cd /home/dragik/tadej/aitlas

# Install the package in editable mode from the current directory
# This ensures that any changes to the source code are reflected immediately
pip install -e .

# Navigate into the tests directory
cd tests

# Run the specified pytest file, disabling warnings to keep the output clean
#pytest --disable-warnings test_dofa_v2_0.py
#pytest -s --disable-warnings test_dofa_v2_wrapper.py
pytest -s --disable-warnings test_scale_mae_wrapper.py
#pytest test_dofa_v2_0.py