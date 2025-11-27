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
#pytest -s --disable-warnings test_satmae_wrapper.py
#pytest -s --disable-warnings test_satmae_plusplus_wrapper.py
#pytest -s --disable-warnings test_scale_mae_wrapper.py
#pytest -s --disable-warnings test_anysat_wrapper.py
#pytest -s --disable-warnings test_presto_wrapper.py
#pytest -s --disable-warnings test_gassl_wrapper.py
#pytest -s --disable-warnings test_seco_wrapper.py
#pytest -s --disable-warnings test_caco_wrapper.py
#pytest -s --disable-warnings test_prithvi_wrapper.py
#pytest -s --disable-warnings test_galileo_wrapper.py
#pytest -s --disable-warnings test_panopticon_wrapper.py
#pytest -s --disable-warnings test_copernicusfm_wrapper.py
pytest -s --disable-warnings test_croma_wrapper.py
#pytest test_dofa_v2_0.py