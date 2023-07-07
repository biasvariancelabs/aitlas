# AiTLAS installation

The best way to install `aitlas`, is if you create a virtual environment and install the  requirements with `pip`. Here are the steps:
- Go to the folder where you cloned the repo.
- Create a virtual environment
```bash
conda create -n aitlas python=3.8
```
- Use the virtual environment
```bash
conda activate aitlas
```
- Before installing `aitlas` on Windows it is recommended to install the following packages 
from [Unofficial Windows wheels repository](https://www.lfd.uci.edu/~gohlke/pythonlibs/):
```bash
pip install GDAL-3.4.1-cp38-cp38-win_amd64.whl 
pip install Fiona-1.8.20-cp38-cp38-win_amd64.whl
pip install rasterio-1.2.10-cp38-cp38-win_amd64.whl
```
- Install the requirements
```bash
pip install -r requirements.txt
```
And, that's it, you can start using `aitlas`!
```bash
python -m aitlas.run configs/example_config.json
```
If you want to use `aitlas` as a package run
```bash
pip install .
```
in the folder where you cloned the repo.

---