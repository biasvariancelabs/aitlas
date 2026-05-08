# Installation

AiTLAS requires Python 3.12. While you can use standard `pip`, we highly recommend [`uv`](https://github.com/astral-sh/uv) for significantly faster installations. This will automatically handle all dependencies defined in `pyproject.toml`.

## Option 1: Install from PyPI (recommended)
The easiest way to install AiTLAS is directly via PyPI:
```bash
uv pip install aitlas
```

## Option 2: Install from the source

- Clone the AiTLAS repository
```bash
git clone https://github.com/biasvariancelabs/aitlas.git
```

- Go to the folder where you cloned the repo

- Install using `uv`
```bash
uv pip install .
```

- Or, for developers (editable mode)
```bash
uv pip install -e .
```

- Verify the installation
```bash
python -c "import aitlas; print(f'AiTLAS version: {aitlas.__version__}')"
```

- Running AiTLAS
```bash
python -m aitlas.run configs/example_config.json
```


<!-- ---

**Note:** You will have to download the datasets from their respective source. You can find a link for each dataset in the respective dataset class in `aitlas/datasets/` or use the **AiTLAS Semantic Data Catalog** available at [eodata.bvlabs.ai](http://eodata.bvlabs.ai).

You can also find various trained models, model configurations and processing details of many datasets (with their corresponding splits used for training and evaluating the models) in our **AiTLAS Benchmark Arena** repository available at [aitlas-arena.bvlabs.ai](aitlas-arena.bvlabs.ai).

--- -->
