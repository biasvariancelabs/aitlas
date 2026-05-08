[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg?style=for-the-badge)](https://www.repostatus.org/#active) [![Python 3.12](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/release/python-3120/) [![License: Apache License 2.0](https://img.shields.io/badge/License-Apache%202.0-olivegreen.svg)](https://github.com/biasvariancelabs/aitlas/blob/master/LICENSE) [![Documentation Status](https://readthedocs.org/projects/aitlas/badge/?version=latest)](https://aitlas.readthedocs.io/en/latest/?badge=latest)

![logo](media/AiTALS_horizontal_gradient_subtitle.png)


The AiTLAS toolbox (Artificial Intelligence Toolbox for Earth Observation) includes state-of-the-art machine learning methods for exploratory and predictive analysis of satellite imagery as well as repository of AI-ready Earth Observation (EO) datasets. It can be easily applied for a variety of Earth Observation tasks, such as land use and cover classification, crop type prediction, localization of specific objects (semantic segmentation), etc. The main goal of AiTLAS is to facilitate better usability and adoption of novel AI methods (and models) by EO experts, while offering easy access and standardized format of EO datasets to AI experts which allows benchmarking of various existing and novel AI methods tailored for EO data.

# 📢 Latest updates
### 📅 **May 8, 2026**
🔥🔥🔥 AiTLAS 2.0.0 is out now! 🔥🔥🔥

## 🛰️ New foundation models & adapters
- Added comprehensive support for the following **foundation models**: [AnySat](https://github.com/gastruc/AnySat), [CACo](https://github.com/utkarshmall13/CACo), [Copernicus-FM](https://github.com/zhu-xlab/Copernicus-FM), [CROMA](https://github.com/antofuller/CROMA), [DOFA](https://github.com/zhu-xlab/DOFA), [Galileo](https://github.com/nasaharvest/galileo), [GASSL](https://github.com/sustainlab-group/geography-aware-ssl), [Panopticon](https://github.com/Panopticon-FM/panopticon), [Presto](https://github.com/nasaharvest/presto), [Prithvi](https://github.com/NASA-IMPACT/Prithvi-EO-2.0), [SatMAE](https://github.com/sustainlab-group/SatMAE), [SatMAE++](https://github.com/techmn/satmae_pp), [Scale-MAE](https://github.com/bair-climate-initiative/scale-mae), [SeCo](https://github.com/ServiceNow/seasonal-contrast), [TerraFM](https://github.com/mbzuai-oryx/TerraFM), [TerraMind](https://github.com/IBM/terramind)

- A table summarizing input modalities for foundation models:

| Foundation model | RGB | S1 | S2 | L8 | DEM |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **AnySat** | 🔴 NO | 🟢 YES | 🟢 YES (no B1, B9, B10) | 🟢 YES | 🔴 NO |
| **CACo** | 🟢 YES | 🔴 NO | 🔴 NO | 🔴 NO | 🔴 NO |
| **Copernicus-FM** | 🟢 YES | 🔴 NO | 🟢 YES | 🔴 NO | 🔴 NO |
| **CROMA** | 🔴 NO | 🟢 YES | 🟢 YES (no B10) | 🔴 NO | 🔴 NO |
| **DOFA** | 🟢 YES | 🔴 NO | 🟢 YES | 🔴 NO | 🔴 NO |
| **Galileo** | 🔴 NO | 🟢 YES | 🟢 YES (no B1, B9, B10) | 🔴 NO | 🔴 NO |
| **GASSL** | 🟢 YES | 🔴 NO | 🔴 NO | 🔴 NO | 🔴 NO |
| **Panopticon** | 🔴 NO | 🟢 YES | 🟢 YES (no B10) | 🟢 YES | 🔴 NO |
| **Presto** | 🔴 NO | 🟢 YES | 🟢 YES (no B1, B9, B10) | 🔴 NO | 🔴 NO |
| **Prithvi** | 🔴 NO | 🔴 NO | 🟢 YES (no B1, B5, B6, B7, B8, B9, B10) | 🔴 NO | 🔴 NO |
| **SatMAE** | 🟢 YES | 🔴 NO | 🟢 YES (no B1, B9, B10) | 🔴 NO | 🔴 NO |
| **SatMAE++** | 🟢 YES | 🔴 NO | 🟢 YES (no B1, B9, B10) | 🔴 NO | 🔴 NO |
| **Scale-MAE** | 🟢 YES | 🔴 NO | 🔴 NO | 🔴 NO | 🔴 NO |
| **SeCo** | 🟢 YES | 🔴 NO | 🔴 NO | 🔴 NO | 🔴 NO |
| **TerraFM** | 🔴 NO | 🔴 NO | 🟢 YES (no B10) | 🔴 NO | 🔴 NO |
| **TerraMind** | 🟢 YES | 🟢 YES | 🟢 YES | 🔴 NO* | 🟢 YES |

\* Can be added as a new modality.

## 🧠 New model architectures
- **Change detection**: Added [BIT](https://github.com/justchenhao/BIT_CD), [CGNet](https://github.com/wutianyiRosun/CGNet), [ChangeFormer V6](https://github.com/wgcban/ChangeFormer), [ChangeMamba](https://github.com/ChenHongruixuan/ChangeMamba), [ChangeVIT](https://github.com/zhuduowang/ChangeViT), [CSSM](https://github.com/Elman295/CSSM), HRNet SiamConc, [SiamCRNN](https://github.com/ChenHongruixuan/SiamCRNN), [STANet](https://github.com/justchenhao/STANet), [TinyCD](https://github.com/AndreaCodegoni/Tiny_model_4_CD), and U-Net SiamConc.
- **Object detection**: [ATSS](https://github.com/sfzhang15/ATSS), [CenterNet](https://github.com/xingyizhou/CenterNet), [DETR](https://github.com/facebookresearch/detr), [EfficientDet](https://github.com/signatrix/efficientdet), [NanoDet-Plus](https://github.com/RangiLyu/nanodet), and [Sparse R-CNN](https://github.com/PeizeSun/SparseR-CNN).
- **Segmentation**: Added [FPN](https://github.com/qubvel/segmentation_models.pytorch), [MaNet](https://github.com/qubvel/segmentation_models.pytorch), [PSPNet](https://github.com/hszhao/PSPNet), [SegFormer](https://github.com/NVlabs/SegFormer), [UNet++](https://github.com/qubvel/segmentation_models.pytorch), and [UPerNet](https://github.com/qubvel/segmentation_models.pytorch).

## 🛠️ Key improvements & features
- **New build system**: Complete migration to `uv` and `pyproject.toml` for faster, reproducible builds.
- **Modern infrastructure**: Switched to `ruff` for ultra-fast linting and formatting.
- **Foundation model architecture**: Implemented `CompositeModel`, allowing for dynamic building of backbones, necks, decoders, heads, and data-model adapters.
- **Training**: Added Automatic Mixed Precision (AMP), early stopping on NaN loss, and state preservation (LR scheduler/checkpoints) for restarts.
- **Adapters**: Implemented specific data-model adapters for foundation models, such as Terramind, AnySat, Galileo, and Panopticon.
- **Examples**: Added Jupyter notebook examples for new foundation models and downstream tasks (e.g., change detection).

## ⚠️ Breaking changes
- Minimum Python version is now **3.12**.
- Removed `requirements.txt` in favor of `pyproject.toml` dependencies.
- Namespaced foundation model classes in `aitlas.models` to resolve implementation collisions.

# Getting started

### **v2.0.0**

AiTLAS examples:
- Multiclass classification with Galileo foundation model: https://github.com/biasvariancelabs/aitlas/blob/aitlas-uv-migration/examples/multiclass_classification_example_galileo_brazildam.ipynb
- Multilabel classification with Copernicus-FM foundation model: https://github.com/biasvariancelabs/aitlas/blob/aitlas-uv-migration/examples/multilabel_classification_example_copernicusfm_dlrsd_multilabel.ipynb
- Semantic segmentation with TerraMind foundation model: https://github.com/biasvariancelabs/aitlas/blob/aitlas-uv-migration/examples/semantic_segmentation_example_terramind_dlrsd.ipynb
- Feature extraction with TerraMind foundation model: https://github.com/biasvariancelabs/aitlas/blob/aitlas-uv-migration/examples/feature_extraction_example_terramind_dlrsd.ipynb
- Change detection with U-Net SiamConc: https://github.com/biasvariancelabs/aitlas/blob/aitlas-uv-migration/examples/change_detection_example_california_flood.ipynb

### **v1.0.0**

AiTLAS Introduction https://youtu.be/-3Son1NhdDg

AiTLAS Software Architecture: https://youtu.be/cLfEZFQQiXc

AiTLAS in a nutshell: https://www.youtube.com/watch?v=lhDjiZg7RwU

AiTLAS examples:
- Land use classification with multi-class classification with AiTLAS: https://youtu.be/JcJXrMch0Rc
- Land use classification with multi-label classification: https://youtu.be/yzHkEMbDW7s
- Maya archeological sites segmentation: https://youtu.be/LBFY4pCfzOU
- Visualization of learning performance: https://youtu.be/wjMfstcWBSs

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

---

**Note:** You will have to download the datasets from their respective source. You can find a link for each dataset in the respective dataset class in `aitlas/datasets/` or use the **AiTLAS Semantic Data Catalog**

---
# Citation
For attribution in academic contexts, please cite our work **'AiTLAS: Artificial Intelligence Toolbox for Earth Observation'** published in Remote Sensing (2023) [link](https://www.mdpi.com/2072-4292/15/9/2343) as

```
@article{aitlas2023,
AUTHOR = {Dimitrovski, Ivica and Kitanovski, Ivan and Panov, Panče and Kostovska, Ana and Simidjievski, Nikola and Kocev, Dragi},
TITLE = {AiTLAS: Artificial Intelligence Toolbox for Earth Observation},
JOURNAL = {Remote Sensing},
VOLUME = {15},
YEAR = {2023},
NUMBER = {9},
ARTICLE-NUMBER = {2343},
ISSN = {2072-4292},
DOI = {10.3390/rs15092343}
}
```
# The AiTLAS Ecosystem

## AiTLAS: Benchmark Arena

An open-source benchmark framework for evaluating state-of-the-art deep learning approaches for image classification in Earth Observation (EO). To this end, it presents a comprehensive comparative analysis of more than 500 models derived from ten different state-of-the-art architectures and compare them to a variety of multi-class and multi-label classification tasks from 22 datasets with different sizes and properties. In addition to models trained entirely on these datasets, it employs benchmark models trained in the context of transfer learning, leveraging pre-trained model variants, as it is typically performed in practice. All presented approaches are general and can be easily extended to many other remote sensing image classification tasks.To ensure reproducibility and facilitate better usability and further developments, all of the experimental resources including the trained models, model configurations and processing details of the datasets (with their corresponding splits used for training and evaluating the models) are available on this repository.

**repo**: [https://github.com/biasvariancelabs/aitlas-arena](https://github.com/biasvariancelabs/aitlas-arena)

**paper**: [Current Trends in Deep Learning for Earth Observation: An Open-source Benchmark Arena for Image Classification](https://www.sciencedirect.com/science/article/pii/S0924271623000205) , ISPRS Journal of Photogrammetry and Remote Sensing, Vol.197, pp 18-35



## Semantic Data Catalog of Earth Observation (EO) datasets (beta)

A novel semantic data catalog of numerous EO datasets, pertaining to various different EO and ML tasks. The catalog, that includes properties of different datasets and provides further details for their use, is available [here](https://eodata.bvlabs.ai/ai4eo)
