## What's new

### New foundation models & adapters
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

### New model architectures
- **Change detection**: Added [BIT](https://github.com/justchenhao/BIT_CD), [CGNet](https://github.com/wutianyiRosun/CGNet), [ChangeFormer V6](https://github.com/wgcban/ChangeFormer), [ChangeMamba](https://github.com/ChenHongruixuan/ChangeMamba), [ChangeVIT](https://github.com/zhuduowang/ChangeViT), [CSSM](https://github.com/Elman295/CSSM), HRNet SiamConc, [SiamCRNN](https://github.com/ChenHongruixuan/SiamCRNN), [STANet](https://github.com/justchenhao/STANet), [TinyCD](https://github.com/AndreaCodegoni/Tiny_model_4_CD), and U-Net SiamConc.
- **Object detection**: [ATSS](https://github.com/sfzhang15/ATSS), [CenterNet](https://github.com/xingyizhou/CenterNet), [DETR](https://github.com/facebookresearch/detr), [EfficientDet](https://github.com/signatrix/efficientdet), [NanoDet-Plus](https://github.com/RangiLyu/nanodet), and [Sparse R-CNN](https://github.com/PeizeSun/SparseR-CNN).
- **Segmentation**: Added [FPN](https://github.com/qubvel/segmentation_models.pytorch), [MaNet](https://github.com/qubvel/segmentation_models.pytorch), [PSPNet](https://github.com/hszhao/PSPNet), [SegFormer](https://github.com/NVlabs/SegFormer), [UNet++](https://github.com/qubvel/segmentation_models.pytorch), and [UPerNet](https://github.com/qubvel/segmentation_models.pytorch).

### Key improvements & features
- **New build system**: Complete migration to `uv` and `pyproject.toml` for faster, reproducible builds.
- **Modern infrastructure**: Switched to `ruff` for ultra-fast linting and formatting.
- **Foundation model architecture**: Implemented `CompositeModel`, allowing for dynamic building of backbones, necks, decoders, heads, and data-model adapters.
- **Training**: Added Automatic Mixed Precision (AMP), early stopping on NaN loss, and state preservation (LR scheduler/checkpoints) for restarts.
- **Adapters**: Implemented specific data-model adapters for foundation models, such as Terramind, AnySat, Galileo, and Panopticon.
- **Examples**: Added Jupyter notebook examples for new foundation models and downstream tasks (e.g., change detection).

### Breaking changes
- Minimum Python version is now **3.12**.
- Removed `requirements.txt` in favor of `pyproject.toml` dependencies.
- Namespaced foundation model classes in `aitlas.models` to resolve implementation collisions.

**Full Changelog**: https://github.com/biasvariancelabs/aitlas/compare/v1.0.0...v2.0.0
