# RetCEVD

Official PyTorch implementation of **RetCEVD**, introduced in:

> Shijia Zhou, Le Cao, Lei Mou, et al. **Memory-guided mutual information optimization in retinal imaging for early stroke detection and subtyping**. *Medical Image Analysis*, 2026, 104382.
> [https://doi.org/10.1016/j.media.2026.104382](https://doi.org/10.1016/j.media.2026.104382)

RetCEVD integrates depth-resolved optical coherence tomography angiography (OCTA) images with structured clinical and quantitative OCTA features. Class-wise memory banks support intra-class consistency and cross-class redundancy regularization through InfoNCE and a pairwise CLUB objective.

The framework addresses three tasks:

1. **CEVD-Det:** auxiliary detection of ischemic cerebrovascular disease (ICVD).
2. **CEVD-Sub:** etiological subtyping of small vessel disease (SVD) and large artery atherosclerosis (LAA).
3. **CEVD-Ris:** current symptomatic-status classification of asymptomatic LAA (ALAA) and symptomatic LAA (SLAA).

## Repository structure

```text
RetCEVD/
├── README.md
├── requirements.txt
├── Code/
│   ├── main.py
│   ├── dataset.py
│   ├── model.py
│   ├── loss.py
│   ├── metrics.py
│   └── utils.py
└── Data/
    └── example_eye/
```

## Installation

Use Python 3.9 or later. An NVIDIA GPU with a compatible CUDA-enabled PyTorch installation is recommended for training.

```bash
git clone https://github.com/iMED-Lab/RetCEVD.git
cd RetCEVD
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Install a matching PyTorch and torchvision build for your hardware using the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/).

## Data preparation

### OCTA images

Each sample corresponds to one eye and contains three grayscale en face images: the superficial vascular complex (SVC), deep vascular complex (DVC), and choriocapillaris (CC). These images are stacked as a three-channel input in SVC–DVC–CC order.

Organize the images for each eye as follows:

```text
images/
└── participant_001_OD/
    ├── scan_浅层血管复合体.png
    ├── scan_深层血管复合体.png
    └── scan_脉络膜毛细血管层.png
```

Use one image per layer at a consistent spatial resolution. Filename suffixes are configured in `OCTATabularDataset.LAYER_SPECS` in `Code/dataset.py`.

A synthetic, single-eye image example is provided in [`Data/example_eye/`](Data/example_eye/), with one grayscale image for each layer. These artificial patterns illustrate the image format and contain no patient data.

### Split files and tabular inputs

Prepare UTF-8, tab-separated files without a header:

```text
splits/
├── cevd_det/
│   ├── train_fold0.tsv
│   ├── test_fold0.tsv
│   └── ...
├── cevd_sub/
│   ├── train_fold0.tsv
│   ├── test_fold0.tsv
│   └── ...
└── cevd_ris/
    ├── train_fold0.tsv
    ├── test_fold0.tsv
    └── ...
```

Each row contains 72 columns: an eye-image directory, a binary label, and 70 tabular values. Relative image directories are resolved against `--image-root`. The loader selects 69 model inputs using `FULL_NO_DEVICE_INDICES`: seven demographic/clinical variables and 62 quantitative OCTA measurements.

Use labels `0/1` for Control/ICVD, SVD/LAA, and ALAA/SLAA in the three tasks, respectively. Partition data by participant, keeping both eyes and repeated acquisitions together.

## Training

Run all three tasks across five folds:

```bash
python Code/main.py \
  --split-root /path/to/splits \
  --image-root /path/to/images \
  --output-dir ./outputs \
  --tasks cevd_det cevd_sub cevd_ris \
  --folds 0 1 2 3 4 \
  --device cuda:0
```

To run a single task and fold, use `--tasks cevd_det --folds 0`. Use `python Code/main.py --help` to inspect all options.

The checkpoint with the highest validation AUC is saved for each fold. Training results are summarized in `summary.json`.

## Evaluation

During training, `test_fold*.tsv` provides the held-out fold for validation and checkpoint selection.

`Code/metrics.py` calculates AUC, accuracy, recall, precision, F1-score, and Cohen's kappa at the input-record level. Patient-level evaluation uses the mean probability across available eyes for each participant.

## Citation

If you find this work useful, please cite:

```bibtex
@article{zhou2026retcevd,
  title={Memory-guided mutual information optimization in retinal imaging for early stroke detection and subtyping},
  author={Zhou, Shijia and Cao, Le and Mou, Lei and Wang, Hang and Hao, Jinkui and Zhang, Xuening and Li, Hongshuo and Yuan, Leilei and Liao, Weicheng and Julian, Thomas H. and Wang, Meng and Liu, Yonghuai and Kwapong, William Robert and Meng, Yanda and Xie, Jianyang and Zheng, Yalin and Fu, Huazhu and Liu, Jiang and Sheng, Bin and Frangi, Alejandro F. and Wu, Bo and Zhao, Yitian},
  journal={Medical Image Analysis},
  pages={104382},
  year={2026},
  doi={10.1016/j.media.2026.104382}
}
```

## Acknowledgements

We thank the developers of [PyTorch](https://pytorch.org/), [torchvision](https://github.com/pytorch/vision), [Albumentations](https://github.com/albumentations-team/albumentations), and [scikit-learn](https://scikit-learn.org/) for their open-source tools.

## Contact

For questions, please contact [Shijia Zhou](mailto:zhoushijia@nimte.ac.cn) or open an issue in this repository.
