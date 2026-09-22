# JalSaafi - Oil-Spill Detection (Sentinel-1 SAR)

Binary classifier (oil / no-oil) for SAR image chips, built with transfer
learning (ResNet18). This is the detection component of the larger JalSaafi
system (HKU AI+ Challenge).

## Status

Proof of concept. A ResNet18 was fine-tuned on the CSIRO dataset below and
evaluated on a held-out, stratified test split (831 chips): balanced accuracy
0.951, ROC-AUC 0.987 (see `outputs/metrics_report.txt`). Training was stopped
manually after epoch 8 of 15 because of time, so the saved checkpoint is epoch
7. Trained weights are not included in this repo - regenerate them with
`scripts/train.py`. These numbers are in-dataset only; see the honesty notes
below.

## Setup

```
py -3.13 -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
```
On macOS/Linux use `python3 -m venv venv` and `venv/bin/python`.

## Dataset

Not included in this repo. Download the CSIRO dataset and place it so the
layout is:
```
kaggle/data/
    Class_0/   (no oil)
    Class_1/   (oil)
```
or point `JALSAAFI_RAW_DATA_DIR` at wherever it lives (read in
[config.py](config.py)). Splits, models, plots and reports are written inside
this project folder.

**Dataset:** "CSIRO Sentinel-1 SAR image dataset of oil- and non-oil features
for machine learning (Deep Learning)", David Blondeau-Patissier, CSIRO, 2022.
DOI: [10.25919/4v55-dn16](https://doi.org/10.25919/4v55-dn16). Licence:
CC BY-SA 4.0 - attribution and share-alike apply to the data and anything
derived from it.

## Pipeline

Run these in order, from the project root, using the venv's Python:

1. **Inspect the data** (counts, dimensions, corrupt files, class balance):
   ```
   venv\Scripts\python.exe scripts\inspect_dataset.py
   ```
   Report saved to `outputs\dataset_inspection_report.txt`.

2. **Build stratified train/val/test splits** (70/15/15):
   ```
   venv\Scripts\python.exe scripts\prepare_splits.py
   ```
   Writes `data\splits\{train,val,test}.csv`. Will error out if a class is
   missing - that's intentional, not a bug.

3. **Train**:
   ```
   venv\Scripts\python.exe scripts\train.py
   ```
   - Fine-tunes ImageNet-pretrained ResNet18.
   - Uses class-weighted loss to handle imbalance.
   - Early stops on validation balanced accuracy.
   - Saves: `models\oil_spill_resnet18.pt`, `outputs\training_curves.png`,
     `outputs\confusion_matrix.png`, `outputs\metrics_report.txt`.

4. **Predict on a single image**:
   ```
   venv\Scripts\python.exe predict.py path\to\image.jpg
   ```
   Or import it: `from predict import predict_image, locate_oil_region`.

## Honesty notes (read before putting numbers in a slide deck)

- This is a proof-of-concept classifier trained on a few thousand chips,
  not a validated operational system.
- SAR oil detection is confounded by **look-alikes**: low-wind zones, algae
  films, and wave shadows all create dark patches that mimic oil's
  low-backscatter signature. A model trained on a curated dataset like this
  one has not been tested against those confounders and likely performs
  worse on them than the headline test-set accuracy suggests.
- Always report **balanced accuracy** and **per-class precision/recall**
  alongside overall accuracy - with class imbalance, overall accuracy alone
  is misleading (a model that always predicts the majority class can score
  high).
- `locate_oil_region()` in `predict.py` is a simple Otsu-thresholding
  heuristic (darkest region's centroid), not a trained segmentation model.
  It's explicitly labeled "approximate" in its output - don't present it as
  precise localization.

## Channel handling

Sentinel-1 SAR is physically single-channel (backscatter intensity), but
ResNet18's pretrained weights expect 3-channel input. Every image is
converted to grayscale then replicated to 3 identical channels
(`scripts/dataset.py: build_transforms`) - this keeps the pretrained
weights fully usable without surgically modifying the first conv layer.

## Licence

Code: MIT (see [LICENSE](LICENSE)). The dataset has its own licence, above.
