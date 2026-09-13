# Multimodal Lung Cancer Risk Prediction — Research Dashboard

Interactive Flask dashboard for a four-branch multimodal lung cancer risk prediction pipeline
(clinical ensemble, 2D CT CNN, 3D-MIL, and whole-lung 3D model) evaluated on the NLST cohort.

This repository contains the dashboard application code only. It does not include:

- Trained model weights (`ct_model_outputs/`, `ct_cnn_3d_mil_outputs/`, `ct_wholelung_pretrained_outputs/`) — too large for git and not required to read the code.
- Patient-level logs or drafts (`patients_log.json`, `clinical_drafts.json`) — these contain NLST participant data whose access and redistribution is governed by the National Cancer Institute Cancer Data Access System (CDAS) and The Cancer Imaging Archive (TCIA) data-use agreements.
- Raw or processed NLST/TCIA imaging and clinical data.

## Contents

- `app.py` — main Flask application and routes.
- `Data_analysis.py` — baseline cohort / Project Analysis figure generation.
- `build_model_performance_assets.py`, `recalculate_model_performance.py` — model performance asset builders.
- `predict_clinical_worker.py`, `predict_ct_worker.py`, `predict_wholelung_worker.py` — per-branch inference workers.
- `ct_dicom_io_utils.py`, `ct_segmentation_utils.py` — CT I/O and segmentation helpers.
- `patient_report.py`, `make_clinical_template.py`, `show_patient_values.py` — reporting utilities.
- `templates/`, `static/`, `dashboard.html`, `base.html`, `style.css`, `app.js` — front end.
- `dashboard_config.json` — dashboard configuration (paths, doctor profile shown in the UI).
- `requirements_dashboard.txt` — Python dependencies.

## Related work

The methodology, validation protocol, and results are described in the accompanying manuscript,
*Multimodal Lung Cancer Risk Prediction from Clinical and CT Data*. Dataset: NLST, via
[The Cancer Imaging Archive (TCIA) NLST collection](https://www.cancerimagingarchive.net/collection/nlst/).

## Setup

```bash
pip install -r requirements_dashboard.txt
python app.py
```

Model weight directories and `clinical_models_for_inference/` must be supplied locally (not included in this
repository) and referenced from `dashboard_config.json` for the app to run inference end to end.
