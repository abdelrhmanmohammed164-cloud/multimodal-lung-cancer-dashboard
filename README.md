# Multimodal Lung Cancer Risk Prediction — Research Dashboard

An interactive Flask dashboard for a four-branch multimodal lung cancer risk prediction pipeline, combining a clinical ensemble, a 2D CT CNN, a 3D multiple-instance-learning (3D-MIL) model, and a whole-lung 3D model, evaluated on the NLST cohort.

This repository holds the dashboard's application code. A few things are deliberately left out:

- Trained model weights (`ct_model_outputs/`, `ct_cnn_3d_mil_outputs/`, `ct_wholelung_pretrained_outputs/`), since they're too large for git and aren't needed to read or review the code.
- Patient-level logs and drafts (`patients_log.json`, `clinical_drafts.json`), which contain NLST participant data. Access to and redistribution of that data is governed by the National Cancer Institute's Cancer Data Access System (CDAS) and by The Cancer Imaging Archive (TCIA) data-use agreements.
- Raw or processed NLST/TCIA imaging and clinical data.

## What's in here

- `app.py` — the main Flask application and its routes.
- `Data_analysis.py` — generates the baseline cohort and Project Analysis figures.
- `build_model_performance_assets.py`, `recalculate_model_performance.py` — build the model performance assets shown in the dashboard.
- `predict_clinical_worker.py`, `predict_ct_worker.py`, `predict_wholelung_worker.py` — inference workers, one per branch.
- `ct_dicom_io_utils.py`, `ct_segmentation_utils.py` — CT I/O and lung/nodule segmentation helpers.
- `patient_report.py`, `make_clinical_template.py`, `show_patient_values.py` — reporting utilities.
- `templates/`, `static/`, `dashboard.html`, `base.html`, `style.css`, `app.js` — the front end.
- `dashboard_config.json` — configuration for the dashboard (paths, and the clinician profile shown in the UI).
- `requirements_dashboard.txt` — Python dependencies.

## Related work

The methodology, validation protocol, and results behind this dashboard are described in the accompanying manuscript, *Multimodal Lung Cancer Risk Prediction from Clinical and CT Data*. The underlying dataset is NLST, accessed through [The Cancer Imaging Archive (TCIA) NLST collection](https://www.cancerimagingarchive.net/collection/nlst/).

## Setup

```bash
pip install -r requirements_dashboard.txt
python app.py
```

To run inference end to end, you'll also need the model weight directories and `clinical_models_for_inference/` in place locally (not included here) and pointed to from `dashboard_config.json`.
