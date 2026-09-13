# Clinical Risk Model — Live Demo

A small, self-contained Flask app that runs just the clinical branch (CatBoost + XGBoost + Logistic Regression) of the multimodal lung cancer risk pipeline, meant to be deployed on a free web-hosting tier so anyone can try it from a browser without installing anything.

This is deliberately not the full dashboard. The 2D CT, 3D-MIL, and whole-lung branches need imaging model weights that are hundreds of megabytes, well past what a free tier can hold, so they are left out here. What's included is small (under 1 MB of model files) and has no imaging dependencies, so it starts quickly and installs cleanly.

## Files

- `app.py` — Flask app: loads the three clinical models once at startup and exposes `/` (the form) and `/api/predict` (JSON scoring endpoint).
- `templates/index.html` — the single-page form and results view.
- `clinical_models_for_inference/` — the trained CatBoost, XGBoost, and logistic-regression models, their shared imputer, feature-column list, and ensemble weights.
- `requirements.txt`, `Procfile`, `render.yaml` — deployment files for Render (works the same way on most Python-friendly hosts).

## Deploying on Render (free tier)

1. Push this folder to a GitHub repository (or use it directly from this one).
2. On [render.com](https://render.com), choose **New > Web Service**, connect the repository, and point the root directory at `demo/` if you keep it nested.
3. Render should pick up `render.yaml` automatically; otherwise set the build command to `pip install -r requirements.txt` and the start command to `gunicorn app:app`.
4. First deploy takes a few minutes since it installs CatBoost and XGBoost. After that it's live at the URL Render gives you.

## Running locally

```bash
pip install -r requirements.txt
python app.py
```

Then open http://localhost:5050.

## What the demo returns

The `clinical_prob` value is the weighted ensemble output of the three clinical models — the same math as the paper's clinical branch, on the same NLST feature set. It is not the paper's validated fused score, which combines this with three CT branches at fixed weights (0.40 clinical / 0.20 2D CT / 0.30 3D-MIL / 0.10 whole-lung) and is only reported for the paper's held-out evaluation split. Treat this demo as a way to explore how the clinical model responds to different inputs, not as a stand-in for the full pipeline's reported accuracy.
