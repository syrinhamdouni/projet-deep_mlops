# FarmPulse

FarmPulse is a PlantVillage leaf-disease classifier built for the Deep Learning
Avance / MLOps project. It follows Axe 1 (CNN + transfer learning) and exposes
an ONNX model through FastAPI.

## Architecture

`data/raw` is versioned with DVC. The training pipeline creates a stratified
train/validation/test split, fine-tunes EfficientNet-B0, logs parameters and
metrics to MLflow, and exports the best checkpoint to ONNX. The FastAPI service
loads that ONNX artifact and the React client calls `/predict`.

## Run locally

Use the repository virtual environment on Windows:

```powershell
.\venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
python -m src.data.prepare
python -m src.models.train
python -m src.models.export_onnx
uvicorn backend.app.main:app --reload --port 8000
cd frontend
npm install
npm run dev
```

The dataset must be present in `data/raw`. If it is stored remotely, restore it
first with `dvc pull`. The training command can be shortened for a smoke run:

```powershell
python -m src.models.train --epochs 1 --max-images-per-class 20
```

MLflow tracking uses the SQLite database `mlflow.db`. Start its UI from the
repository root with:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Then open `http://localhost:5000`.

## Docker Compose

```powershell
docker compose up --build
```

This starts the API on `http://localhost:8000`, the React UI on
`http://localhost:3000`, and MLflow on `http://localhost:5000`. Training is
intentionally run on the host (or a GPU runner), not during API image build.

## MLOps workflow

1. Put new PlantVillage images in `data/raw`, then run `dvc add data/raw`.
2. Run `python -m src.data.prepare` and inspect the generated manifest.
3. Train the simple CNN baseline with `python -m src.models.baseline --epochs 10`.
4. Train EfficientNet-B0 with `python -m src.models.train --architecture efficientnet_b0`.
5. Train DenseNet121 with `python -m src.models.train --architecture densenet121`.
6. Evaluate each saved checkpoint on the same test split, e.g.
   `python -m src.models.evaluate --checkpoint models/densenet121_best.pt --split test`.
7. Compare recorded runs in MLflow or open `notebooks/06_model_comparison.ipynb`.
8. Export the selected checkpoint to `models/model.onnx`, then run tests and build
   the containers through GitHub Actions.

Training runs log the architecture, hyperparameters, dataset fingerprint,
training time, validation metrics, checkpoint, and metrics artifact to MLflow.
They also register the trained PyTorch model as a new MLflow Model Registry
version and assign it the `staging` alias. To promote a selected version:

```powershell
python -m src.models.promote --architecture densenet121 --version 1 --stage Production
```

The test set is for final comparison only and should not be used to select
hyperparameters. `notebooks/06_model_comparison.ipynb` displays the MLflow
comparison without changing the existing transfer-learning notebook.

The API endpoint `/monitoring/metrics` reports prediction counts, mean
confidence/latency, predicted-class counts, and a PSI-based prediction
distribution drift warning after 30 predictions. These counters are in memory
and reset when the API restarts; this is a lightweight project demonstration,
not persistent production monitoring.

The model is a multi-class classifier, so bounding boxes are not applicable to
this dataset. The UI displays the predicted disease, confidence, top-k scores,
and server/client latency.
