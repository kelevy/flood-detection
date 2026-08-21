# Flood Detection from Satellite Imagery using Deep Learning


An end-to-end deep learning pipeline for flood water segmentation from Sentinel-1
SAR imagery, trained on Google Cloud (Vertex AI) and deployed as a live API on
Cloud Run.

## Motivation

Detecting flooding quickly is critical for disaster response, but the moments
when floods occur are often exactly when optical satellite imagery is least
useful since heavy cloud cover during storms blocks the view entirely.

Synthetic Aperture Radar (SAR) imagery solves this: it penetrates cloud cover
and works day or night. The tradeoff is that SAR images are difficult for the
human eye to interpret directly, as water appears as a subtle, dark, low-texture
region within otherwise noisy radar backscatter, not as an obvious visual feature.

This project trains a deep learning model (U-Net) to automatically identify
water extent from raw Sentinel-1 SAR imagery, extracting information that would
otherwise require expert interpretation, and deploys it as a live, callable API.

**Scope note:** The model performs per-image water segmentation (identifying
standing water in a single SAR image), not change detection. Distinguishing
"flooding" specifically (water in an abnormal location) from a permanent water
body would require comparing multiple dates, a direction partially explored in
the climate correlation analysis (`notebooks/03_climate_analysis.ipynb`).

## Live API

The trained model is deployed on Google Cloud Run:

**Endpoint:** `https://flood-detection-api-339736685737.europe-west1.run.app/predict`

Try it with a sample chip (see `samples/` for details):

```bash
curl -X POST "https://flood-detection-api-339736685737.europe-west1.run.app/predict" \
     -F "file=@samples/Bolivia_290290_S1Hand.tif" \
     --output prediction.tif
```

Returns a single-band GeoTIFF (0: not water, 1: water) with georeferencing
matching the input, viewable in QGIS or any GIS tool.

**Note:** The API may take 10-30s to respond on the first request after a
period of inactivity (Cloud Run scales the container to zero when idle).

## Results

Evaluated on a held-out validation split (89 chips, 20% of the hand-labeled
dataset), after 15 epochs of training:

| Metric | Water | Not-water |
|---|---|---|
| IoU | 0.63 | 0.95 |
| Precision | 0.76 | 0.98 |
| Recall | 0.79 | 0.97 |
| F1 | 0.78 | 0.97 |

**Mean IoU: 0.79**

The model performs very well on large, contiguous water bodies (main river
channels, large flood extents) but consistently under-detects smaller,
fragmented water patches, a known limitation of U-Net-style architectures,
where repeated downsampling in the encoder loses fine spatial detail. See
`notebooks/02_model_evaluation.ipynb` for full analysis and prediction examples.


## Climate Correlation Analysis

A secondary analysis (`notebooks/03_climate_analysis.ipynb`) tests whether
antecedent precipitation (ERA5 reanalysis, via the Open-Meteo API) correlates
with detected flood extent across the 10 events with hand-labeled data.

Max single-day precipitation showed a moderate positive correlation with flood
extent (Pearson r = 0.61, p = 0.064), suggestive but not statistically
significant given the small sample (n=10). This is an exploratory, single-snapshot
analysis, not a time-series trend study.

## Stack

- **Model:** U-Net (PyTorch, `segmentation-models-pytorch`, ResNet34 encoder)
- **Data:** [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) (Bonafilia et al., 2020), Sentinel-1 SAR imagery, hand-labeled flood masks
- **Training:** Google Cloud Vertex AI (custom training job)
- **Serving:** FastAPI, containerized with Docker, deployed on Google Cloud Run
- **Storage:** Google Cloud Storage (data, model checkpoints)
- **Climate data:** ERA5 reanalysis via Open-Meteo Historical Weather API

## Project structure

```
flood-detection/
├── data/ # Sen1Floods11 dataset (not committed)
├── models/ # trained checkpoint (not committed)
├── notebooks/
│ ├── 01_eda.ipynb # dataset exploration, class balance
│ ├── 02_model_evaluation.ipynb # metrics, predictions, error analysis
│ └── 03_climate_analysis.ipynb # precipitation vs flood extent correlation
├── src/
│ ├── dataset.py # PyTorch Dataset for Sen1Floods11
│ ├── model.py # U-Net architecture
│ ├── train.py # local training loop
│ ├── train_vertex.py # Vertex AI training entrypoint (GCS I/O)
│ └── evaluate.py # evaluation metrics
├── api/
│ ├── app.py # FastAPI inference app
│ └── requirements.txt
├── samples/ # example SAR chips for testing the live API
├── Dockerfile
├── vertex_job_config.yaml # Vertex AI custom training job config
├── setup.py # packaging for Vertex AI training
└── requirements.txt
```

## How to run

### 1. Set up the environment
```bash
conda create -n flood-detection python=3.10
conda activate flood-detection
pip install -r requirements.txt
```

### 2. Get the data
```bash
mkdir -p data/sen1floods11
gsutil -m rsync -r gs://sen1floods11 data/sen1floods11
```

### 3. Train locally
```bash
cd src
python train.py
```

### 4. Train on Vertex AI
```bash
python setup.py sdist --formats=gztar
gsutil cp dist/flood_detection_trainer-0.1.tar.gz gs://YOUR_BUCKET/packages/
gcloud ai custom-jobs create \
  --region=europe-west1 \
  --display-name=flood-detection-training \
  --config=vertex_job_config.yaml
```

### 5. Evaluate
```bash
cd src
python evaluate.py --checkpoint ../models/best_model.pt --force-cpu
```

### 6. Run the API locally
```bash
cd api
uvicorn app:app --reload
```

### 7. Deploy to Cloud Run
```bash
docker build --platform linux/amd64 -t flood-detection-api .
docker tag flood-detection-api REGION-docker.pkg.dev/PROJECT/REPO/flood-detection-api:latest
docker push REGION-docker.pkg.dev/PROJECT/REPO/flood-detection-api:latest
gcloud run deploy flood-detection-api \
  --image=REGION-docker.pkg.dev/PROJECT/REPO/flood-detection-api:latest \
  --region=europe-west1 --memory=4Gi --cpu=2 --allow-unauthenticated
```

## Known Limitations

- Water segmentation, not flood/change detection (see Scope note above)
- Under-detects small, fragmented water patches (see Results)
- Individual chip predictions vary in quality; aggregate metrics are a better
  indicator of real-world performance than any single example
- Climate correlation analysis is exploratory (n=10), not confirmatory
- Cold-start latency on the deployed API after periods of inactivity