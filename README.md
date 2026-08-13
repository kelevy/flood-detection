# Flood Detection from Satellite Imagery using Deep Learning

## Overview
An end-to-end deep learning pipeline for flood extent mapping using 
Sentinel-2 multispectral satellite imagery. A U-Net model is trained 
on the Sen1Floods11 dataset and deployed as a REST API on Google Cloud 
Platform.

A secondary analysis correlates detected flood extent with ERA5 climate 
indicators (ENSO, regional temperature anomalies) to investigate whether 
flood frequency is increasing over the 2017-2024 period.

## Motivation
Flood detection from satellite imagery is a critical tool for disaster 
response and climate change monitoring. This project demonstrates an 
operational ML pipeline combining state-of-the-art deep learning with 
scalable cloud infrastructure.

## Stack
- **Model:** U-Net (PyTorch, segmentation-models-pytorch)
- **Data:** Sen1Floods11 (Sentinel-2 multispectral imagery)
- **Cloud:** GCP — GCS (data), Vertex AI (training), Cloud Run (API)
- **Climate analysis:** ERA5 reanalysis data, xarray, scipy

## Project Structure
TBD

## Results
TBD
