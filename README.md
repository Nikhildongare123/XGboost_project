# XGboost_project
# 🏗️ Concrete Compressive Strength Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.0-orange.svg)](https://xgboost.readthedocs.io/)

## 📌 Overview

An interactive web application that predicts the compressive strength of concrete based on mixture proportions using a trained XGBoost machine learning model. This tool helps civil engineers, construction professionals, and researchers estimate concrete strength without waiting for traditional 28-day testing.

## ✨ Features

- **Real-time Predictions**: Instant strength prediction as you adjust mixture parameters
- **Interactive Input Controls**: User-friendly sliders and number inputs for all 8 concrete components
- **Strength Categorization**: Automatic classification into strength levels with visual indicators
- **Data Visualization**: 
  - Strength distribution histograms
  - Feature correlation heatmaps
  - Similar mixtures analysis
- **File Upload Support**: Upload custom models and datasets
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Professional UI**: Clean, modern interface with custom styling

## 🎯 Input Parameters

| Parameter | Description | Typical Range (kg/m³) |
|-----------|-------------|----------------------|
| Cement | Cement content | 100 - 550 |
| Blast Furnace Slag | Slag content | 0 - 360 |
| Fly Ash | Fly ash content | 0 - 200 |
| Water | Water content | 120 - 230 |
| Superplasticizer | Superplasticizer content | 0 - 35 |
| Coarse Aggregate | Coarse aggregate content | 800 - 1150 |
| Fine Aggregate | Fine aggregate content | 600 - 1000 |
| Age | Concrete age (days) | 1 - 365 |

## 📊 Output

- **Predicted Compressive Strength** (MPa)
- **Strength Category** (Low/Medium/High/Very High)
- **Percentile Ranking** compared to dataset
- **Similar Mixtures** from training data

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the repository
```bash
git clone https://github.com/your-username/concrete-strength-predictor.git
cd concrete-strength-predictor
