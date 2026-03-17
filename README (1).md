# Emotion Recognition with Deep Learning

## EfficientNet & ResNet-Based Facial Emotion Classification

------------------------------------------------------------------------

## Project Overview

This project implements a deep learning pipeline for facial emotion
recognition using transfer learning on convolutional neural networks.

The goal is to classify human facial expressions into 7 emotion
categories and analyze how different architectures perform on this task.

Additionally, the project introduces a mapped satisfaction
classification layer, converting emotions into higher-level sentiment
categories.

------------------------------------------------------------------------

## Objectives

-   Train deep learning models for facial emotion classification
-   Compare performance across multiple architectures:
    -   EfficientNet-B0
    -   EfficientNet-B3
    -   ResNet-50
-   Apply data augmentation techniques to improve generalization
-   Evaluate models using classification metrics
-   Map emotion predictions to customer satisfaction categories

------------------------------------------------------------------------

## Models Used

  Model             Description
  ----------------- --------------------------------------
  EfficientNet-B0   Lightweight baseline model
  EfficientNet-B3   Higher-capacity EfficientNet variant
  ResNet-50         Deep residual network

All models are initialized with ImageNet pretrained weights and
fine-tuned for 7-class emotion classification.

------------------------------------------------------------------------

## Emotion Classes

\["angry", "sad", "disgust", "fear", "happy", "neutral", "surprise"\]

------------------------------------------------------------------------

## Satisfaction Mapping

  Emotion    Satisfaction
  ---------- --------------
  angry      dissatisfied
  sad        dissatisfied
  disgust    dissatisfied
  fear       dissatisfied
  happy      satisfied
  surprise   satisfied
  neutral    neutral

------------------------------------------------------------------------

## Repository Structure

Emotion-Recognition/ │ ├── run.py ├── train.py ├── models.py ├──
settings.py ├── util.py ├── logger.py ├── environment.yml ├──
minimal_environment.yml └── README.md

------------------------------------------------------------------------

## Dataset Note

Due to size constraints, the dataset is not included in this repository.

This project automatically downloads the dataset using:

kagglehub.dataset_download("fahadullaha/facial-emotion-recognition-dataset")

### Setup Steps

1.  Install Kaggle API credentials: \~/.kaggle/kaggle.json

2.  The script will:

    -   Download dataset
    -   Copy processed images into ./images/

3.  Ensure dataset path:

DATASET_PATH = "./images"

------------------------------------------------------------------------

## Installation

### Clone Repository

git clone `<your-repo-url>`{=html} cd Emotion-Recognition

### Create Environment

conda env create -f environment.yml conda activate emotion-recognition

### Install KaggleHub

pip install kagglehub

------------------------------------------------------------------------

## How to Run

python run.py

------------------------------------------------------------------------

## Execution Pipeline

1.  Download dataset
2.  Apply data augmentation
3.  Split dataset (80/10/10)
4.  Train models (batch sizes 16, 32, 64)
5.  Evaluate with accuracy and classification report

------------------------------------------------------------------------

## Outputs

-   Training logs
-   Accuracy metrics
-   Classification reports

------------------------------------------------------------------------

## Reproducibility

Use: - batch sizes \[16, 32, 64\] - epochs = 30

Run:

python run.py

------------------------------------------------------------------------

## Future Improvements

-   Confusion matrix visualization
-   Hyperparameter tuning
-   Attention-based models
-   Real-time emotion detection

------------------------------------------------------------------------

## Technologies

-   Python
-   PyTorch
-   Torchvision
-   Scikit-learn
-   KaggleHub

------------------------------------------------------------------------

## Author

Florence Lourdes\
MS in Artificial Intelligence & Machine Learning\
Drexel University

------------------------------------------------------------------------

## Notes

This project was developed for academic and research purposes.
