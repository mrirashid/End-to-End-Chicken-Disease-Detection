# End-to-End Chicken Disease Detection

A complete end-to-end deep learning project for classifying chicken fecal images into two categories: Healthy and Coccidiosis. The project uses transfer learning with the VGG16 convolutional neural network and is served through a Flask web application with a dark-themed user interface.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [Pipeline Stages](#pipeline-stages)
- [Configuration](#configuration)
- [Installation](#installation)
- [Usage](#usage)
- [Web Application](#web-application)
- [Technologies Used](#technologies-used)
- [License](#license)
- [Author](#author)

---

## Overview

Coccidiosis is a parasitic disease that affects the intestinal tract of poultry and can lead to significant economic losses in the poultry industry. Early and accurate detection of this disease is critical for effective treatment and prevention.

This project provides an automated solution by analyzing chicken fecal images using a deep learning model built on the VGG16 architecture pretrained on ImageNet. The entire workflow from data preparation to model deployment is implemented as a modular and reproducible pipeline.

---

## Dataset

The dataset consists of 390 chicken fecal images divided equally into two classes:

| Class       | Number of Images | Description                                   |
| ----------- | ---------------- | --------------------------------------------- |
| Coccidiosis | 195              | Images showing signs of coccidiosis infection |
| Healthy     | 195              | Images from healthy chickens                  |

The images are stored in the `Chicken-fecal-images/` directory with separate subfolders for each class. During training, the data is split into 80% training and 20% validation sets. During evaluation, a 70/30 split is used.

---

## Model Architecture

The model is built using transfer learning with VGG16 as the base network.

**Base Model:** VGG16 pretrained on ImageNet with the top classification layers removed.

**Custom Head:**

- Flatten layer applied to the VGG16 output (output shape: 25088)
- Dense layer with 2 units and softmax activation for binary classification

**Training Configuration:**

- Input image size: 224 x 224 x 3
- Optimizer: Stochastic Gradient Descent (SGD) with learning rate 0.01
- Loss function: Categorical Crossentropy
- All VGG16 base layers are frozen during training
- Data augmentation is applied including rotation, horizontal flip, width and height shift, shear, and zoom

**Model Summary:**

- Total parameters: 14,764,866 (56.32 MB)
- Trainable parameters: 50,178 (196.01 KB)
- Non-trainable parameters: 14,714,688 (56.13 MB)

---

## Project Structure

```
End-to-End-Chicken-Disease-Detection/
|
|-- config/
|   |-- config.yaml                          # Central configuration file
|
|-- src/
|   |-- ChickenDiseaseClassification/
|       |-- __init__.py                      # Logger setup
|       |-- components/
|       |   |-- prepare_base_model.py        # VGG16 base model preparation
|       |   |-- model_trainer.py             # Model training with data augmentation
|       |   |-- model_evaluation.py          # Model evaluation and score saving
|       |-- config/
|       |   |-- configuration.py             # ConfigurationManager class
|       |-- constants/
|       |   |-- __init__.py                  # File path constants
|       |-- entity/
|       |   |-- config_entity.py             # Dataclass definitions for configs
|       |-- pipeline/
|       |   |-- state_02_prepare_base_model.py
|       |   |-- state_03_model_trainer.py
|       |   |-- state_04_model_evaluation.py
|       |   |-- predict.py                   # Prediction pipeline for inference
|       |-- utils/
|           |-- common.py                    # Utility functions
|
|-- templates/
|   |-- index.html                           # Web application frontend
|
|-- Chicken-fecal-images/
|   |-- Coccidiosis/                         # 195 coccidiosis images
|   |-- Healthy/                             # 195 healthy images
|
|-- research/
|   |-- prepare_base_model.ipynb             # Research notebook for base model
|   |-- trials.ipynb                         # Experimentation notebook
|
|-- artifacts/                               # Generated during pipeline execution
|   |-- prepared_base_model/                 # Base and updated model files
|   |-- training/                            # Trained model (model.h5)
|   |-- evaluation/                          # Evaluation artifacts
|
|-- logs/                                    # Application logs
|-- app.py                                   # Flask web application entry point
|-- main.py                                  # Pipeline orchestration script
|-- params.yaml                              # Hyperparameters
|-- config/config.yaml                       # Project configuration
|-- requirements.txt                         # Python dependencies
|-- setup.py                                 # Package setup
|-- template.py                              # Project template generator
|-- LICENSE                                  # MIT License
|-- README.md                                # Project documentation
```

---

## Pipeline Stages

The project follows a modular pipeline architecture with three main stages executed sequentially through `main.py`.

### Stage 1: Prepare Base Model

- Loads the VGG16 model pretrained on ImageNet without the top classification layers.
- Adds a custom Flatten layer and a Dense output layer with 2 units.
- Freezes all VGG16 base layers to retain pretrained features.
- Compiles the model with SGD optimizer and categorical crossentropy loss.
- Saves both the base model and the updated model to the artifacts directory.

### Stage 2: Model Training

- Loads the updated base model from the previous stage.
- Applies data augmentation (rotation, horizontal flip, width shift, height shift, shear, zoom) on the training set.
- Rescales pixel values to the range 0 to 1.
- Trains the model using an 80/20 train-validation split.
- Saves the trained model as `artifacts/training/model.h5`.

### Stage 3: Model Evaluation

- Loads the trained model from the training artifacts.
- Evaluates the model on a 30% validation split of the dataset.
- Computes loss and accuracy metrics.
- Saves the evaluation scores to `scores.json` in the project root.

---

## Configuration

### config/config.yaml

Defines all directory paths and file paths for each pipeline stage:

```yaml
artifacts_root: artifacts

data_ingestion:
  unzip_dir: artifacts/data_ingestion

preparare_base_model:
  root_dir: artifacts/prepared_base_model
  base_model_path: artifacts/prepared_base_model/base_model.h5
  updated_base_model_path: artifacts/prepared_base_model/updated_base_model.h5

training:
  root_dir: artifacts/training
  trained_model_path: artifacts/training/model.h5

evaluation:
  root_dir: artifacts/evaluation
  path_of_model: artifacts/training/model.h5
```

### params.yaml

Defines all model hyperparameters:

```yaml
AUGMENTATION: True
IMAGE_SIZE: [224, 224, 3]
BATCH_SIZE: 16
INCLUDE_TOP: False
EPOCHS: 1
CLASSES: 2
LEARNING_RATE: 0.01
WEIGHTS: imagenet
```

---

## Installation

### Prerequisites

- Python 3.8
- Conda (recommended) or pip
- macOS, Linux, or Windows

### Steps

1. Clone the repository:

```bash
git clone https://github.com/mrirashid/End-to-End-Chicken-Disease-Detection.git
cd End-to-End-Chicken-Disease-Detection
```

2. Create and activate a conda environment:

```bash
conda create -n classification python=3.8 -y
conda activate classification
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

This will install all dependencies including TensorFlow, Flask, and the project package in editable mode.

---

## Usage

### Run the Full Training Pipeline

Execute all three pipeline stages (prepare base model, training, evaluation) sequentially:

```bash
python main.py
```

This will:

- Prepare the VGG16 base model and save it to `artifacts/prepared_base_model/`
- Train the model on the chicken fecal images and save it to `artifacts/training/model.h5`
- Evaluate the trained model and save scores to `scores.json`

### Run the Web Application

Start the Flask web server:

```bash
python app.py
```

The application will be available at `http://localhost:8080`.

---

## Web Application

The web application provides a dark-themed user interface with the following features:

### Prediction

- Upload a chicken fecal image using the file upload area or drag and drop.
- Click the Analyze Image button to classify the image.
- The model returns one of two results: Healthy or Coccidiosis.
- Results are displayed with color-coded indicators.

### Training

- Click the Train Model button to retrain the model directly from the web interface.
- The training process runs all three pipeline stages and updates the model artifacts.

### API Endpoints

| Method   | Endpoint | Description                                               |
| -------- | -------- | --------------------------------------------------------- |
| GET      | /        | Serves the web application frontend                       |
| GET/POST | /train   | Triggers the full training pipeline                       |
| POST     | /predict | Accepts a base64-encoded image and returns the prediction |

**Prediction Request Format:**

```json
{
  "image": "<base64-encoded-image-string>"
}
```

**Prediction Response Format:**

```json
[
  {
    "image": "Healthy"
  }
]
```

---

## Technologies Used

| Category                 | Technology                |
| ------------------------ | ------------------------- |
| Programming Language     | Python 3.8                |
| Deep Learning Framework  | TensorFlow 2.13 and Keras |
| Pretrained Model         | VGG16 (ImageNet weights)  |
| Web Framework            | Flask with Flask-CORS     |
| Configuration Management | PyYAML and python-box     |
| Data Validation          | ensure                    |
| Image Processing         | Pillow and scipy          |
| Serialization            | joblib                    |
| Frontend                 | HTML, CSS, JavaScript     |
| Package Management       | Conda and pip             |
| Version Control          | Git and GitHub            |

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


