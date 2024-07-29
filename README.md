# Image Classification Project

## Overview

This project aims to classify images into different categories using a Convolutional Neural Network (CNN). The script provided trains a neural network model on a dataset of images, evaluates its performance, and saves the trained model for future use.

## Project Structure

- `image_classification_solid_debris.py`: Python script that loads and preprocesses image data, creates and trains a CNN model, evaluates its performance, and saves the model.
- `requirements.txt`: A file listing all necessary dependencies for running the project.

## Setup Instructions

### Prerequisites

Ensure you have Python 3.6 or higher installed on your machine. The required libraries are listed in `requirements.txt`.

### Installing Dependencies

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

### Directory Structure

Your project directory should be structured as follows:

```
project_root/
│
├── image_classification_solid_debris.py
├── requirements.txt
└── data/
    ├── class1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── class2/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── class3/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
```

- Replace `class1`, `class2`, `class3` with the actual class names.
- Ensure the images are in `.jpg` format.

## Usage

### Running the Script

To train the model and save it, run:

```bash
python image_classification_solid_debris.py
```

### Script Breakdown

1. **Data Loading and Preprocessing**:
   - The script loads images from the specified directory and preprocesses them into the required format.

2. **Data Splitting**:
   - The data is split into training and testing sets.

3. **Model Creation**:
   - A Convolutional Neural Network (CNN) model is created using TensorFlow/Keras.

4. **Model Training**:
   - The model is trained on the training data.

5. **Model Evaluation**:
   - The model is evaluated on the testing data, and accuracy is printed.

6. **Model Saving**:
   - The trained model is saved as `image_classification_model.h5`.

## Dependencies

The project requires the following libraries, as specified in `requirements.txt`:

- numpy
- tensorflow
- scikit-learn

To install all dependencies, run:

```bash
pip install -r requirements.txt
```

## Acknowledgments

This project uses TensorFlow and Keras for creating and training the neural network model. The data preprocessing and splitting are handled using NumPy and scikit-learn.

