# Character Recognition using EMNIST

This project implements various machine learning and deep learning models for character recognition using the **EMNIST (Extended MNIST)** dataset. The models are trained on the `balanced` split of the EMNIST dataset, which contains 47 different classes of characters (letters and digits).

## Dataset

- **EMNIST ('balanced' split)**: Contains 47 classes.
- The data is loaded and transformed (rotated and horizontally flipped) to match the correct orientation of characters before being converted to tensors and normalized.

## Implemented Models

The project is structured into different directories, each containing implementations and training scripts for a specific type of model:

### 1. Convolutional Neural Networks (CNN)
Located in the `CNN/` directory.
- `train.py`: Implements a custom **ResNet** architecture in PyTorch specifically designed for the EMNIST dataset. It includes data augmentation, L1/L2 regularization, and evaluates both training and test accuracy over epochs.
- `train26Char.ipynb`: A Jupyter Notebook for training on a 26-character subset. 

### 2. Ensemble Methods
Located in the `Ensemble/` directory.
Multiple ensemble techniques are implemented using Scikit-Learn (SGDClassifier, etc.):
- `bagging.py`: Bagging ensemble. 
- `gradBoost.py`: Gradient Boosting ensemble (XGBoost).
- `randomForest.py`: Random Forest classifier. 
- `voting.py`: A Voting Classifier ensemble using different linear models (Logistic Regression, SVM, Perceptron).

### 3. K-Nearest Neighbors (KNN)
Located in the `KNN/` directory.
- `eval.py`: Evaluates a KNN classifier for different values of `K` (from 1 to 14) using distance-based weights. Generates a plot of accuracy vs. number of neighbors. 

### 4. Logistic Regression
Located in the `LogisticRegression/` directory.
- `train.py`: Implements a multi-class Logistic Regression model using PyTorch. Uses L1 and L2 regularization during training. 

### 5. Support Vector Machines (SVM)
Located in the `SVM/` directory.
- `train.py`: Standard SVM training.
- `trainHOG.py`: Extracts **Histogram of Oriented Gradients (HOG)** features from the images and trains a linear SVM using `SGDClassifier` from Scikit-Learn. 
## Accuracy Comparison

Here is a summary comparing the accuracies of the different models implemented in this project:

| Model | Accuracy | Notes |
| :--- | :--- | :--- |
| CNN (26-Char Subset) | 94.20% | Trained on 26-character subset |
| CNN (ResNet) | 88.23% | Custom ResNet architecture |
| SVM (with HOG features) | 83.03% | Linear SVM on extracted HOG features |
| Gradient Boosting (XGBoost) | 79.51% | Ensemble method |
| K-Nearest Neighbors (KNN) | 78.90% | Distance-weighted KNN |
| Random Forest | 78.18% | Ensemble method |
| Bagging | 75.98% | Ensemble method |
| Logistic Regression | 67.72% | Multi-class Logistic Regression with PyTorch |
| Voting Classifier | 57.20% | Ensemble of Logistic Regression, SVM, Perceptron |
| SVM (Standard) | 55.94% | Linear SVM on raw pixels |

## Requirements

The core dependencies required to run the scripts in this project are listed in `requirements.txt`. Some of the main libraries include:
- `torch` and `torchvision`
- `scikit-learn`
- `scikit-image` (for HOG feature extraction)
- `matplotlib` (for plotting results)
- `numpy`
- `joblib`

You can install all required dependencies using:

```bash
pip install -r requirements.txt
```

## How to Run

Navigate to the specific model's directory and run its training script. For example, to train the CNN ResNet model:

```bash
cd CNN
python train.py
```

Most scripts will save the trained model weights/parameters (e.g., `.pth` or `.pkl`) and generate accuracy plots (e.g., `accuracy_vs_epoch.png`) in their respective directories or a `results/` subdirectory.
