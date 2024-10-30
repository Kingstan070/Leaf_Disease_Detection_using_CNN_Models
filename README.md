# Leaf_Disease_Detection_using_CNN_Models
# 🍃 Mango Leaf Disease Classification Using CNN 🌿

## Overview
This project focuses on applying Convolutional Neural Network (CNN) architectures to classify images of mango leaves into one of eight categories. The models are trained to determine whether a mango leaf is healthy or affected by specific diseases, including **Anthracnose**, **Bacterial Canker**, **Cutting Weevil**, **Die Back**, **Gall Midge**, **Powdery Mildew**, and **Sooty Mould**. Multiple CNN architectures were trained and compared, optimizing the pipeline for reliable multi-class classification outcomes.

## Dataset Information 🗂️
The dataset used in this study is the **MangoLeafBD Dataset** (Ali et al., 2022), containing **4000 images** of mango leaves with a resolution of **240x320 pixels**. The images were collected across four mango orchards in Bangladesh, ensuring diverse environmental representation. The dataset consists of:
- **4000 Images** (1800 unique images and additional augmented copies)
- **8 Class Labels**: Each class contains 500 images (7 diseases + 1 healthy category)

### Data Format
- **Image Type**: JPG
- **Image Resolution**: 240x320 pixels

## Hyperparameter Tuning and Training ⚙️
To achieve optimal model performance, the following hyperparameters were adjusted:
- **Batch Size**: 32 (balancing memory usage and processing efficiency)
- **Epochs**: 5 (to achieve model convergence without excessive training time)
- **Learning Rate**: 0.001 (stabilizing gradient descent)

The models were compiled using the **Adam optimizer** with **categorical cross-entropy loss**, suitable for multi-class classification tasks. A **20% validation split** of the training data provided insights into model performance during training.

## Project Pipeline 🚀

### A. Data Loading and Preparation 📥
- **Image Resizing and Array Conversion**: Images were upscaled to **256x256 pixels** and converted to arrays for CNN model input.
- **Data Splitting**: A **5% test split** was applied, ensuring **95%** for training and **5%** for testing.

### B. Data Preprocessing 🧹
- **Normalization**: Each image’s pixel values were scaled by **255.0** to fit a **0–1 range**.
- **One-Hot Encoding**: Target labels were encoded into categorical vectors for multi-class predictions.

### C. CNN Model Configuration 🏗️
Multiple CNN architectures were tested, including:
- **VGG16**
- **VGG19**
- **ResNet50**
- **DenseNet121**
- **Custom CNN model** based on AlexNet

**Model Structures**:
- **Feature Extraction Layers**: Pre-trained CNNs were used for extracting nuanced image features.
- **Fully Connected Layers**: Additional layers integrated feature sets for accurate classification.
- **Regularization**: L2 regularization was applied to fully connected layers to mitigate overfitting, along with **Dropout layers** for improved generalization.

### D. Training 🏋️
Each CNN model underwent training on both training and validation data. Metrics like **loss** and **accuracy** were monitored across epochs, allowing for adjustments as necessary.

### E. Model Evaluation 📊
Post-training, each model’s performance was assessed on the test dataset. A classification report facilitated a comparative analysis across architectures, highlighting each model’s ability to identify specific diseases.

## Model Performance 📈
| Model        | Test Accuracy | Precision (Macro Avg) | Recall (Macro Avg) | F1-Score (Macro Avg) |
|--------------|---------------|-----------------------|---------------------|-----------------------|
| VGG16        | 99.00%        | 0.99                  | 0.98                | 0.99                  |
| VGG19        | 99.50%        | 0.99                  | 1.00                | 0.99                  |
| ResNet50     | 100.00%       | 1.00                  | 1.00                | 1.00                  |
| DenseNet121  | 100.00%       | 1.00                  | 1.00                | 1.00                  |
| AlexNet      | 75.00%        | 0.79                  | 0.75                | 0.75                  |


## Conclusion 🎉
The performance of the models varied based on architectural complexity and parameterization. While models like VGG and ResNet demonstrated strong feature extraction capabilities, most exhibited signs of overfitting. Despite this, the high testing accuracy indicates that the models are well-suited for mango leaf disease classification. Future iterations could improve overfitting through additional dropout layers and other regularization techniques. These adjustments were not applied in this evaluation, which focused on the vanilla versions of CNN models.

## Requirements 📋
- Python 3.x
- TensorFlow/Keras
- NumPy
- Matplotlib
- Scikit-learn

## Installation 🔧
To set up the project, clone the repository and install the required libraries:

## Kaggle Notebook 📓
For a detailed exploration and implementation of the project, you can check out the Kaggle notebook: [Leaf Disease Prediction Using Different CNN Models](https://www.kaggle.com/code/tallwinkingstan/leaf-disease-prediction-using-different-cnn-models).

