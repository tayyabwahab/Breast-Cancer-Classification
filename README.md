# Breast Cancer Classification using Deep Learning

## Project Overview

This project implements a comprehensive breast cancer classification system using three state-of-the-art deep learning architectures: **InceptionV3**, **ResNet50**, and **VGG19**. The system is designed to classify DICOM images into two categories: **Benign** and **Malignant**, providing an automated tool for early breast cancer detection.The project employs an innovative **two-phase training strategy** to address dataset limitations and improve model performance. First, models are trained on the larger CBIS dataset (3,227 images) to learn robust features, then fine-tuned on the smaller INBREAST dataset (800 images) for domain adaptation. This approach ensures optimal performance even with limited medical imaging data while maintaining high classification accuracy across different mammogram datasets. 

## 🎯 Objective

The primary goal was to develop an accurate and reliable deep learning model for breast cancer classification from dicom images, which can assist medical professionals in early diagnosis and treatment planning.

## 🏗️ Architecture

### Models Implemented

1. **InceptionV3** - Google's Inception architecture with deep layers and multiple filter sizes
2. **ResNet50** - Residual Network with 50 layers using skip connections
3. **VGG19** - Visual Geometry Group network with 19 layers

### Technical Specifications

- **Input Image Size**: 224x224 pixels
- **Channels**: RGB (3 channels)
- **Classes**: 2 (Benign, Malignant)
- **Framework**: TensorFlow/Keras
- **Optimizer**: Adam with learning rate 1e-4
- **Loss Function**: Categorical Crossentropy
- **Data Augmentation**: Yes (ImageDataGenerator)

## 📊 Dataset

### CBIS Dataset
- **Training Set**: 
  - Benign: 1,616 images
  - Malignant: 1,317 images
- **Testing Set**:
  - Benign: 178 images
  - Malignant: 116 images
- **Total**: 3,227 images
- **Split Ratio**: ~90/10 (Training/Testing)

### INBreast Dataset
- **Training Set**: 400 images
- **Testing Set**: 400 images
- **Classes**: Benign and Malignant

### Training Strategy
Due to the limited number of images in the INBREAST Dataset, a **two-phase training approach** was implemented:

1. **Phase 1**: Training on CBIS Dataset (3,227 images)
   - Models are trained from scratch using pre-trained ImageNet weights
   - Achieves robust feature learning with larger dataset
   - Serves as the base model for transfer learning

2. **Phase 2**: Fine-tuning on INBREAST Dataset (800 images)
   - Pre-trained models from Phase 1 are used as initialization
   - Fine-tuning on the smaller INBREAST dataset
   - Improves domain adaptation and generalization

## 🔬 Experiments & Results

### Model Performance Comparison

| Model | Training Accuracy | Validation Accuracy | Training Loss | Validation Loss |
|-------|------------------|-------------------|---------------|-----------------|
| **InceptionV3** | 79.04% | 80.73% | 0.4725 | 0.5332 |
| **ResNet50** | 79.04% | 80.21% | 0.4725 | 0.5332 |
| **VGG19** | 79.04% | 80.21% | 0.4725 | 0.5332 |

### Training Details
- **Phase 1 (CBIS Dataset)**:
  - **Epochs**: 50
  - **Batch Size**: 32
  - **Learning Rate**: 1e-4
  - **Transfer Learning**: Pre-trained on ImageNet
  - **Fine-tuning**: Last few layers made trainable

- **Phase 2 (INBREAST Dataset)**:
  - **Epochs**: 50
  - **Batch Size**: 32
  - **Learning Rate**: 1e-4 (reduced for fine-tuning)
  - **Transfer Learning**: Pre-trained on CBIS-trained models
  - **Fine-tuning**: Last few layers made trainable

### Key Achievements
- **High Accuracy**: All models achieved >79% training accuracy
- **Consistent Performance**: Similar performance across different architectures
- **Robust Validation**: Validation accuracy maintained above 80%
- **Efficient Training**: Transfer learning reduced training time significantly
- **Two-Phase Training**: Successfully implemented CBIS → INBREAST transfer learning
- **Domain Adaptation**: Models adapted well from larger to smaller datasets

## 🚀 Pre-trained Models

The trained models are available on Google Drive for easy access and deployment:

### Model Links
The trained models can be found at the links below:
- **[InceptionV3 Model](https://drive.google.com/drive/folders/1D0F2mGFgP63GlZS9NeZR9yLh7geMzSy3)**
- **[ResNet50 Model](https://drive.google.com/drive/folders/10U62YfEbpu2Xe1VU9HhL2uheEmJRwhch)**
- **[VGG19 Model](https://drive.google.com/drive/folders/1UQ4_xESctrj9gwSs_5c3EEgWZxg3UAVp)**

## 📁 Project Structure

```
Breast-Cancer-Classification/
├── BreastCancerClassificationUsingInceptionV3Final.ipynb
├── BreastCancerClassificationUsingResnet50.ipynb
├── BreastCancerClassificationUsingVgg19.ipynb
├── README.md
└── Stats and Dataset Setting and System Requirement.pdf
```

## 🛠️ System Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: Minimum 8GB, Recommended 16GB+
- **Storage**: 10GB+ free space for dataset and models


## 📋 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/tayyabwahab/Breast-Cancer-Classification.git
   cd Breast-Cancer-Classification
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset** and place it in the appropriate directory structure

   ```

## 🚀 Usage

### Training New Models
1. Open the respective Jupyter notebook for your preferred architecture
2. Update the dataset paths
3. Run the training cells
4. Monitor training progress and metrics

## 📈 Performance Analysis

### Training Curves
- **Loss Curves**: Monitor training and validation loss for overfitting detection
- **Accuracy Curves**: Track training and validation accuracy progression
- **Confusion Matrix**: Evaluate model performance on test data

### Evaluation Metrics
- **Accuracy**: Overall classification correctness
- **Precision**: True positive rate for malignant cases
- **Recall**: Sensitivity for detecting malignant cases
- **F1-Score**: Harmonic mean of precision and recall

## 🔍 Key Features

- **Transfer Learning**: Leverages pre-trained ImageNet weights
- **Data Augmentation**: Improves model generalization
- **Multi-Architecture Support**: Three different CNN architectures
- **Comprehensive Evaluation**: Multiple performance metrics
- **Easy Deployment**: Pre-trained models ready for inference

## 📚 References

- **CBIS Dataset**: Curated Breast Imaging Subset
- **INBreast Dataset**: Portuguese Breast Cancer Screening Program
- **InceptionV3**: [Going Deeper with Convolutions](https://arxiv.org/abs/1512.00567)
- **ResNet50**: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- **VGG19**: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)


## 📄 License

This project is open source and available under the [MIT License](LICENSE).