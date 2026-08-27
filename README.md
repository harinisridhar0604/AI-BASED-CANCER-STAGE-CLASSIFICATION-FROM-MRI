# AI-BASED-CANCER-STAGE-CLASSIFICATION-FROM-MRI
AI-based cancer stage classification system using MRI images with ResNet50, Transfer Learning, Grad-CAM explainability, and Streamlit for automated Stage I–IV prediction.

##  Project Overview

AI Based Cancer Stage Classification from MRI is a deep learning project developed to classify cancer stages from MRI images.

The system uses a fine-tuned ResNet50 Convolutional Neural Network with transfer learning to classify MRI images into four stages:

- Stage I
- Stage II
- Stage III
- Stage IV

Grad-CAM is integrated to provide visual explanations by highlighting the regions of the MRI image that influenced the model's prediction.

A Streamlit web interface is used to allow users to upload MRI images and obtain the predicted cancer stage along with a confidence score and Grad-CAM visualization.

##  Objectives

- To develop an AI-based system for automatic cancer stage classification.
- To classify MRI images into Stage I–IV.
- To implement ResNet50 using transfer learning.
- To perform MRI image preprocessing.
- To use Grad-CAM for visual explainability.
- To develop a simple Streamlit-based user interface.
- To evaluate the model using classification metrics.

##  Methodology

The system follows the following pipeline:

MRI Image
↓
Preprocessing
↓
ResNet50 CNN
↓
Stage Prediction
↓
Grad-CAM
↓
Streamlit Output

### 1. MRI Input

The system accepts MRI images as input.

Supported formats:

- PNG
- JPG/JPEG
- DICOM

### 2. Preprocessing

The input MRI image is processed before classification.

The preprocessing steps include:

- Image validation
- Image resizing to 224 × 224 pixels
- Intensity normalization
- Noise filtering
- Data augmentation during training

### 3. ResNet50 Classification

ResNet50 is used as the main deep learning architecture.

Transfer learning from ImageNet is used to improve feature extraction and classification performance.

The final layer produces four classes:

- Stage I
- Stage II
- Stage III
- Stage IV

### 4. Grad-CAM

Grad-CAM is used to visualize the important regions of the MRI image that contributed to the model prediction.

This improves the explainability of the AI model.

### 5. Streamlit Interface

A Streamlit web application allows the user to:

1. Upload an MRI image.
2. Process the image.
3. Predict the cancer stage.
4. Display the confidence score.
5. View the Grad-CAM heatmap.

##  Technologies Used

- Python 3.9
- TensorFlow
- Keras
- ResNet50
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Streamlit
- Grad-CAM
- Google Colab

##  Project Structure

```text
AI-Cancer-Stage-Classification/
│
├── README.md
├── Cancer_VSB_Final.pptx
├── app.py
├── requirements.txt
│
├── model/
│   └── resnet50_model.h5
│
├── dataset/
│
├── src/
│   ├── preprocessing.py
│   ├── prediction.py
│   └── gradcam.py
