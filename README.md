# blood-cell-classifier
A convolutional neural network (CNN) that classifies different types of blood cells.

## 1. Introduction

Blood cells can be classified into white blood cells (WBC), red blood cells (RBC) and blood platelets. These are important for many functions of your body, such as coagulation, immune response and transport of oxygen. Certain counts of these cells are also used for diagnostic purposes to diagnose diseases like coagulopathy or lymphoma. Furthermore, microscopy is very important to the diagnostic process of, for example, lymphoma, to classify this further into the different subtypes.

This model aims to recognize certain types of blood cells based on microscopy of the cells.

## 2. Methods

### 2.1 Dataset

The dataset that was used for training the model is BloodMNIST from the MedMNIST dataset, found in the [following link](https://medmnist.com/).

Distribution of the data:
- Training: 11.959.
- Validation: 1.712.
- Testing: 3.421.

### 2.2 Model

The model is an adaptation of the ResNet18 model, further optimized to handle the input and output of BloodMNIST.
- Input: image with dimensions 3x28x28.
- Output: 8 classes.

## 3. Results

On the validation dataset, the model an accuracy of 91,5%, a precision of 91,5%, recall of 91,5%, F1-score of 0,915 and an AUROC of 0,995 reached after 10 epochs.

The testing dataset has not been implemented yet.