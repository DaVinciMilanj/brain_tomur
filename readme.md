# Brain Tumor Detection Using CNN

This project implements a **Convolutional Neural Network (CNN)** to detect brain tumors from MRI images. The model classifies MRI scans into **Tumor** and **No Tumor** categories. It also provides evaluation metrics, visualizations, and training statistics.

---

## 🔹 Project Overview

- **Dataset:** MRI images categorized as `Positive` (with tumor) and `Negative` (without tumor).  
- **Image Preprocessing:** Images are resized to 128x128 pixels and normalized (0-1 scale).  
- **Data Augmentation:** Rotation, zoom, and horizontal flip are applied to increase dataset variability.  
- **Model Architecture:**  
  - 2 Convolutional + MaxPooling layers  
  - Flatten layer  
  - Dropout for regularization  
  - Fully Connected Dense layer  
  - Output Dense layer with sigmoid activation (binary classification)  
- **Training:**  
  - Optimizer: Adam  
  - Loss: Binary Crossentropy  
  - Metrics: Accuracy, AUC, Precision, Recall  
  - Early stopping and model checkpointing for best validation accuracy  
  - Class weighting to handle imbalanced data  

- **Evaluation:**  
  - Accuracy, F1 Score, AUC  
  - Confusion matrix visualization  
  - ROC curve  
  - Misclassified samples analysis  

---

## 🔹 Installation

Make sure you have **Python 3.9+** installed. Clone this repository and install the required packages:

```bash
pip install numpy opencv-python matplotlib tensorflow scikit-learn imutils


## 🔹 Usage

### Mount Google Drive (if using Colab)
```python
from google.colab import drive
drive.mount('/content/drive')
### Set dataset paths
```python
positive_path = '/content/drive/MyDrive/Brain_Tumor_Dataset/Positive'
negative_path = '/content/drive/MyDrive/Brain_Tumor_Dataset/Negative'


### Load images and preprocess
Images are resized to 128x128 and normalized. Labels are assigned as follows:  
- `1` → Positive (tumor)  
- `0` → Negative (no tumor)

### Train the CNN model
```python
brain_tumour_cnn.fit(
    aug.flow(train_mri, train_labels, batch_size=32),
    epochs=50,
    validation_data=(test_mri, test_labels),
    callbacks=[save_best, stop_training],
    class_weight=class_weight_dict
)

### Load the best saved model
```python
from tensorflow.keras.models import load_model

tumour_model = load_model('/content/drive/MyDrive/Brain_Tumor_Dataset/model/brain_cnn.keras')


### Make predictions
```python
y_pred_prob = tumour_model.predict(test_mri)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()


### Evaluate the model

- Classification report  
- Confusion matrix  
- ROC curve  
- F1 Score, AUC, Accuracy  
- Misclassified samples count



## 🔹 Project Results

After training, the model outputs:

- Training and validation accuracy/loss/precision/recall/AUC plots  
- Confusion matrix and ROC curve  
- Detailed classification metrics  
- Number of misclassified images and their indices




## 🔹 File Structure

```plaintext
Brain_Tumor_Dataset/
│
├── Positive/          # MRI images with tumor
├── Negative/          # MRI images without tumor
├── model/             # Saved CNN model (brain_cnn.keras)
├── brain-tumor.ipynb  # Training and evaluation notebook
└── README.md          # Project documentation


## 🔹 References

- TensorFlow Keras documentation  
- OpenCV image processing tutorials  
- Sklearn metrics and evaluation functions
