# Cancer Prediction using Machine Learning

## Author
**KS Harshavardhan**

## Problem Statement
Early detection of cancer plays a crucial role in improving survival rates.
This project uses supervised machine learning models to predict whether a
tumor is **benign or malignant** based on medical diagnostic features.

## Dataset
- Source: Breast Cancer Wisconsin Dataset
- Features: Mean radius, texture, perimeter, area, smoothness, etc.
- Target:
  - 0 → Benign
  - 1 → Malignant

## Models Implemented
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest Classifier
- K-Nearest Neighbors (KNN)

## Improvements Added
- Feature scaling using StandardScaler
- Train-test split with stratification
- Model comparison using accuracy, precision, recall, and F1-score
- Confusion matrix visualization

## Results
Random Forest achieved the highest accuracy with improved recall for malignant cases.

## Technologies Used
- Python
- NumPy, Pandas
- Scikit-learn
- Matplotlib, Seaborn

## Future Enhancements
- Hyperparameter tuning using GridSearchCV
- XGBoost / LightGBM integration
- Model deployment using Flask

