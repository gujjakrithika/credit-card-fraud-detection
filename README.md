# Credit Card Fraud Detection using Machine Learning

This project focuses on building a machine learning model to detect fraudulent credit card transactions. Credit card fraud is a significant issue in the financial industry, and early detection is crucial to prevent financial losses and protect customers.

We use **Logistic Regression**, a simple and interpretable classification algorithm, to predict whether a transaction is fraudulent or not based on a set of anonymized features.

---

## 📌 Project Overview

- **Goal**: Accurately classify credit card transactions as fraudulent (`Class = 1`) or legitimate (`Class = 0`).
- **Model Used**: Logistic Regression
- **Techniques**:
  - Data preprocessing
  - Feature scaling
  - Train-test split
  - Model evaluation using accuracy score and confusion matrix

---

## 🧠 Why This Project?

Fraud detection is challenging due to **imbalanced data**—fraudulent transactions are extremely rare compared to legitimate ones. This makes it a great real-world problem to apply machine learning and learn about:

- Handling imbalanced datasets
- Evaluating models beyond accuracy
- Real-world use of classification algorithms

---

## 📂 Dataset Information

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Records**: 284,807 transactions
- **Fraud Cases**: 492 (only ~0.17%)
- **Features**: 30 features, anonymized for confidentiality (V1 to V28), plus `Time` and `Amount`
- **Target**: `Class` (0 = Not Fraud, 1 = Fraud)

---

## ⚙️ Tools and Libraries

- Python
- NumPy
- Pandas
- Scikit-learn (LogisticRegression, train_test_split, StandardScaler, accuracy_score, confusion_matrix)
- Matplotlib

---

## 🚀 Workflow

1. **Load Data**  
   Load the dataset using Pandas.

2. **Explore & Preprocess Data**  
   - Check class distribution.
   - Split data into features (`X`) and label (`y`).
   - Scale features using `StandardScaler`.

3. **Train-Test Split**  
   Split the data into training and testing sets using `train_test_split`.

4. **Model Training**  
   Train a `LogisticRegression` model on the scaled training data.

5. **Evaluation**  
   - Predict results on test data.
   - Evaluate using accuracy score and confusion matrix.
   - Check training accuracy for overfitting.

---

## 📊 Confusion Matrix

The confusion matrix helps evaluate how well the model identifies frauds:

- **True Positives (TP)**: Frauds correctly identified
- **True Negatives (TN)**: Legitimate transactions correctly identified
- **False Positives (FP)**: Legitimate transactions incorrectly marked as fraud
- **False Negatives (FN)**: Frauds missed by the model

This is especially important due to the **class imbalance**.

---

## 🧩 Future Improvements

- Use advanced models like Random Forest, XGBoost, or Neural Networks.
- Handle class imbalance using:
  - Undersampling
  - Oversampling (SMOTE)
  - Class weighting
- Add precision, recall, F1-score, and ROC curve for better evaluation.

---

## 📸 Sample Output

- Training Accuracy: ~99%
- Model trained on scaled features
- Confusion matrix plotted using `matplotlib`


