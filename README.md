Project title : Payment Fraud detector Created by Taofeeq Isiaka

An end-to-end machine learning solution designed to detect and prevent digital finance fraud. This repository covers data preprocessing model training, and evaluation metrics for identifying fraudulent behavior in large-scale financial datasets

Overview
The goal of this project is to detect fraudulent credit card transactions using a supervised machine learning model. The pipeline:

Cleans and validates the raw data.
Engineers features to reduce skewness and improve model performance.
Addresses severe class imbalance.
Trains and evaluates a Random Forest classifier.
Saves the trained model and scaler for later use in an application or API case .
Data & Preprocessing
Input data: creditcard.csv (raw transaction data with fraud labels from kaggle).
Data integrity checks:
Verified absence of null values.
Detected and removed duplicate records.
Cleaned data saved to: data/processed/creditcard_cleaned.csv.
Key preprocessing steps:

Applied log transformation to the Amount feature to reduce skewness (log_Amount = log1p(Amount)).
Dropped Time and original Amount columns after transformation.
Saved feature‑engineered dataset to: data/processed/creditcard_features.csv.
Handling Class Imbalance
The dataset is highly imbalanced (very few fraud cases compared to normal transactions). To prevent the model from always predicting the majority class:

Used ADASYN (Adaptive Synthetic Sampling) to oversample the minority (fraud) class on the training set.
Performed a stratified train‑test split to preserve the original class distribution in the test set.
Model
Algorithm: RandomForestClassifier (from scikit‑learn).
Features: All columns except Class.
Target: Class (0 = normal, 1 = fraud).
Training pipeline:

Split data into train and test sets with train_test_split (stratified, 80/20).
Fitted a RobustScaler on log_Amount to reduce the impact of outliers.
Applied ADASYN to the training data to balance fraud vs non‑fraud classes.
Trained a RandomForestClassifier with a fixed random state for reproducibility.
Evaluated the model using:
Classification report (precision, recall, f1‑score, support).
Confusion matrix.
Counts of:
Missed frauds (actual fraud predicted as normal).
Caught frauds (actual fraud predicted as fraud).
Results (Example Interpretation)
From the confusion matrix:

[[56644, 7], [24, 71]]
56,644 normal transactions correctly classified as normal.
7 normal transactions misclassified as fraud.
71 fraud transactions correctly identified as fraud.
24 fraud transactions misclassified as normal.
Model Persistence
Trained artifacts are saved using joblib:

Model: Models/fraud_detection_model.pkl
Scaler: Models/robust_scaler.pkl
These files can be loaded in a separate script or service to perform real‑time or batch inference on new transaction data.

Due to github file limit, I removed the data folder but i can be found here : https://drive.google.com/drive/folders/1g6-muFsHmyc0u041s07dhJcIM9wJSIML?usp=sharing

Learning Resources :

Data validation : https://builtin.com/data-science/data-validation - How to check and validate data

https://www.geeksforgeeks.org/pandas/dealing-with-rows-and-columns-in-pandas-dataframe/ - Rows and Columns With Pandas
