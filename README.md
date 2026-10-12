# Payment Fraud detector Created by Taofeeq Isiaka

An end-to-end machine learning project  designed to detect and prevent digital finance fraud. This repo covers data preprocessing model training, and evaluation metrics for identifying fraudulent behavior in financial datasets.

<h3> Overview </h3>
The goal of this project is to detect fraudulent credit card transactions using a supervised machine learning model. 


<h3> The pipeline:</h3>

1.Cleaning and validation of   raw data.

2. feature engineering to reduce skewness and improve model performance.

3. Addresses severe class imbalance.
4. Training on Random Forest classifier Algorithm.
5. Saved the trained model and scaler for later use in an application or API case .

<h3> Data & Preprocessing </h3>




Key preprocessing steps:

Applied log transformation to the Amount feature to reduce skewness (log_Amount = log1p(Amount)).
Dropped Time and original Amount columns after transformation.
Saved feature‑engineered dataset


<b>Handling Class Imbalance</b>

The dataset is highly imbalanced (very few fraud cases compared to normal transactions). 

To prevent the model from always predicting the majority class:

Used ADASYN (Adaptive Synthetic Sampling) to oversample the minority (fraud) class on the training set.
Performed a stratified train‑test split to preserve the original class distribution in the test set.
Model
Algorithm: RandomForestClassifier (from scikit‑learn).
Features: All columns except Class.
Target: Class (0 = normal, 1 = fraud).


<h3> Training pipeline: </h3>

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




<h3>Results </h3>
[[56644, 7], [24, 71]]
56,644 normal transactions correctly classified as normal.
7 normal transactions misclassified as fraud.
71 fraud transactions correctly identified as fraud.
24 fraud transactions misclassified as normal.
Model Persistence
Trained artifacts are saved using joblib:



