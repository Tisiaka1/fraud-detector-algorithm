import pandas as pd  
import numpy as np 
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import ADASYN
import joblib 

# load the cleaned data 

File_path = 'data/processed/creditcard_cleaned.csv'
df = pd.read_csv(File_path)
df["log_Amount"] = np.log1p(df["Amount"])  # Apply log transformation to 'Amount' column - we do this to reduce the skewness of the data
df.drop(['Time', 'Amount'], axis=1, inplace=True)  # Drop original 'Time'  since the lapse in seconds is not very useful for modeling
df.to_csv('data/processed/creditcard_features.csv', index=False)  # Save the modified dataframe with new features
X = df.drop('Class', axis=1)  # We are Selecting All columns except class as features. class is our target varaible ( 1 or 0 )
Y = df['Class']  # Target variable
X_train , X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)  # Stratified split to maintain class distribution
#print(f"df shape: {df.shape}")
#print(f"X shape: {X_train .shape}")
#print(f"Y_train shape: {Y_train.shape}")
#print(f"X_test shape: {X_test.shape}")
#print(f"Y_test shape: {Y_test.shape}")

scaler = RobustScaler()  
scaler.fit(X_train[['log_Amount']])  
X_train[['log_Amount']] = scaler.transform(X_train[['log_Amount']])
X_test[['log_Amount']] = scaler.transform(X_test[['log_Amount']]) 

#Im applying adaysn due to class imabalance so the model doesnt just predict the majority class all the time.
adasyn = ADASYN(sampling_strategy='auto', random_state=42)
x_train_balanced, y_train_balanced = adasyn.fit_resample(X_train, Y_train)

model = RandomForestClassifier( random_state=42)

model.fit(x_train_balanced, y_train_balanced)
y_pred = model.predict(X_test)
print(classification_report(Y_test, y_pred))

print(confusion_matrix(Y_test, y_pred))

##[[56644     7] - Normal transactions correctly classified as normal (56644) and normal transactions misclassified as fraud (7)
 ##[   24    71]] - Fraud transactions misclassified as normal (24) and fraud transactions correctly classified as fraud (71)

 #

missed_frauds_mask = (Y_test == 1) & (y_pred == 0)  # Actual fraud, predicted normal
caught_frauds_mask = (Y_test == 1) & (y_pred == 1)  # Actual fraud, predicted fraud

missed_frauds = X_test[missed_frauds_mask]
caught_frauds = X_test[caught_frauds_mask]

print(f"Number of missed frauds: {missed_frauds.shape[0]}")
print(f"Number of caught frauds: {caught_frauds.shape[0]}")



try:
    joblib.dump(model,"Models/fraud_detection_model.pkl") #Save the  model that was just created. 
    joblib.dump(scaler,"Models/robust_scaler.pkl")
except Exception as e:
    print(f"Error saving models: {e}")
