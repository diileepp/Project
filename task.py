import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
train_path = "C:\\Users\\AJ\\Downloads\\archive (11)\\fraudTrain.csv"
test_path = "C:\\Users\\AJ\\Downloads\\archive (11)\\fraudTest.csv"

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Display the first few rows of the training data
print(train_data.head())

# Check for missing values
print(train_data.isnull().sum())

# Drop unnecessary columns
# We will drop columns like 'first', 'last', 'street', 'city', 'state', 'zip', 'job', 'dob', 'trans_num' as they are not useful for fraud detection
train_data = train_data.drop(['first', 'last', 'street', 'city', 'state', 'zip', 'job', 'dob', 'trans_num'], axis=1)
test_data = test_data.drop(['first', 'last', 'street', 'city', 'state', 'zip', 'job', 'dob', 'trans_num'], axis=1)

# Convert 'trans_date_trans_time' to datetime and extract useful features
train_data['trans_date_trans_time'] = pd.to_datetime(train_data['trans_date_trans_time'])
test_data['trans_date_trans_time'] = pd.to_datetime(test_data['trans_date_trans_time'])

train_data['hour'] = train_data['trans_date_trans_time'].dt.hour
train_data['day'] = train_data['trans_date_trans_time'].dt.day
train_data['month'] = train_data['trans_date_trans_time'].dt.month
train_data['year'] = train_data['trans_date_trans_time'].dt.year

test_data['hour'] = test_data['trans_date_trans_time'].dt.hour
test_data['day'] = test_data['trans_date_trans_time'].dt.day
test_data['month'] = test_data['trans_date_trans_time'].dt.month
test_data['year'] = test_data['trans_date_trans_time'].dt.year

# Drop the original 'trans_date_trans_time' column
train_data = train_data.drop('trans_date_trans_time', axis=1)
test_data = test_data.drop('trans_date_trans_time', axis=1)

# Encode the 'merchant' column using Label Encoding
label_encoder = LabelEncoder()
train_data['merchant'] = label_encoder.fit_transform(train_data['merchant'])
test_data['merchant'] = label_encoder.transform(test_data['merchant'])

# Convert categorical variables to numerical using one-hot encoding
train_data = pd.get_dummies(train_data, columns=['category', 'gender'], drop_first=True)
test_data = pd.get_dummies(test_data, columns=['category', 'gender'], drop_first=True)

# Separate features and target variable
X_train = train_data.drop('is_fraud', axis=1)
y_train = train_data['is_fraud']
X_test = test_data.drop('is_fraud', axis=1)
y_test = test_data['is_fraud']

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train an XGBoost model (without use_label_encoder)
model = xgb.XGBClassifier(eval_metric='logloss')
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nROC AUC Score:")
print(roc_auc_score(y_test, y_pred_proba))

# Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()