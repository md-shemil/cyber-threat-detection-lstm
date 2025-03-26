import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Test Data
df_test = pd.read_csv("UNSW_NB15_testing-set.csv", low_memory=False)
df_test.columns = df_test.columns.str.lower().str.strip()

# Define Features
features = ["dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", 
            "sttl", "dttl", "sload", "dload", "sinpkt", "dinpkt", "sjit", "djit",
            "ct_srv_src", "ct_state_ttl", "smean", "dmean"]

# Load target variable
target = "label"
X_test_real = df_test[features]
y_test_real = df_test[target]

# Normalize test data
scaler = StandardScaler()
X_test_real_scaled = scaler.fit_transform(X_test_real)

# Reshape for LSTM input
X_test_real = X_test_real_scaled.reshape(X_test_real_scaled.shape[0], 1, X_test_real_scaled.shape[1])

# Load trained model
from tensorflow.keras.models import load_model
import tensorflow_addons as tfa  

model = load_model("cyber_threat_lstm.keras", custom_objects={"SigmoidFocalCrossEntropy": tfa.losses.SigmoidFocalCrossEntropy()})

# Make predictions
y_pred_probs = model.predict(X_test_real)
threshold = 0.6
y_pred = (y_pred_probs > threshold).astype(int)

# Compute accuracy
accuracy = accuracy_score(y_test_real, y_pred)

# Compute classification report
report = classification_report(y_test_real, y_pred, output_dict=True)

# Compute confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test_real, y_pred).ravel()

# Compute False Positive Rate (FPR) and False Negative Rate (FNR)
fpr = fp / (fp + tn)  # False positives / Total actual negatives
fnr = fn / (fn + tp)  # False negatives / Total actual positives

# Print results
print(f"Model Accuracy on Test Data: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision: {report['1']['precision']:.4f} ({report['1']['precision'] * 100:.2f}%)")
print(f"Recall: {report['1']['recall']:.4f} ({report['1']['recall'] * 100:.2f}%)")
print(f"F1-Score: {report['1']['f1-score']:.4f} ({report['1']['f1-score'] * 100:.2f}%)")
print(f"False Positive Rate (FPR): {fpr:.4f} ({fpr * 100:.2f}%)")
print(f"False Negative Rate (FNR): {fnr:.4f} ({fnr * 100:.2f}%)")
print("Confusion Matrix:")
print(confusion_matrix(y_test_real, y_pred))
