import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ReduceLROnPlateau
import tensorflow_addons as tfa  

# Load dataset
df = pd.read_csv("UNSW_NB15_training-set.csv", low_memory=False)
df.columns = df.columns.str.lower().str.strip()
target = "label"

features = ["dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", 
            "sttl", "dttl", "sload", "dload", "sinpkt", "dinpkt", "sjit", "djit",
            "ct_srv_src", "ct_state_ttl", "smean", "dmean"]

X = df[features]
y = df[target]

# Convert categorical target to numeric
if y.dtype == "object":
    y = pd.factorize(y)[0]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Reshape for LSTM (Add time step dimension)
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Define LSTM Model
model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True), input_shape=(1, X_train.shape[2])),
    Dropout(0.3),
    Bidirectional(LSTM(64)),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

# Compile model
model.compile(optimizer="adam", loss=tfa.losses.SigmoidFocalCrossEntropy(), metrics=["accuracy"])

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))

# Save model
model.save("cyber_threat_lstm.keras")

# Load Test Data
df_test = pd.read_csv("UNSW_NB15_testing-set.csv", low_memory=False)
df_test.columns = df_test.columns.str.lower().str.strip()
X_test_real = df_test[features]
X_test_real_scaled = scaler.transform(X_test_real)
y_test_real = df_test[target]

# Reshape for LSTM
X_test_real = X_test_real_scaled.reshape(X_test_real_scaled.shape[0], 1, X_test_real_scaled.shape[1])

# Predict
y_pred_probs = model.predict(X_test_real)
threshold = 0.6
y_pred = (y_pred_probs > threshold).astype(int)

# Evaluate Model
accuracy = accuracy_score(y_test_real, y_pred)
print(f"Model Accuracy on Test Data: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test_real, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test_real, y_pred))