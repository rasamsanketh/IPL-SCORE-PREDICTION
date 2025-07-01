import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib
import os

# Load dataset
df = pd.read_csv("ipl_data.csv")

# Drop unwanted columns
df.drop(['date', 'runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5', 'mid', 'striker', 'non-striker'], axis=1, inplace=True)

# Encode categorical features
encoders = {}
for col in ['batting_team', 'bowling_team', 'venue']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Save the encoders
os.makedirs('model', exist_ok=True)
joblib.dump(encoders, 'model/encoders.pkl')

# Feature and target split
X = df.drop('total', axis=1)
y = df['total']

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'model/scaler.pkl')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Neural Network model with 3 hidden layers
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))  # Output layer

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# Save the model
model.save('model/ipl_model.h5')

print("✅ Training complete. Model and preprocessors saved in 'model/' folder.")
