import pickle
import numpy as np

# Load the model
with open("deepfake_model.pkl", "rb") as f:
    model = pickle.load(f)

# Create a dummy feature array (random values similar to extracted features)
test_features = np.array([[120, 70, 80, 255, 0, 75, 62, 89, 127, 115]]) / 255.0

# Print features
print("🔍 Test Features:", test_features)

# Get the prediction
prediction = model.predict(test_features)
confidence = model.predict_proba(test_features)

print("🔮 Model Prediction:", prediction)
print("📊 Confidence Scores:", confidence)
