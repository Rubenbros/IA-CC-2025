"""
Debug script to check if model is working correctly
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from modelo import cargar_modelo, cargar_datos, preparar_datos, crear_features, seleccionar_features
from modelo import JUGADA_A_NUM, NUM_A_JUGADA, PIERDE_CONTRA

print("=" * 60)
print("   MODEL DEBUGGING SCRIPT")
print("=" * 60)

# Load model
try:
    modelo = cargar_modelo()
    print("\n✅ Model loaded successfully")
except Exception as e:
    print(f"\n❌ Error loading model: {e}")
    exit(1)

# Load data
df = cargar_datos()
df = preparar_datos(df)
df = crear_features(df)
X, y = seleccionar_features(df)

print(f"\n📊 Dataset info:")
print(f"   Total samples: {len(X)}")
print(f"   Features: {X.shape[1]}")

# Test on training data
y_pred = modelo.predict(X)
accuracy = (y_pred == y).mean()

print(f"\n🎯 Model performance on TRAINING data:")
print(f"   Accuracy: {accuracy:.3f} ({accuracy * 100:.1f}%)")

if accuracy < 0.40:
    print("   ❌ CRITICAL: Model is performing WORSE than random!")
    print("   🐛 There is likely a bug in the code")
elif accuracy < 0.50:
    print("   ⚠️  WARNING: Model is barely better than random")
    print("   💡 Model is not learning patterns effectively")
elif accuracy < 0.70:
    print("   ⚠️  Low accuracy: Model is underfitting")
    print("   💡 Try: more features, different model, or more data")
else:
    print("   ✅ Good accuracy on training data")

# Check predictions distribution
unique, counts = np.unique(y_pred, return_counts=True)
print(f"\n📈 Prediction distribution:")
for move_num, count in zip(unique, counts):
    move = NUM_A_JUGADA[move_num]
    percentage = count / len(y_pred) * 100
    print(f"   {move}: {count} ({percentage:.1f}%)")

# Check actual distribution
unique_actual, counts_actual = np.unique(y, return_counts=True)
print(f"\n📈 Actual distribution (training data):")
for move_num, count in zip(unique_actual, counts_actual):
    move = NUM_A_JUGADA[move_num]
    percentage = count / len(y) * 100
    print(f"   {move}: {count} ({percentage:.1f}%)")

# Simulate a few predictions
print(f"\n🎮 Simulating 10 predictions:")
print("   (Using random samples from training data)\n")

np.random.seed(42)
sample_indices = np.random.choice(len(X), size=min(10, len(X)), replace=False)

correct = 0
for idx in sample_indices:
    features = X.iloc[idx:idx + 1]
    actual = y.iloc[idx]
    predicted = modelo.predict(features)[0]

    actual_move = NUM_A_JUGADA[actual]
    predicted_move = NUM_A_JUGADA[predicted]
    counter_move = PIERDE_CONTRA[predicted_move]

    match = "✅" if predicted == actual else "❌"

    print(f"   Actual: {actual_move} | Predicted: {predicted_move} | Counter: {counter_move} {match}")

    if predicted == actual:
        correct += 1

print(f"\n   Simulation accuracy: {correct}/10 ({correct * 10}%)")

# Check if model has learned anything
print(f"\n🧠 Model intelligence check:")

# If model always predicts the same thing = broken
if len(unique) == 1:
    print("   ❌ CRITICAL: Model only predicts ONE move!")
    print("   🐛 Model is completely broken")
elif len(unique) == 2:
    print("   ⚠️  WARNING: Model only uses 2 out of 3 moves")
    print("   💡 Model is heavily biased")
else:
    print("   ✅ Model uses all 3 moves")

# Check confidence
if hasattr(modelo, 'predict_proba'):
    probas = modelo.predict_proba(X)
    avg_max_proba = probas.max(axis=1).mean()
    print(f"\n📊 Average prediction confidence: {avg_max_proba:.3f}")

    if avg_max_proba < 0.40:
        print("   ⚠️  Very low confidence - model is uncertain")
    elif avg_max_proba > 0.80:
        print("   ⚠️  Very high confidence - model might be overfitting")
    else:
        print("   ✅ Reasonable confidence level")

print("\n" + "=" * 60)
print("   DEBUG COMPLETE")
print("=" * 60)