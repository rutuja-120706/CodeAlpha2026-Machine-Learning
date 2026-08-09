import numpy as np
import joblib

from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# --------------------------------------------------
# 1. Load trained model
# --------------------------------------------------

model = joblib.load("models/handwritten_mlp.pkl")

print("Model loaded successfully!")

# --------------------------------------------------
# 2. Load MNIST dataset
# --------------------------------------------------

print("\nLoading MNIST test data...")

mnist = fetch_openml(
    "mnist_784",
    version=1,
    as_frame=False
)

X = mnist.data
y = mnist.target.astype(np.int64)

# --------------------------------------------------
# 3. Normalize images
# --------------------------------------------------

X = X.astype("float32") / 255.0

# --------------------------------------------------
# 4. Use last 10,000 images as test data
# --------------------------------------------------

X_test = X[60000:]
y_test = y[60000:]

print("Test images:", X_test.shape[0])

# --------------------------------------------------
# 5. Evaluate model
# --------------------------------------------------

predicted_labels = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predicted_labels
)

print("\n==============================")
print("MODEL TEST RESULT")
print("==============================")

print(f"Test Accuracy : {accuracy * 100:.2f}%")

# --------------------------------------------------
# 6. Classification report
# --------------------------------------------------

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predicted_labels
    )
)

# --------------------------------------------------
# 7. Confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    y_test,
    predicted_labels
)

print("\nConfusion Matrix:")
print(cm)

# --------------------------------------------------
# 8. Test individual images
# --------------------------------------------------

print("\nSample Predictions:")
print("------------------------------")

for i in range(10):

    image = X_test[i:i + 1]

    prediction = model.predict(image)

    predicted_digit = prediction[0]

    # MLPClassifier does not return neural-network
    # probabilities directly from predict()
    probabilities = model.predict_proba(image)

    confidence = np.max(probabilities) * 100

    print(
        f"Actual: {y_test[i]} | "
        f"Predicted: {predicted_digit} | "
        f"Confidence: {confidence:.2f}%"
    )

print("\nTesting completed successfully!")