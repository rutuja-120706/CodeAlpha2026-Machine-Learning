import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import joblib

# --------------------------------------------------
# 1. Create models folder
# --------------------------------------------------

os.makedirs("models", exist_ok=True)

# --------------------------------------------------
# 2. Load MNIST dataset
# --------------------------------------------------

print("Loading MNIST dataset...")
print("This may take some time the first time...")

mnist = fetch_openml(
    "mnist_784",
    version=1,
    as_frame=False
)

X = mnist.data
y = mnist.target.astype(np.int64)

print("Dataset loaded successfully!")
print("Total images:", X.shape)
print("Total labels:", y.shape)

# --------------------------------------------------
# 3. Normalize images
# --------------------------------------------------

X = X.astype("float32") / 255.0

# --------------------------------------------------
# 4. Split dataset
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.1,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# --------------------------------------------------
# 5. Build Neural Network model
# --------------------------------------------------

print("\nCreating neural network model...")

model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    batch_size=64,
    learning_rate_init=0.001,
    max_iter=20,
    random_state=42,
    verbose=True
)

# --------------------------------------------------
# 6. Train model
# --------------------------------------------------

print("\nTraining model...\n")

model.fit(X_train, y_train)

print("\nTraining completed!")

# --------------------------------------------------
# 7. Make predictions
# --------------------------------------------------

print("\nTesting model...")

y_pred = model.predict(X_test)

# --------------------------------------------------
# 8. Evaluate model
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n-----------------------------")
print("Model Evaluation")
print("-----------------------------")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# --------------------------------------------------
# 9. Save model
# --------------------------------------------------

model_path = "models/handwritten_mlp.pkl"

joblib.dump(model, model_path)

print("\nModel saved successfully!")
print(f"Location: {model_path}")

# --------------------------------------------------
# 10. Display sample predictions
# --------------------------------------------------

print("\nSample Predictions:")
print("-----------------------------")

for i in range(10):

    prediction = model.predict(X_test[i].reshape(1, -1))[0]

    print(
        f"Image {i + 1}: "
        f"Actual = {y_test[i]}, "
        f"Predicted = {prediction}"
    )

# --------------------------------------------------
# 11. Display sample images
# --------------------------------------------------

plt.figure(figsize=(10, 5))

for i in range(10):

    plt.subplot(2, 5, i + 1)

    image = X_test[i].reshape(28, 28)

    plt.imshow(image, cmap="gray")

    prediction = model.predict(
        X_test[i].reshape(1, -1)
    )[0]

    plt.title(
        f"Actual: {y_test[i]}\nPredicted: {prediction}"
    )

    plt.axis("off")

plt.tight_layout()

plt.savefig("sample_predictions.png")

plt.show()

# --------------------------------------------------
# 12. Training information
# --------------------------------------------------

if hasattr(model, "loss_curve_"):

    plt.figure(figsize=(8, 5))

    plt.plot(
        model.loss_curve_,
        label="Training Loss"
    )

    plt.title("MLP Training Loss")

    plt.xlabel("Iteration")

    plt.ylabel("Loss")

    plt.legend()

    plt.savefig("training_loss.png")

    plt.show()

print("\n================================")
print("Training completed successfully!")
print("================================")