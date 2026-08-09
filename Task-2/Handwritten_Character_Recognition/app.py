from flask import Flask, render_template, request, jsonify

import numpy as np
import joblib
import cv2

from PIL import Image


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)


# =========================================================
# LOAD MODEL
# =========================================================

MODEL_PATH = "models/handwritten_mlp.pkl"

try:

    model = joblib.load(MODEL_PATH)

    print("================================")
    print("Model loaded successfully!")
    print("================================")

except Exception as e:

    print("================================")
    print("ERROR LOADING MODEL")
    print("================================")

    print(e)

    model = None


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template("index.html")


# =========================================================
# PREPROCESS IMAGE
# =========================================================

def preprocess_image(file):

    # -----------------------------------------------------
    # Read uploaded image
    # -----------------------------------------------------

    image = Image.open(file).convert("L")

    image = np.array(image)

    print("Original image shape:", image.shape)


    # -----------------------------------------------------
    # Resize large image if necessary
    # -----------------------------------------------------

    height, width = image.shape

    max_size = 1000

    if max(height, width) > max_size:

        scale = max_size / max(height, width)

        new_width = int(width * scale)
        new_height = int(height * scale)

        image = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )


    # -----------------------------------------------------
    # Reduce noise
    # -----------------------------------------------------

    blurred = cv2.GaussianBlur(
        image,
        (5, 5),
        0
    )


    # -----------------------------------------------------
    # Threshold
    #
    # Black background
    # White handwritten digit
    # -----------------------------------------------------

    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )


    # -----------------------------------------------------
    # Remove small noise
    # -----------------------------------------------------

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )


    # -----------------------------------------------------
    # Find contours
    # -----------------------------------------------------

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    if not contours:

        raise ValueError(
            "Could not detect the handwritten digit."
        )


    # -----------------------------------------------------
    # Find the largest useful contour
    # -----------------------------------------------------

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )


    digit_contour = None

    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)

        area = cv2.contourArea(contour)

        # Ignore extremely small objects
        if area < 20:
            continue

        # Ignore very long notebook lines
        if w > image.shape[1] * 0.8 and h < 30:
            continue

        if h > 15 and w > 5:

            digit_contour = contour

            break


    if digit_contour is None:

        raise ValueError(
            "Could not detect the digit. "
            "Please upload a clearer image."
        )


    # -----------------------------------------------------
    # Get bounding box
    # -----------------------------------------------------

    x, y, w, h = cv2.boundingRect(
        digit_contour
    )


    print(
        "Detected digit bounding box:",
        x, y, w, h
    )


    # -----------------------------------------------------
    # Add padding
    # -----------------------------------------------------

    padding = int(
        max(w, h) * 0.25
    )


    x1 = max(
        0,
        x - padding
    )

    y1 = max(
        0,
        y - padding
    )

    x2 = min(
        image.shape[1],
        x + w + padding
    )

    y2 = min(
        image.shape[0],
        y + h + padding
    )


    # -----------------------------------------------------
    # Crop digit
    # -----------------------------------------------------

    digit = binary[
        y1:y2,
        x1:x2
    ]


    # -----------------------------------------------------
    # Make square canvas
    # -----------------------------------------------------

    height, width = digit.shape

    size = max(
        height,
        width
    )

    canvas = np.zeros(
        (size, size),
        dtype=np.uint8
    )


    # Center cropped digit

    x_offset = (
        size - width
    ) // 2

    y_offset = (
        size - height
    ) // 2


    canvas[
        y_offset:y_offset + height,
        x_offset:x_offset + width
    ] = digit


    # -----------------------------------------------------
    # Resize to MNIST size
    # -----------------------------------------------------

    canvas = cv2.resize(
        canvas,
        (28, 28),
        interpolation=cv2.INTER_AREA
    )


    # -----------------------------------------------------
    # Normalize
    # -----------------------------------------------------

    canvas = canvas.astype(
        "float32"
    ) / 255.0


    # -----------------------------------------------------
    # Flatten for MLP
    # -----------------------------------------------------

    final_image = canvas.reshape(
        1,
        784
    )


    return final_image, canvas


# =========================================================
# PREDICTION
# =========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # -------------------------------------------------
        # Check model
        # -------------------------------------------------

        if model is None:

            return jsonify({

                "success": False,

                "error":
                    "Model is not loaded."

            })


        # -------------------------------------------------
        # Check file
        # -------------------------------------------------

        if "image" not in request.files:

            return jsonify({

                "success": False,

                "error":
                    "Please upload an image."

            })


        file = request.files["image"]


        # -------------------------------------------------
        # Preprocess
        # -------------------------------------------------

        image_array, processed_image = (
            preprocess_image(file)
        )


        # -------------------------------------------------
        # Predict
        # -------------------------------------------------

        prediction = model.predict(
            image_array
        )[0]


        predicted_digit = int(
            prediction
        )


        # -------------------------------------------------
        # Probability
        # -------------------------------------------------

        probabilities = model.predict_proba(
            image_array
        )[0]


        confidence = (
            float(
                np.max(probabilities)
            ) * 100
        )


        probability_list = [

            round(
                float(p) * 100,
                2
            )

            for p in probabilities

        ]


        print(
            "Predicted digit:",
            predicted_digit
        )

        print(
            "Confidence:",
            confidence
        )


        # -------------------------------------------------
        # Return result
        # -------------------------------------------------

        return jsonify({

            "success": True,

            "prediction":
                predicted_digit,

            "confidence":
                round(
                    confidence,
                    2
                ),

            "probabilities":
                probability_list

        })


    except Exception as e:

        print(
            "Prediction error:",
            e
        )


        return jsonify({

            "success": False,

            "error":
                str(e)

        })


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )