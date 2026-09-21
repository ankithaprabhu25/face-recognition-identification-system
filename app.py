import base64

import cv2
import numpy as np

from flask import Flask, render_template, request

from src.enrollment import enroll_person
from src.recognition import recognize_face


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


def decode_webcam_image(image_data):
    """
    Convert the Base64 webcam image from the browser
    into an OpenCV image.
    """

    if not image_data:
        return None

    try:

        # Remove the data:image/jpeg;base64, part
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        # Decode Base64
        image_bytes = base64.b64decode(image_data)

        # Convert bytes to NumPy array
        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        # Convert to OpenCV image
        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        return image

    except Exception as error:

        print("Image decoding error:", error)

        return None


# =========================================================
# ENROLL
# =========================================================

@app.route("/enroll", methods=["POST"])
def enroll():

    name = request.form.get("name", "").strip()

    image_data = request.form.get("image", "")


    # Name missing
    if not name:

        return render_template(
            "index.html",
            message="Please enter a person's name.",
            captured_image=image_data
        )


    # Image missing
    if not image_data:

        return render_template(
            "index.html",
            message="Please capture a face using the webcam first."
        )


    # Decode image
    image = decode_webcam_image(image_data)


    if image is None:

        return render_template(
            "index.html",
            message="Could not process the captured image.",
            captured_image=image_data
        )


    # Enroll face
    success, message = enroll_person(
        name,
        image
    )


    # Return page with captured image
    return render_template(
        "index.html",
        message=message,
        captured_image=image_data
    )


# =========================================================
# RECOGNIZE
# =========================================================

@app.route("/recognize", methods=["POST"])
def recognize():

    image_data = request.form.get("image", "")


    # Image missing
    if not image_data:

        return render_template(
            "index.html",
            message="Please capture a face using the webcam first."
        )


    # Decode image
    image = decode_webcam_image(image_data)


    if image is None:

        return render_template(
            "index.html",
            message="Could not process the captured image.",
            captured_image=image_data
        )


    # Recognize face
    name, score, message = recognize_face(image)


    # Return result + captured image
    return render_template(
        "index.html",
        message=message,
        name=name,
        score=score,
        captured_image=image_data
    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(debug=True)