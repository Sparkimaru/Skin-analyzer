import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import tempfile
import os

# Title
st.set_page_config(page_title="AI Skin Analyzer", layout="centered")
st.title("🤖 Smart AI-Based Skin Condition Analyzer")
st.write("Upload or capture a face image to analyze skin conditions like acne, oiliness, and dryness.")

# Function to detect and analyze skin conditions (very basic simulation)
def analyze_skin(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    diff = cv2.absdiff(gray, blur)
    _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    acne_pixels = cv2.countNonZero(thresh)

    # Simulated analysis based on pixel counts
    if acne_pixels > 2000:
        acne = "High"
    elif acne_pixels > 1000:
        acne = "Moderate"
    else:
        acne = "Low"

    # Simulated oiliness and dryness using brightness and contrast
    brightness = np.mean(gray)
    if brightness > 170:
        dryness = "High"
        oiliness = "Low"
    elif brightness < 80:
        oiliness = "High"
        dryness = "Low"
    else:
        oiliness = "Moderate"
        dryness = "Moderate"

    return acne, oiliness, dryness

# Webcam capture
def capture_webcam_image():
    cam = cv2.VideoCapture(0)
    st.info("Press 'Space' to capture, 'Esc' to exit.")
    captured = False
    img = None

    while True:
        ret, frame = cam.read()
        if not ret:
            st.error("Failed to access camera.")
            break
        cv2.imshow("Webcam - Press Space to Capture", frame)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        elif key == 32:  # Space
            captured = True
            img = frame
            break

    cam.release()
    cv2.destroyAllWindows()
    return img if captured else None

# Image input
option = st.radio("Choose Image Source:", ["📷 Capture via Webcam", "📁 Upload from Device"])

image = None
if option == "📁 Upload from Device":
    uploaded = st.file_uploader("Upload a clear face image", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), 1)

elif option == "📷 Capture via Webcam":
    if st.button("Capture Photo"):
        img = capture_webcam_image()
        if img is not None:
            image = img
            st.success("Image captured successfully!")

# Process the image
if image is not None:
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Input Image", use_column_width=True)

    with st.spinner("Analyzing skin..."):
        acne, oiliness, dryness = analyze_skin(image)

    st.subheader("🧪 Analysis Results")
    st.write(f"**Acne Level:** {acne}")
    st.write(f"**Oiliness:** {oiliness}")
    st.write(f"**Dryness:** {dryness}")

    st.subheader("💡 Skincare Tips")

    if acne == "High":
        st.info("Use salicylic acid or benzoyl peroxide cleansers. Avoid picking acne.")
    elif acne == "Moderate":
        st.info("Cleanse twice daily and use mild exfoliants.")
    else:
        st.info("Maintain with gentle skincare routine.")

    if oiliness == "High":
        st.warning("Use oil-free moisturizers and clay masks to reduce sebum.")
    elif dryness == "High":
        st.warning("Use hyaluronic acid and heavy moisturizers. Avoid hot water washes.")

    st.markdown("---")
    st.caption("🔍 Disclaimer: This tool gives general suggestions and is not a replacement for professional medical advice.")

