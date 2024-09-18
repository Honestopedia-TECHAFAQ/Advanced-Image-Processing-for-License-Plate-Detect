import tempfile
import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Image enhancement using OpenCV
def enhance_image(img):
    # Convert PIL image to OpenCV format
    img = np.array(img)
    
    # Resize the image to enhance resolution
    scale_percent = 200  # Scale by 200%
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # Resize image
    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
    
    # Apply noise reduction
    denoised_img = cv2.fastNlMeansDenoisingColored(resized_img, None, 10, 10, 7, 21)
    
    return denoised_img

# Streamlit App
def main():
    st.title("Image and Video Enhancement App")
    st.write("Upload an image or video to enhance its resolution and clarity.")

    # File upload
    uploaded_file = st.file_uploader("Choose an image or video file", type=["jpg", "jpeg", "png", "mp4", "avi"])

    if uploaded_file is not None:
        if uploaded_file.type.startswith('image'):
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_column_width=True)

            enhanced_img = enhance_image(image)

            st.image(enhanced_img, caption="Enhanced Image", use_column_width=True)

        elif uploaded_file.type.startswith('video'):
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_file.read())

            cap = cv2.VideoCapture(tfile.name)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Enhance video frame
                enhanced_frame = enhance_image(frame)

                st.image(enhanced_frame, caption="Enhanced Video Frame", use_column_width=True)

            cap.release()

if __name__ == "__main__":
    main()
