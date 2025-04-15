from Threshold import binarize
from Morphology import erode, dilate
from Contour import keep_largest_contour
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import streamlit as st
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter1d
import streamlit as st
import numpy as np
import cv2
from funcs import *




st.title("Iris detection and recognition")


uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    # Read image bytes
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

    # Decode image with OpenCV
    image_raw = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    st.image(image_raw, caption="Uploaded image", use_container_width=True, channels="GRAY")


    image_center, (x, y), radius_pupil = get_pupil(image_raw)

    # wyświetlenie zdjęć i wykresów
    st.image(image_center, caption='Center and Radius Detection', channels="BGR")




    image_center, (x, y), radius_iris = get_iris(image_raw, x, y, radius_pupil)


    st.image(image_center, caption='Center and Radius Detection', channels="BGR")

   


    norm = normalize_iris(image_raw, x, y, radius_pupil, radius_iris)
    plt.imshow(norm, cmap='gray')


    normalized = norm 
    st.image(normalized, caption='Normalized Iris', channels="GRAY", use_container_width=True)


    code = iris_code(normalized, f=3)
    st.image(code * 255, caption='Iris Code', channels="GRAY", use_container_width=True)


    uploaded_file2 = st.file_uploader("Choose a second image", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded_file2 is not None:
        # Read image bytes
        file_bytes2 = np.asarray(bytearray(uploaded_file2.read()), dtype=np.uint8)

        # Decode image with OpenCV
        image_raw2 = cv2.imdecode(file_bytes2, cv2.IMREAD_GRAYSCALE)
        st.image(image_raw2, caption="Uploaded image", use_container_width=True, channels="GRAY")
    
        image_center2, (x2, y2), radius_pupil2 = get_pupil(image_raw2)
        image_center2, (x2, y2), radius_iris2 = get_iris(image_raw2, x2, y2, radius_pupil2)
        norm2 = normalize_iris(image_raw2, x2, y2, radius_pupil2, radius_iris2)
        plt.imshow(norm2, cmap='gray')
        normalized2 = norm2
        st.image(normalized2, caption='Normalized Iris', channels="GRAY", use_container_width=True)
        code2 = iris_code(normalized2, f=3)
        st.image(code2 * 255, caption='Iris Code', channels="GRAY", use_container_width=True)
    

        
        # Calculate Hamming distance
        distance = hamming_distance(code, code2)
        st.markdown(f"## Hamming distance: __{distance:.4f}__")
        # Display result
        if distance < 0.36:
            st.success("The irises match!")
        elif distance < 0.45:
            st.warning("result inconclusive.")
        else:
            st.error("The irises do not match.")