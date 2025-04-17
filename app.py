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
from Gabor import generate_iris_code

st.set_page_config(page_title="Iris detection", page_icon="👁️")
st.title("👁️ Iris detection and recognition")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    # Read image bytes
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

    # Decode image with OpenCV
    image_raw = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    col0, col1, col2 = st.columns([1, 2, 1])
    with col1:
        st.image(image_raw, caption="Uploaded image", use_container_width=True, channels="GRAY")

    col3, col4 = st.columns(2)
    with col3:
        image_center_pupil, (x, y), radius_pupil = get_pupil(image_raw)
        st.image(image_center_pupil, caption='Center and Radius Detection - pupil', channels="BGR", use_container_width=True)
    with col4:
        image_center_iris, (x, y), radius_iris = get_iris(image_raw, x, y, radius_pupil)
        st.image(image_center_iris, caption='Center and Radius Detection - iris', channels="BGR", use_container_width=True)

    #norm = normalize_iris(image_raw, x, y, radius_pupil, radius_iris)
    unwrapped = unwrap_iris_with_masks(image=image_raw, x=x, y=y, r_pupil=radius_pupil, r_iris=radius_iris, height=512, width=2048)
    st.image(unwrapped, caption='Normalized Iris', channels="GRAY", use_container_width=True)

    #code = iris_code(unwrapped, f=3)
    code = generate_iris_code(unwrapped)
    st.image(cv2.resize(code * 255, (code.shape[1] * 2, code.shape[0] * 8), interpolation=cv2.INTER_NEAREST), 
             caption='Iris Code', channels="GRAY", use_container_width=True)

    uploaded_file2 = st.file_uploader("Choose a second image", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded_file2 is not None:
        # Read image bytes
        file_bytes2 = np.asarray(bytearray(uploaded_file2.read()), dtype=np.uint8)

        # Decode image with OpenCV
        image_raw2 = cv2.imdecode(file_bytes2, cv2.IMREAD_GRAYSCALE)

        col0, col1, col2 = st.columns([1, 2, 1])
        with col1:
            st.image(image_raw2, caption="Uploaded image", use_container_width=True, channels="GRAY")

        col3, col4 = st.columns(2)
        with col3:
            image_center_pupil2, (x2, y2), radius_pupil2 = get_pupil(image_raw2)
            st.image(image_center_pupil2, caption='Center and Radius Detection - pupil', channels="BGR", use_container_width=True)
        with col4:
            image_center_iris2, (x2, y2), radius_iris2 = get_iris(image_raw2, x2, y2, radius_pupil2)
            st.image(image_center_iris2, caption='Center and Radius Detection - iris', channels="BGR", use_container_width=True)
    
        unwrapped2 = unwrap_iris_with_masks(image=image_raw2, x=x2, y=y2, r_pupil=radius_pupil2, r_iris=radius_iris2, height=512, width=2048)
        st.image(unwrapped2, caption='Normalized Iris', channels="GRAY", use_container_width=True)

        code2 = generate_iris_code(unwrapped2)
        st.image(cv2.resize(code2 * 255, (code2.shape[1] * 2, code2.shape[0] * 8), interpolation=cv2.INTER_NEAREST), 
             caption='Iris Code', channels="GRAY", use_container_width=True)


        # norm2 = normalize_iris(image_raw2, x2, y2, radius_pupil2, radius_iris2)
        # plt.imshow(norm2, cmap='gray')

        # st.image(norm2, caption='Normalized Iris', channels="GRAY", use_container_width=True)
        # code2 = iris_code(norm2, f=3)
        # st.image(code2 * 255, caption='Iris Code', channels="GRAY", use_container_width=True)
    

        
        # Calculate Hamming distance
        distance = hamming_distance(code, code2)
        st.markdown(f"## Hamming distance: __{distance:.4f}__")
        # Display result
        if distance < 0.229:
            st.success("The irises match!")
        elif distance < 0.24:
            st.warning("result inconclusive.")
        else:
            st.error("The irises do not match.")