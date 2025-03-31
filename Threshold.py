import numpy as np

def binarize(grayscale_image, threshold):
    h, w = grayscale_image.shape[0:2]
    mean_intensity = np.sum(grayscale_image) / (h * w)
    binary_image = np.where(grayscale_image > mean_intensity * threshold, 255, 0).astype(np.uint8)

    
    return binary_image

