import cv2
import numpy as np

def keep_largest_contour(image):
    inverted = cv2.bitwise_not(image)
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        biggest_contour = max(contours, key=cv2.contourArea)
        result = np.ones_like(image) * 255
        cv2.drawContours(result, [biggest_contour], -1, (0), thickness=cv2.FILLED)
        
        return result
    
    return image