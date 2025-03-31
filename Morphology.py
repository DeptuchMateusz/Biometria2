import numpy as np
import csv

def erode(image, struct_elem, repr_point):
    """
    Wykonuje operację erozji na obrazie binarnym zgodnie z elementem strukturalnym i punktem reprezentatywnym.
    
    :param image: Obraz binarny jako numpy array (0 - tło, 1 - obiekt).
    :param struct_elem: Element strukturalny jako numpy array.
    :param repr_point: Punkt reprezentatywny w elemencie strukturalnym (współrzędne w macierzy struct_elem).
    :return: Obraz po operacji erozji.
    """
    image = image[:, :, 0]
    img_h, img_w = image.shape
    kernel_h, kernel_w = struct_elem.shape
    rep_h, rep_w = repr_point
    
    eroded_image = np.zeros_like(image)
    
    for i in range(rep_h, img_h - (kernel_h - rep_h - 1)):
        for j in range(rep_w, img_w - (kernel_w - rep_w - 1)):

            region = image[i - rep_h:i + (kernel_h - rep_h), 
                           j - rep_w:j + (kernel_w - rep_w)]
            
            if not np.array_equal(region, struct_elem):
                eroded_image[i, j] = 255
    
    return eroded_image


def dilate(image, struct_elem, repr_point):

    #image = image[:, :, 0]
    img_h, img_w = image.shape
    kernel_h, kernel_w = struct_elem.shape
    rep_h, rep_w = repr_point

    dilated_image = image.copy()
    
    for i in range(rep_h, img_h - (kernel_h - rep_h - 1)):
        for j in range(rep_w, img_w - (kernel_w - rep_w - 1)):

            region = image[i - rep_h:i + (kernel_h - rep_h), 
                           j - rep_w:j + (kernel_w - rep_w)]
            
            if image[i,j] == 0:
                if not np.array_equal(region, struct_elem):
                    dilated_image[i - rep_h:i + (kernel_h - rep_h), 
                                  j - rep_w:j + (kernel_w - rep_w)] = 0

    return dilated_image