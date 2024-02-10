"""
filter.py - Module for image filtering operations.

This module provides functions for image filtering operations, such as box filtering.

Author: Prem Gaikwad
Date: Feb 2024

Usage:
from VisionCraft.vision.filter import boxFilter

"""


import numpy as np
import cv2
from vision.utils import imshow
import matplotlib.pyplot as plt

def boxFilter(img:np.ndarray = None, 
              path: str = "", 
              filter_size:int = 3, 
              show:bool = False, 
              height:int = 10, 
              width:int = 8) -> np.ndarray:
    """
    Applies a box filter to the input image.

    Parameters:
    - img (np.ndarray, Required is path not given): Input image as a NumPy array. If not provided,
                                  and 'path' is specified, the image will be loaded
                                  from the given path using OpenCV.
    - path (str, Required if img not given): Path to the image file. If provided, 'img' parameter is ignored.
    - filter_size (int, optional): Size of the box filter. Should be an odd number for best results.
    - show (bool, optional): If True, displays the original and filtered images using Matplotlib.
    - height (int, optional): Height of the Matplotlib figure when 'show' is True.
    - width (int, optional): Width of the Matplotlib figure when 'show' is True.

    Returns:
    - np.ndarray: The filtered image as a NumPy array.

    Note:
    - If 'path' is provided but the image is not found, a message is printed, and None is returned.
    - If 'filter_size' is an even number, a message is printed, recommending the use of odd numbers.
    """
    if path!="":
        img = cv2.imread(path, 0)
        if img is None:
            print("\n\n404: Image not found at given path\n\n")
            return
    if filter_size % 2 == 0:
        print("Please Try using Odd Numbers for filter_size to get good results")
    
    rows, cols = img.shape
    
    img1 = np.pad(img, pad_width=int(np.ceil(filter_size/2)), mode='constant', constant_values=0)
    filtered_img = np.zeros_like(img)
    for row in range(rows):
        for col in range(cols):
            replace = np.floor(np.sum(img1[row:row+filter_size, col:col+filter_size])/(filter_size*filter_size))
            filtered_img[row,col]=  replace
    if show:
        plt.figure(figsize=(height, width))
        imshow("Original Image",img, subplot=True, row=2,col=1, num=1)
        imshow("Box Filter",filtered_img,subplot=True, row=2,col=1, num=2)
        plt.show()  
        
    return filtered_img