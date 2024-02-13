"""
filter.py - Module for image filtering operations.

This module provides functions for image filtering operations.

Author: Prem Gaikwad
Date: Feb 2024
"""


import numpy as np
import matplotlib.pyplot as plt
from VisionCraft.vision.utils import imShow, imgRead

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
    img = imgRead(path)
    if img is None:
        return img
    if filter_size % 2 == 0:
        print("Please Try using Odd Numbers for filter_size to get good results")
    
    rows, cols = img.shape
    
    img1 = np.pad(img, pad_width=int(np.ceil(filter_size/2)), mode='constant', constant_values=255)
    filtered_img = np.zeros_like(img)
    for row in range(rows):
        for col in range(cols):
            replace = np.floor(np.sum(img1[row:row+filter_size, col:col+filter_size])/(filter_size*filter_size))
            filtered_img[row,col]=  replace
    if show:
        plt.figure(figsize=(height, width))
        imShow("Original Image",img, subplot=True, row=2,col=1, num=1)
        imShow("Box Filter",filtered_img,subplot=True, row=2,col=1, num=2)
        plt.show()  
        
    return filtered_img

def weightedAvgFilter(img:np.ndarray = None, 
                      path: str = "", 
                      show:bool = False, 
                      height:int = 10, 
                      width:int = 8) -> np.ndarray:  
    """
    Apply a 3x3 weighted average filter to the input image.

    Parameters:
    - image (np.ndarray): Input image (grayscale).
    - image_path (str): Path to the input image file (if 'image' is not provided).
    - show_result (bool): If True, display the original and filtered images.
    - figure_height (int): Height of the Matplotlib figure (if 'show_result' is True).
    - figure_width (int): Width of the Matplotlib figure (if 'show_result' is True).

    Returns:
    - np.ndarray: Filtered image.

    If 'image' is not provided and 'image_path' is specified, it loads the image from the path.
    The weighted average filter is applied to the image using a 3x3 filter kernel.
    If 'show_result' is True, the original and filtered images are displayed using Matplotlib.
    """
    img = imgRead(path)
    if img is None:
        return img
    filter = np.array([[1,2,1],[2,4,2],[1,2,1]])
    
    rows, cols = img.shape
    
    img1 = np.pad(img, pad_width=1, mode='constant', constant_values=255)
    filtered_img = np.zeros_like(img)
    for row in range(rows):
        for col in range(cols):
            replace = np.sum(img1[row:row+3, col:col+3] * filter)/16
            filtered_img[row,col]=  replace
    if show:
        plt.figure(figsize=(height, width))
        imShow("Original Image",img, subplot=True, row=2,col=1, num=1)
        imShow("Box Filter",filtered_img,subplot=True, row=2,col=1, num=2)
        plt.show()  
        
    return filtered_img
    
def medianFilter(img:np.ndarray = None, 
                 path: str = "", 
                 filter_size : int = 3,
                 show:bool = False, 
                 height:int = 10, 
                 width:int = 8) -> np.ndarray:  
    """
    Apply a median filter to the input image.

    Parameters:
    - img (np.ndarray): Input image (grayscale).
    - path (str): Path to the input image file (if 'img' is not provided).
    - filter_size (int): Size of the median filter (odd number recommended).
    - show_result (bool): If True, display the original and filtered images.
    - height (int): Height of the Matplotlib figure (if 'show_result' is True).
    - width (int): Width of the Matplotlib figure (if 'show_result' is True).

    Returns:
    - np.ndarray: Filtered image.

    If 'img' is not provided and 'path' is specified, it loads the image from the path.
    The median filter is applied to the image using a square neighborhood of size 'filter_size'.
    If 'filter_size' is even, a message is printed recommending odd numbers for better results.
    If 'show_result' is True, the original and filtered images are displayed using Matplotlib.
    """
    img = imgRead(path)
    if img is None:
        return img
    if filter_size % 2 == 0:
        print("Please Try using Odd Numbers for filter_size to get good results")
    
    rows, cols = img.shape
    
    img1 = np.pad(img, pad_width=int(np.ceil(filter_size/2)), mode='constant', constant_values=255)
    filtered_img = np.zeros_like(img)
    for row in range(rows):
        for col in range(cols):
            replace = np.median(img1[row:row+filter_size, col:col+filter_size])
            filtered_img[row,col]=  replace
    if show:
        plt.figure(figsize=(height, width))
        imShow("Original Image",img, subplot=True, row=2,col=1, num=1)
        imShow("Box Filter",filtered_img,subplot=True, row=2,col=1, num=2)
        plt.show()  
        
    return filtered_img
    