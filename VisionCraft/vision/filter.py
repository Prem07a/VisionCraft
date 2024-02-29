"""
filter.py - Module for image filtering operations.

This module provides functions for image filtering operations.

Author: Prem Gaikwad
Date: Feb 2024
"""


import numpy as np
from typing import Union
import matplotlib.pyplot as plt
from VisionCraft.vision.utils import imShow, imRead

def boxFilter(img:np.ndarray = None, 
              path: str = "", 
              filter_size:int = 3, 
              show:bool = False, 
              height:int = 10, 
              width:int = 8,
              CONSTANT = 255) -> Union[np.ndarray,None]:
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
    - CONSTANT: Value to add in padding
    Returns:
    - np.ndarray: The filtered image as a NumPy array.

    Note:
    - If 'path' is provided but the image is not found, a message is printed, and None is returned.
    - If 'filter_size' is an even number, a message is printed, recommending the use of odd numbers.
    """
    if img is None:
        img = imRead(path)
        if img is None:
            return img
        
    if filter_size % 2 == 0:
        print("Please Try using Odd Numbers for filter_size to get good results")
    
    rows, cols = img.shape
    
    img1 = np.pad(img, pad_width=int(np.floor(filter_size/2)), mode='constant', constant_values=CONSTANT)
    filtered_img = np.zeros_like(img)
    for row in range(rows):
        for col in range(cols):
            replace = np.round(np.sum(img1[row:row+filter_size, col:col+filter_size])/(filter_size*filter_size))
            filtered_img[row,col]=  replace
    if show:
        plt.figure(figsize=(height, width))
        imShow("Original Image",img, subplot=True, row=2,col=1, num=1)
        imShow("Box Filter",filtered_img,subplot=True, row=2,col=1, num=2)
        plt.show()  
        
    return filtered_img

def weightedAvgFilter(img:np.ndarray = None, 
                      path: str = "", 
                      filter_size:int = 3, 
                      sigma:int = 1, 
                      show:bool = False, 
                      height:int = 10, 
                      width:int = 8,
                      CONSTANT=255  ) -> Union[np.ndarray,None]:  
    """
    Apply a weighted average filter to the input image.

    Parameters:
    - image (np.ndarray): Input image (grayscale).
    - image_path (str): Path to the input image file (if 'image' is not provided).
    - filter_size (int, optional): Size of the box filter. Should be an odd number for best results.
    - show_result (bool): If True, display the original and filtered images.
    - figure_height (int): Height of the Matplotlib figure (if 'show_result' is True).
    - figure_width (int): Width of the Matplotlib figure (if 'show_result' is True).
    - CONSTANT: Value to add in padding

    Returns:
    - np.ndarray: Filtered image.

    If 'image' is not provided and 'image_path' is specified, it loads the image from the path.
    The weighted average filter is applied to the image using a 3x3 filter kernel.
    If 'show_result' is True, the original and filtered images are displayed using Matplotlib.
    """
    if img is None:
        img = imRead(path)
        if img is None:
            return img
    
    filter = np.empty((filter_size,filter_size))
    sum_filter = 0
    for i in range(filter_size):
        for j in range(filter_size):
            x = i - filter_size//2
            y = j - filter_size//2
            value = 2**(-(x*x+y*y)/(sigma**2))
            filter[i][j] = value
            sum_filter += value
    filter = filter/sum_filter

    rows, cols = img.shape
    pad_width = filter_size//2
    img1 = np.pad(img, pad_width=pad_width, mode='constant', constant_values=CONSTANT)
    filtered_img = np.zeros_like(img)
    for row in range(rows):
        for col in range(cols):
            replace = np.round(np.sum(img1[row:row+filter_size, col:col+filter_size] * filter))
            filtered_img[row,col]=  replace
    if show:
        plt.figure(figsize=(height, width))
        imShow("Original Image",img, subplot=True, row=2,col=1, num=1)
        imShow("Weighted Avg Filter",filtered_img,subplot=True, row=2,col=1, num=2)
        plt.show()  
        
    return filtered_img
    
def medianFilter(img:np.ndarray = None, 
                 path: str = "", 
                 filter_size : int = 3,
                 show:bool = False, 
                 height:int = 10, 
                 width:int = 8,
                 CONSTANT=255) -> Union[np.ndarray,None]:  
    """
    Apply a median filter to the input image.

    Parameters:
    - img (np.ndarray): Input image (grayscale).
    - path (str): Path to the input image file (if 'img' is not provided).
    - filter_size (int): Size of the median filter (odd number recommended).
    - show_result (bool): If True, display the original and filtered images.
    - height (int): Height of the Matplotlib figure (if 'show_result' is True).
    - width (int): Width of the Matplotlib figure (if 'show_result' is True).
    - CONSTANT: Value to add in padding
    
    Returns:
    - np.ndarray: Filtered image.

    If 'img' is not provided and 'path' is specified, it loads the image from the path.
    The median filter is applied to the image using a square neighborhood of size 'filter_size'.
    If 'filter_size' is even, a message is printed recommending odd numbers for better results.
    If 'show_result' is True, the original and filtered images are displayed using Matplotlib.
    """
    if img is None:
        img = imRead(path)
        if img is None:
            return img
        
    if filter_size % 2 == 0:
        print("Please Try using Odd Numbers for filter_size to get good results")
    
    rows, cols = img.shape
    
    img1 = np.pad(img, pad_width=int(np.floor(filter_size/2)), mode='constant', constant_values=CONSTANT)
    filtered_img = np.zeros_like(img)
    for row in range(rows):
        for col in range(cols):
            replace = np.median(img1[row:row+filter_size, col:col+filter_size])
            filtered_img[row,col]=  replace
    if show:
        plt.figure(figsize=(height, width))
        imShow("Original Image",img, subplot=True, row=2,col=1, num=1)
        imShow("Median Filter",filtered_img,subplot=True, row=2,col=1, num=2)
        plt.show()  
        
    return filtered_img

def minMaxFilter(img:np.ndarray = None, 
                 path: str = "",
                 minimum: bool = True, 
                 filter_size : int = 3,
                 show:bool = False, 
                 height:int = 10, 
                 width:int = 8,
                 CONSTANT=0) -> Union[np.ndarray,None]:  
    """
    Apply a minMax filter to the input image.

    Parameters:
    - img (np.ndarray): Input image (grayscale).
    - path (str): Path to the input image file (if 'img' is not provided).
    - filter_size (int): Size of the median filter (odd number recommended).
    - show_result (bool): If True, display the original and filtered images.
    - height (int): Height of the Matplotlib figure (if 'show_result' is True).
    - width (int): Width of the Matplotlib figure (if 'show_result' is True).
    - CONSTANT: Value to add in padding
    
    Returns:
    - np.ndarray: Filtered image.

    If 'img' is not provided and 'path' is specified, it loads the image from the path.
    The minMax filter is applied to the image using a square neighborhood of size 'filter_size'.
    If 'filter_size' is even, a message is printed recommending odd numbers for better results.
    If 'show_result' is True, the original and filtered images are displayed using Matplotlib.
    """
    if img is None:
        img = imRead(path)
        if img is None:
            return img
    
    if filter_size % 2 == 0:
        print("Please Try using Odd Numbers for filter_size to get good results")
        
    rows, cols = img.shape 
    img1 = np.pad(img, pad_width=int(np.floor(filter_size/2)), mode='constant', constant_values=CONSTANT)
    filtered_img = np.zeros_like(img)
    for row in range(rows):
        for col in range(cols):
            if minimum:
                replace = np.min(img1[row:row+filter_size, col:col+filter_size])
                filtered_img[row,col]=  replace
            else:
                replace = np.max(img1[row:row+filter_size, col:col+filter_size])
                filtered_img[row,col]=  replace
    if show:
        plt.figure(figsize=(height, width))
        imShow("Original Image",img, subplot=True, row=2,col=1, num=1)
        if minimum:
            imShow("Min Filter",filtered_img,subplot=True, row=2,col=1, num=2)
        else:
            imShow("Max Filter",filtered_img,subplot=True, row=2,col=1, num=2)
        plt.show()  
        
    return filtered_img