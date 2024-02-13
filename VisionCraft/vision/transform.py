"""
transform.py - Module for image transformation operations.

This module provides functions for image transformations, including gray level slicing and histogram equalization.

Author: Prem Gaikwad
Date: Feb 2024
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple
from VisionCraft.vision.utils import imShow, imRead
import matplotlib.pyplot as plt

def grayLevelSlicing(img : np.ndarray = None, 
                     path : str = "",
                     lower : int = 100, 
                     upper : int = 200, 
                     bg : bool = False, 
                     THRESHOLD : int = 256, 
                     show : bool = False, 
                     height : int = 10, 
                     width : int = 8,
                     ) -> Union[np.ndarray,None]:
    """
    Performs gray level slicing on the input image.

    Parameters:
    - img (np.ndarray, optional): Input image as a NumPy array. If provided, 'path' is ignored.
    - path (str, optional): Path to the image file. If provided, 'img' parameter is ignored.
    - lower (int, optional): Lower bound for gray level slicing.
    - upper (int, optional): Upper bound for gray level slicing.
    - bg (bool, optional): If True, retains the background in the sliced image.
    - THRESHOLD (int, optional): Threshold value for gray level slicing.
    - show (bool, optional): If True, displays the original and sliced images using Matplotlib.
    - height (int, optional): Height of the Matplotlib figure when 'show' is True.
    - width (int, optional): Width of the Matplotlib figure when 'show' is True.

    Returns:
    - np.ndarray: The sliced image as a NumPy array.

    Note:
    - If 'path' is provided but the image is not found, a message is printed, and None is returned.
    """
    if image is None:
        image = imRead(path)
        if image is None:
            return image
        
    plt.figure(figsize=(height, width))
    rows, cols = img.shape
    img1 = np.copy(img)
    for row in range(rows):
        for col in range(cols):
            if lower <= img[row][col] <= upper:
                img1[row][col] = THRESHOLD-1
            else:
                if bg:
                    pass
                else:
                    img1[row][col] = 0
    if show:
        imShow("Original Image", img, subplot=True, row=1, col=2, num=1)
        if bg:
            imShow("Grey Level Slicing With BG", img, subplot=True, row=1, col=2, num=2)
        else:
            imShow("Grey Level Slicing Without BG", img1, subplot=True, row=1, col=2, num=2)
        plt.show()
    return img1

def bitPlaneSlicing(img : np.ndarray = None, 
                    path : str = "", 
                    show : bool = False,
                    height : int = 10, 
                    width : int = 8) -> Union[np.ndarray,None]:
    """
    Performs bit-plane slicing on the input image.

    Parameters:
    - img (np.ndarray, optional): Input image as a NumPy array. If provided, 'path' is ignored.
    - path (str, optional): Path to the image file. If provided, 'img' parameter is ignored.
    - show (bool, optional): If True, displays the original and bit-plane sliced images using Matplotlib.
    - height (int, optional): Height of the Matplotlib figure when 'show' is True.
    - width (int, optional): Width of the Matplotlib figure when 'show' is True.

    Returns:
    - List[np.ndarray]: List containing bit-plane sliced images as NumPy arrays.

    Note:
    - If 'path' is provided but the image is not found, a message is printed, and None is returned.
    """
    if image is None:
        image = imRead(path)
        if image is None:
            return image
        
    planes = []
    for bit in range(8):
        img1 = np.copy(img)
        rows, cols = img1.shape
        for row in range(rows):
            for col in range(cols):
                binary = bin(img1[row][col])[2:]
                img1[row][col] = 255 if ("0"*(8-len(binary)) + binary)[::-1][bit] == "1" else 0
        if show:
            plt.figure(figsize=(height, width))
            imShow(f"Bit Plane {bit}", img1, subplot=True, row = 2, col = 4, num=bit+1)
        planes.append(img1)
    if show:
        plt.show()
    return planes


def contrastStretching(img : np.ndarray = None, 
                       path : str = "",
                       s1 : int = 30, 
                       s2 : int = 150, 
                       r1 : int = 80, 
                       r2 : int = 150, 
                       L : int = 256,
                       show : bool = False,
                       height : int = 10, 
                       width : int = 8) -> Union[np.ndarray,None]:
    """
    Performs contrast stretching on the input image.

    Parameters:
    - img (np.ndarray, optional): Input image as a NumPy array. If provided, 'path' is ignored.
    - path (str, optional): Path to the image file. If provided, 'img' parameter is ignored.
    - s1 (int, optional): Stretching value for the lower limit.
    - s2 (int, optional): Stretching value for the upper limit.
    - r1 (int, optional): Original lower limit.
    - r2 (int, optional): Original upper limit.
    - L (int, optional): Maximum gray level.
    - show (bool, optional): If True, displays the original and contrast-stretched images using Matplotlib.
    - height (int, optional): Height of the Matplotlib figure when 'show' is True.
    - width (int, optional): Width of the Matplotlib figure when 'show' is True.

    Returns:
    - np.ndarray: The contrast-stretched image as a NumPy array.

    Note:
    - If 'path' is provided but the image is not found, a message is printed, and None is returned.
    """
    if image is None:
        image = imRead(path)
        if image is None:
            return image
        
    img1 = np.copy(img)
    a = s1/r1
    b = (s2-s1)/(r2-r1)
    g = (L-s2-1)/(L-r2-1)

    rows, cols = img.shape
    for row in range(rows):
        for col in range(cols):
            if img[row][col] <= r1:
                img1[row][col] = a*img[row][col]
            elif r1 < img[row][col] <= r2:
                r = img[row][col]
                img1[row][col] = b*(r-r1) + s1
            else:
                r = img[row][col]
                img1[row][col] = g*(r-r2) + s2
    if show:
        plt.figure(figsize=(height, width))
        imShow("Original Image", img, subplot=True, row = 2, col = 2, num=1)
        plt.subplot(2,2,3)
        plt.title("Original Histogram")
        plt.hist(img.ravel(), 256, [0,256])
        imShow("Contrast Stretching Image", img1, subplot=True, row = 2, col = 2, num=2)
        plt.subplot(2,2,4)
        plt.title("Contrasted Histogram")
        plt.hist(img1.ravel(),256,[0,256])
        plt.show()
    return img1

def histogramEquilization(img : np = None,
                          path : str = "",
                          show : bool = False, 
                          height : int = 10, 
                          width : int = 8, 
                          eq_table : bool = False) -> Union[np.ndarray, None, Tuple[pd.DataFrame, np.ndarray]]:
    """
    Performs histogram equalization on the input image.

    Parameters:
    - img (np.ndarray, optional): Input image as a NumPy array. If provided, 'path' is ignored.
    - path (str, optional): Path to the image file. If provided, 'img' parameter is ignored.
    - show (bool, optional): If True, displays the original and equalized images using Matplotlib.
    - height (int, optional): Height of the Matplotlib figure when 'show' is True.
    - width (int, optional): Width of the Matplotlib figure when 'show' is True.
    - eq_table (bool, optional): If True, returns the equalization table along with the equalized image.

    Returns:
    - np.ndarray: The equalized image as a NumPy array.

    Note:
    - If 'path' is provided but the image is not found, a message is printed, and None is returned.
    """
    
    if img is None:
        img = imRead(path)
        if img is None:
            return img
        
    img1 = np.copy(img)
    freq = {}
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            r = img[row][col]
            if r in freq:
                freq[r] += 1
            else:
                freq[r] = 1
    for i in range(256):
        if i not in freq:
            freq[i] = 0
    data = {
        "GrayLevel":list(freq.keys()),
        "Nk":list(freq.values())
    }
    df = pd.DataFrame(data)
    df = df.sort_values(by="GrayLevel")
    df.reset_index(inplace=True, drop=True)
    df["PDF"] = df["Nk"]/(img.shape[0]*img.shape[1])
    df["CDF"] = df["PDF"].cumsum()
    df["Sk"] = df["CDF"]*255
    df["New_Histogram"] = df["Sk"].apply(lambda x:round(x))
    grouped_df = df[['New_Histogram', 'Nk']].groupby('New_Histogram').sum().reset_index() 
    freq = {}
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            r = img[row][col]
            img1[row][col] = df.loc[r,"New_Histogram"]
    
    if show:
        plt.figure(figsize=(height, width))
        imShow("Original Image", img, subplot=True, row=2, col=2, num=1)
        plt.subplot(2,2,2)
        plt.bar(freq.keys(), freq.values())
        plt.title("Original Histogram")
        plt.xlabel("Gray Level")
        plt.ylabel("Frequency")
        plt.subplot(2,2,4)
        plt.bar(grouped_df['New_Histogram'], grouped_df['Nk'])
        plt.title("Equalized Histogram")
        plt.xlabel("New Gray Level")
        plt.ylabel("Frequency")
        imShow("Histogram Equalization", img1, row=2, col=2, num=3, subplot=True)
        plt.show()
    if eq_table:
        return df, img1
    return img1