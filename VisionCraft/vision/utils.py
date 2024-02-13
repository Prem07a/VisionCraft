"""
utils.py - Module for utility functions related to image processing.

This module provides utility functions for displaying images and plotting various transformations.

Author: Prem Gaikwad
Date: Feb 2024

Usage:
from VisionCraft.vision.utils import imshow

"""


import cv2
import numpy as np
from typing import Union
import matplotlib.pyplot as plt

def imShow(title: str = "", 
           image: np.ndarray = None, 
           path: str = "",
           subplot: bool = False, 
           row: int = 0, 
           col: int = 0, 
           num: int = 0) -> None:
    """
    Display an image using Matplotlib.

    Parameters:
    - title (str, optional): Title of the displayed image.
    - image (np.ndarray, optional): Image as a NumPy array.
    - path (str, optional): Path to the image file. If provided, 'image' parameter is ignored.
    - subplot (bool, optional): If True, displays the image as a subplot.
    - row (int, optional): Row position for the subplot.
    - col (int, optional): Column position for the subplot.
    - num (int, optional): Number of the subplot.

    Returns:
    - None

    Note:
    - If 'path' is provided but the image is not found, a message is printed, and None is returned.
    """

    if image is None:
        image = imRead(path)
        if image is None:
            return image
        
    try:
        if subplot:
            plt.subplot(row, col, num)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title(title)
            plt.axis('off')
        else:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title(title)
            plt.axis('off')
            plt.show()         
    except:
        if subplot:
            plt.subplot(row, col, num)
            plt.imshow(image, cmap='gray')
            plt.title(title)
            plt.axis('off')
        else:
            plt.imshow(image, cmap='gray')
            plt.title(title)
            plt.axis('off')
            plt.show()

def imRead(path : str,
            show : bool = False,
            BGR : bool = False) -> Union[np.ndarray,None]:
    """
    Reads an image from the specified path.

    Parameters:
    - path (str): The path to the image file.
    - show (bool): If True, displays the image using the `imShow` function.
                   Default is False.
    - BGR (bool): If True, reads the image in BGR format. If False, reads in
                  grayscale format. Default is False.

    Returns:
    - img (numpy.ndarray or None): The image read from the specified path.
                                   Returns None if the image is not found.

    """
    img = cv2.imread(path, int(BGR))   
    if img == None:
        print("No Image found at given Location")
        return None
    if show:
        imShow(title=path.split('/')[-1], image=img)
    return img  

def plotLogTransform(height : int = 10, 
                     width : int = 8) -> None:
    """
    Visualize logarithmic transformations and their inverses.

    Parameters:
    - height (int, optional): Height of the Matplotlib figure.
    - width (int, optional): Width of the Matplotlib figure.

    Returns:
    - None
    """
    plt.figure(figsize=(height, width))
    img_range = range(256)
    c = 255 / np.log(1 + np.max(np.array(img_range)))
    img_log = c * np.log(1 + np.array(img_range))
    img_inv = np.exp(img_range/c)

    plt.plot(img_range, img_log, label='Logarithmic \nTransformation')
    plt.plot(img_range, img_inv, label='Inverse Logarithmic \nTransformation')
    plt.legend(loc='upper left')

    plt.xlabel('Pixel Value')
    plt.ylabel('Transformed Value')
    plt.title('Logarithmic Transformation and its Inverse')
    plt.show()


def plotPowerLaw(height :int = 10, 
                 width : int = 8) -> None:
    """
    Visualize power-law transformations with different gamma values.

    Parameters:
    - height (int, optional): Height of the Matplotlib figure.
    - width (int, optional): Width of the Matplotlib figure.

    Returns:
    - None
    """
    plt.figure(figsize=(height, width))
    img_range = np.arange(256)
    gammas = [0.04, 0.1, 0.2, 0.4, 0.67, 1, 1.5, 2.5, 5, 10, 25]

    for gamma in gammas:
        img_pl = (255 * (img_range / 255) ** gamma).astype(np.uint8)
        plt.plot(img_range, img_pl, label=f'Gamma = {gamma}')

    plt.title('Power Law Transformation with Different Gamma Values')
    plt.xlabel('Input Pixel Value')
    plt.ylabel('Transformed Pixel Value')

    plt.legend(loc='best')

    plt.show()