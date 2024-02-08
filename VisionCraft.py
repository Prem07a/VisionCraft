"""
    Creator: Prem Gaikwad
    Student from PICT
"""

import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt


class vision:
    """
    A simple computer vision utility class using OpenCV, Matplotlib, and NumPy.

    Parameters:
    - path (str): The path to the image file (optional).

    Attributes:
    - img (numpy.ndarray): The image loaded using OpenCV.

    Methods:
    - imshow(title="", image=None, subplot=False, row=0, col=0, num=0):
        Display an image using Matplotlib.

        Parameters:
        - title (str): Title of the displayed image.
        - image (numpy.ndarray): The image data.
        - subplot (bool): Whether to use subplot (default: False).
        - row (int): Number of rows in the subplot grid (if subplot is True).
        - col (int): Number of columns in the subplot grid (if subplot is True).
        - num (int): The index of the subplot (if subplot is True).

    - imgNegative():
        Display the original and negative image.

    - imgLog():
        Display the original and logarithmically transformed image.
    """
    def __init__(self, path: str = None) -> None:
        """
        Initialize the vision class.

        Parameters:
        - path (str): The path to the image file (optional).
        """
        if path:
            self.img = cv2.imread(path,0)
        else:
            print("No Path for Image has been Passed")
        print("OpenCV Version: ",cv2.__version__)
        print("Matplotlib version:", matplotlib.__version__)
        print("NumPy version:", np.__version__)
    
    def imshow(self,
               title: str = "",
               image: np.ndarray = None,
               given = False,
               subplot: bool = False,
               row: int = 0, col: int = 0, num: int = 0) -> None:
        """
        Display an image using Matplotlib.

        Parameters:
        - title (str): Title of the displayed image.
        - image (numpy.ndarray): The image data.
        - subplot (bool): Whether to use subplot (default: False).
        - row (int): Number of rows in the subplot grid (if subplot is True).
        - col (int): Number of columns in the subplot grid (if subplot is True).
        - num (int): The index of the subplot (if subplot is True).
        """
        if given:
            image = self.img
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
        except cv2.error as e:
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
            
    def imgNegative(self):
        """
        Display the original and negative image.
        """
        self.imshow("Original Image",self.img, subplot=True, row=1, col=2, num=1)
        img_negative = 255 - self.img
        self.imshow("Image Negation",img_negative, subplot=True, row=1, col=2, num=2)
        plt.show()

    def imgLog(self):
        """
        Display the original and logarithmically transformed image.
        """
        c = 255 / np.log(1 + np.max(np.array(self.img)))
        img_log = c * np.log(1 + np.array(self.img))    
        self.imshow("Original Image",self.img, subplot=True, row=1, col=2, num=1)
        self.imshow("Logarithmic Transformation",img_log, subplot=True, row=1, col=2, num=2)
        plt.show()

    def plotLogTransform(self):
        plt.figure(figsize=(15,12))
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
        
    def powerLaw(self):
        plt.figure(figsize=(15,12))
        gammas = [0.04, 0.1, 0.2, 0.4, 0.67, 1 , 1.5, 2.5, 5, 10, 25]
        img_no = 1
        for gamma in gammas:
            img_pl = 255*(self.img/255)**gamma
            if img_no == 1:
                self.imshow("Original Image",self.img, subplot=True, row=3, col=4, num=1)
            else:
                self.imshow(f"Gamma {gamma}",img_pl, subplot=True, row=3, col=4, num=img_no)
            img_no += 1
        plt.show()

    def plotPowerLaw(self):
        plt.figure(figsize=(15,12))
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

    def flipImg(self):
        plt.figure(figsize=(15,12))
        self.imshow("Original Image",self.img, subplot=True, row=1, col=3, num=1)           
        img_flip_v = cv2.flip(self.img, 0)
        self.imshow("Vertical Flip",img_flip_v, subplot=True, row=1, col=3, num=2)
        img_flip_h = cv2.flip(self.img, 1)
        self.imshow("Horizontal Flip",img_flip_h, subplot=True, row=1, col=3, num=3)
        plt.show()
        
    def grayLevelSlicing(self, lower:int=100, upper:int=200, bg:bool=False, THRESHOLD = 256):
        plt.figure(figsize=(15,12))
        rows, cols = self.img.shape
        img = np.copy(self.img)
        for row in range(rows):
            for col in range(cols):
                if lower <= self.img[row][col] <= upper:
                    img[row][col] = THRESHOLD-1
                else:
                    if bg:
                        pass
                    else:
                        img[row][col] = 0
        self.imshow("Original Image", self.img, subplot=True, row=1, col=2, num=1)
        if bg:
            self.imshow("Grey Level Slicing With BG", img, subplot=True, row=1, col=2, num=2)
        else:
            self.imshow("Grey Level Slicing Without BG", img, subplot=True, row=1, col=2, num=2)
        plt.show()
    
    def bitPlaneSlicing(self):
        plt.figure(figsize=(15,12))
        for bit in range(8):
            img = np.copy(self.img)
            rows, cols = img.shape
            for row in range(rows):
                for col in range(cols):
                    binary = bin(img[row][col])[2:]
                    img[row][col] = 255 if ("0"*(8-len(binary)) + binary)[::-1][bit] == "1" else 0
            self.imshow(f"Bit Plane {bit}", img, subplot=True, row = 2, col = 4, num=bit+1)
        plt.show()
        
    def contrastStretching(self, s1=30, s2 = 150, r1=80, r2=150, L=255):
        plt.figure(figsize=(15,12))
        img = np.copy(self.img)
        self.imshow("Original Image", img, subplot=True, row = 2, col = 2, num=1)
        plt.subplot(2,2,3)
        plt.title("Original Histogram")
        plt.hist(self.img.ravel(), 256, [0,256])

        a = s1/r1
        b = (s2-s1)/(r2-r1)
        g = (L-s2)/(L-r2)

        rows, cols = img.shape
        for row in range(rows):
            for col in range(cols):
                if img[row][col] <= r1:
                    img[row][col] = a*img[row][col]
                elif r1 < img[row][col] <= r2:
                    r = img[row][col]
                    img[row][col] = b*(r-r1) + s1
                else:
                    r = img[row][col]
                    img[row][col] = g*(r-r2) + s2
        self.imshow("Contrast Stretching Image", img, subplot=True, row = 2, col = 2, num=2)
        plt.subplot(2,2,4)
        plt.title("Contrasted Histogram")
        plt.hist(img.ravel(),256,[0,256])
        plt.show()
            


