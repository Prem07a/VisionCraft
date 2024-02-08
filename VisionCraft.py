"""
    Creator: Prem Gaikwad
    Student from PICT
"""

import cv2
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class vision:
    """
    A simple computer vision utility class using OpenCV, Matplotlib, and NumPy.

    Parameters:
    - path (str): The path to the image file (optional).
    - height (int): Height for figure size in Matplotlib (default: 10).
    - width (int): Width for figure size in Matplotlib (default: 8).

    Attributes:
    - img (numpy.ndarray): The image loaded using OpenCV.
    - h (int): Height for figure size in Matplotlib.
    - w (int): Width for figure size in Matplotlib.

    Methods:
    - imshow(title="", image=None, given=False, subplot=False, row=0, col=0, num=0):
        Display an image using Matplotlib.

        Parameters:
        - title (str): Title of the displayed image.
        - image (numpy.ndarray): The image data.
        - given (bool): If True, use the class attribute 'img' as the image data.
        - subplot (bool): Whether to use subplot (default: False).
        - row (int): Number of rows in the subplot grid (if subplot is True).
        - col (int): Number of columns in the subplot grid (if subplot is True).
        - num (int): The index of the subplot (if subplot is True).

    - imgNegative():
        Display the original and negative image.

    - imgLog():
        Display the original and logarithmically transformed image.

    - plotLogTransform():
        Plot the logarithmic transformation and its inverse.

    - powerLaw():
        Display the original image and its power-law transformations.

    - plotPowerLaw():
        Plot power-law transformations with different gamma values.

    - flipImg():
        Display the original image, vertical flip, and horizontal flip.

    - grayLevelSlicing(lower=100, upper=200, bg=False, THRESHOLD=256):
        Display the original image and grayscale level slicing.

    - bitPlaneSlicing():
        Display bit-plane sliced images.

    - contrastStretching(s1=30, s2=150, r1=80, r2=150, L=255):
        Display the original image and its contrast-stretched version.

    - histogramEquilization():
        Display the original image, its histogram, and the equalized histogram.
    """
    def __init__(self, path: str = None, height=10, width=8) -> None:
        """
        Initialize the Vision class.

        Parameters:
        - path (str): The path to the image file (optional).
        - height (int): Height for figure size in Matplotlib (default: 10).
        - width (int): Width for figure size in Matplotlib (default: 8).
        """
        self.h = height 
        self.w = width
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
        - given (bool): If True, use the class attribute 'img' as the image data.
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
        plt.figure(figsize=(self.h,self.w))
        c = 255 / np.log(1 + np.max(np.array(self.img)))
        img_log = c * np.log(1 + np.array(self.img))    
        self.imshow("Original Image",self.img, subplot=True, row=1, col=2, num=1)
        self.imshow("Logarithmic Transformation",img_log, subplot=True, row=1, col=2, num=2)
        plt.show()

    def plotLogTransform(self):
        """
        Plot the logarithmic transformation and its inverse graph.
        """
        plt.figure(figsize=(self.h,self.w))
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
        """
        Display the original image and its power-law transformations.
        """
        plt.figure(figsize=(self.h,self.w))
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
        """
        Plot power-law transformations graph with different gamma values.
        """
        plt.figure(figsize=(self.h,self.w))
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
        """
        Display the original image, vertical flip, and horizontal flip.
        """        
        plt.figure(figsize=(self.h,self.w))
        self.imshow("Original Image",self.img, subplot=True, row=1, col=3, num=1)           
        img_flip_v = cv2.flip(self.img, 0)
        self.imshow("Vertical Flip",img_flip_v, subplot=True, row=1, col=3, num=2)
        img_flip_h = cv2.flip(self.img, 1)
        self.imshow("Horizontal Flip",img_flip_h, subplot=True, row=1, col=3, num=3)
        plt.show()
        
    def grayLevelSlicing(self, lower:int=100, upper:int=200, bg:bool=False, THRESHOLD = 256):
        """
        Display the original image and grayscale level slicing.

        Parameters:
        - lower (int): Lower bound for slicing.
        - upper (int): Upper bound for slicing.
        - bg (bool): If True, include the background in the sliced image.
        - THRESHOLD (int): Value to replace with.
        """
        plt.figure(figsize=(self.h,self.w))
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
        """
        Display bit-plane sliced images.
        """
        plt.figure(figsize=(self.h,self.w))
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
        """
        Display the original image and its contrast-stretched version.

        Parameters:
        - s1 (int): Stretch value for region 1.
        - s2 (int): Stretch value for region 2.
        - r1 (int): Pixel value for region 1.
        - r2 (int): Pixel value for region 2.
        - L (int): Maximum pixel value.
        """
        plt.figure(figsize=(self.h,self.w))
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
        
    def histogramEquilization(self):
        """
        Display the original image, its histogram, and the equalized histogram.
        """
        plt.figure(figsize=(self.h,self.w))
            
        self.imshow("Original Image", self.img, subplot=True, row=2, col=2, num=1)
        img = np.copy(self.img)
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
        plt.subplot(2,2,2)
        plt.bar(freq.keys(), freq.values())
        plt.title("Original Histogram")
        plt.xlabel("Gray Level")
        plt.ylabel("Frequency")

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
        plt.subplot(2,2,4)
        grouped_df = df[['New_Histogram', 'Nk']].groupby('New_Histogram').sum().reset_index()
        plt.bar(grouped_df['New_Histogram'], grouped_df['Nk'])
        plt.title("Equalized Histogram")
        plt.xlabel("New Gray Level")
        plt.ylabel("Frequency")
        freq = {}
        for row in range(img.shape[0]):
            for col in range(img.shape[1]):
                r = img[row][col]
                img[row][col] = df.loc[r,"New_Histogram"]
        self.imshow("Histogram Equilization", img, row=2, col=2, num=3, subplot=True)
        plt.show()

