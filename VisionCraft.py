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
            self.img = cv2.imread(path,cv2.COLOR_BGR2GRAY)
        else:
            print("No Path for Image has been Passed")
        print("OpenCV Version: ",cv2.__version__)
        print("Matplotlib version:", matplotlib.__version__)
        print("NumPy version:", np.__version__)
        
    def imshow(self,
               title: str = "",
               image: np.ndarray = None,
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
        if subplot:
            plt.subplot(row, col, num)
            plt.imshow(image, cmap="gray")
            plt.title(title)
            plt.axis('off')
        else:
            plt.imshow(image, cmap="gray")
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
    