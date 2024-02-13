"""
TITLE: APPLY SMOOTHING SPATIAL FILTERS ON AN IMAGE 
AIM: 
To implement :
1. Box Filter
2. Weighted Filter
3. Median Filter
"""


def Algorithm():
    print("""
    >>  Box Filter
    
        1. Input Handling:
        - Check if `img` is provided.
        - If not, read the image from the specified `path` using `cv2.imread()`.
        - If the image is still None, print an error message and exit the function.

        2. Filter Size Validation:
        - Check if `filter_size` is even.
            - If true, print a message suggesting odd values for better results.

        3. Image Dimensions:
        - Get the dimensions (rows, cols) of the input image.

        4. Padding:
        - Pad the input image with a constant value (e.g., 255) to handle the filter along the edges.

        5. Initialization:
        - Create an empty array `filtered_img` with the same shape as the input image.

        6. Filtering Process:
        - Loop through each pixel (row, col) in the input image:
            - Initialize a variable `region` to store the sub-image centered around the current pixel.
            - Extract a region from the padded image centered around the current pixel with size `filter_size`.
            - Calculate the average of the pixel values in the region.
            - Assign the calculated average to the corresponding pixel in `filtered_img`.

        7. Visualization (Optional):
        - If the `show` parameter is True:
            - Create a matplotlib figure with specified `height` and `width`.
            - Add the original image as the first subplot.
            - Add the filtered image as the second subplot.
            - Display the figure.

        8. Return:
        - Return the filtered image.

        9. Function Call:
        - Call the `boxFilter` function with the desired parameters.

    >> Weighted Filter (Gaussian Blur)
    
        1. Input Handling:
        - Check if `img` is provided.
            - If not, read the image from the specified `path` using `imRead`.
            - If the image is still None, print an error message and exit the function.

        2. Filter Definition:
        - Define the weighted filter as a 3x3 matrix.

        3. Image Dimensions:
        - Get the dimensions (rows, cols) of the input image.

        4. Padding:
        - Pad the input image with a constant value (e.g., 255) to handle the filter along the edges.

        5. Initialization:
        - Create an empty array `filtered_img` with the same shape as the input image.

        6. Filtering Process:
        - Loop through each pixel (row, col) in the input image:
            - Initialize a variable `replace` to store the weighted average of the pixel values.
            - Extract a region from the padded image centered around the current pixel with a 3x3 filter.
            - Calculate the weighted average of the pixel values in the region using the defined filter.
            - Assign the calculated weighted average to the corresponding pixel in `filtered_img`.

        7. Visualization (Optional):
        - If the `show` parameter is True:
            - Create a matplotlib figure with specified `height` and `width`.
            - Add the original image as the first subplot.
            - Add the filtered image as the second subplot.
            - Display the figure.

        8. Return:
        - Return the filtered image.

        9. Function Call:
        - Call the `weightedAvgFilter` function with the desired parameters.
    
    >>> Median Filter
    
        1. Input Handling:
        - Check if `img` is provided.
            - If not, read the image from the specified `path` using `imRead`.
            - If the image is still None, print an error message and exit the function.

        2. Filter Size Validation:
        - Check if `filter_size` is even.
            - If true, print a message suggesting odd values for better results.

        3. Image Dimensions:
        - Get the dimensions (rows, cols) of the input image.

        4. Padding:
        - Pad the input image with a constant value (e.g., 255) to handle the filter along the edges.

        5. Initialization:
        - Create an empty array `filtered_img` with the same shape as the input image.

        6. Filtering Process:
        - Loop through each pixel (row, col) in the input image:
            - Calculate the median of the pixel values in the region using the defined filter.
            - Assign the calculated median to the corresponding pixel in `filtered_img`.

        7. Visualization (Optional):
        - If the `show` parameter is True:
            - Create a matplotlib figure with specified `height` and `width`.
            - Add the original image as the first subplot.
            - Add the filtered image as the second subplot.
            - Display the figure.

        8. Return:
        - Return the filtered image.

        9. Function Call:
        - Call the `medianFilter` function with the desired parameters.
          """)

def InstructionCV():
    print("""
    # Import Libraries
    import cv2
    
    image = cv2.imread('your_image_path.jpg')

    # 1. Box Filter
    box_filtered = cv2.boxFilter(image, -1, (3, 3))  # Kernel size is (3, 3), -1 indicates the same depth as the input image

    # 2. Weighted Filter (Gaussian Blur)
    weighted_filtered = cv2.GaussianBlur(image, (3, 3), 0)  # Kernel size is (3, 3), 0 for automatic standard deviation

    # 3. Median Filter
    median_filtered = cv2.medianBlur(image, 3)     
          """)
    
def InstructionVC(code = False):
    print("""
    
    # Import Libraries
    from VisionCraft.vision.utils import imShow, imRead
    from VisionCraft.vision.filter import boxFilter, weightedAvgFilter, medianFilter

    # Read Image
    img = imRead('image_path', show = True) # show = True to display the Image
    
    # 1. Box Filter
    box_filtered = boxFilter(img, filter_size = 7, show = True) # Displays the original and filtered images.
          """)
    if code:
        print("""
    
    >>> Code:        
    
    def boxFilter(img:np.ndarray = None, 
              path: str = "", 
              filter_size:int = 3, 
              show:bool = False, 
              height:int = 10, 
              width:int = 8) -> Union[np.ndarray,None]:
        if image is None:
            image = imRead(path)
            if image is None:
                return image
            
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
  
              """)
    print("""
    # 2. Weighted Filter (Gaussian Blur)
    weighted_filtered = weightedAvgFilter(img, show = True)
          """)
    
    if code:
        print("""
    >>> Code:
    
    def weightedAvgFilter(img:np.ndarray = None, 
                      path: str = "", 
                      show:bool = False, 
                      height:int = 10, 
                      width:int = 8) -> Union[np.ndarray,None]:  
        if image is None:
            image = imRead(path)
            if image is None:
                return image
            
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

              """)
    print("""
    # 3. Median Filter
    median_filtered = medianFilter(img, filter_size = 7, show = True)
          """)
    
    if code:
        print("""
    
    >>> Code:
    
    def medianFilter(img:np.ndarray = None, 
                 path: str = "", 
                 filter_size : int = 3,
                 show:bool = False, 
                 height:int = 10, 
                 width:int = 8) -> Union[np.ndarray,None]:  
        if image is None:
            image = imRead(path)
            if image is None:
                return image
            
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
              """)
