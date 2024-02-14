"""
TITLE: PERFORM HISTOGRAM EQUALIZATION ON AN IMAGE
AIM: 
1. To plot the Histogram of an image (Dark, Light, Low contrast, High contrast)
2. To Perform the Histogram equalization on a Low Contrast Image.
"""


def Algorithm():
    pass
def InstructionCV():
    print("""
    # Import Library
    import cv2
    from matplotlib.pyplot as plt
    
    # Plot Histogram of image
    def plot_histogram(image, title):
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        plt.plot(hist, color='black')
        plt.title(title)
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.show()
    
    
    dark_img = cv2.imread(dark_img_path, 0) 
    plot_histogram(dark_img, "Dark Image Histogram")
    
    light_img = cv2.imread(light_img_path, 0) 
    plot_histogram(light_img, "Light Image Histogram")
    
    low_contrast_img = cv2.imread(low_contrast_img_path, 0) 
    plot_histogram(low_contrast_img, "Low Contrast Image Histogram")
    
    high_contrast_img = cv2.imread(high_contrast_img_path, 0) 
    plot_histogram(high_contrast_img, "High Contrast Image Histogram")
    
    
    img_eq = cv2.equalizeHist(low_contrast_img)
    plot_histogram(img_eq, "Historgram Equilized Image")
          """)
def InstructionVC(code = False):
    pass    
