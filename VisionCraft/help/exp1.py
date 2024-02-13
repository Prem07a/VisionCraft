"""
TITLE: SIMPLE IMAGE PROCESSING WITH OPENCV
AIM: 
Implementing various OpenCV commands using Python.

"""

def Instruction() -> None:
    print("""
    
    Installation: !pip install opencv-python
    
    import cv2
              
    # 1. Read Image: 
        cv2.imread(path, flag)
        - flag = 0 for B/W image
        - flag = 1 for BGR image
        
    # 2. Display Image: 
        cv2.imshow(img: np.ndarray)
    
    # 3. Resize Image: 
        cv2.resize(input_image, output_size, fx, fy, interpolation)
        - input_image: input image (mandatory)
        - output_size: desired output size (mandatory)
        - fx: horizontal scale factor (optional)
        - fy: vertical scale factor (optional)
        - interpolation: method for resizing (e.g., INTER_NEAREST, INTER_LINEAR, INTER_AREA)
        
    # 4. Rotate Image: 
        cv2.rotate(img, rotateCode)
        - rotateCode options:
            - cv2.ROTATE_90_CLOCKWISE
            - cv2.ROTATE_180
            - cv2.ROTATE_90_COUNTERCLOCKWISE

    # 5. Add Images: 
        cv2.add(img1, img2)
    
    # 6. Histogram Function: 
        cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
        - channels: list of channels
        - mask: optional mask
        - histSize: histogram sizes
        - ranges: bin boundaries
        - hist: Output histogram
        - accumulate: accumulation flag

    # 7. Color Conversions: 
        cv2.cvtColor(src, code[, dst[, dstCn]])
        - src: Input image
        - code: Color space conversion code
        - dst: Output image (optional)
        - dstCn: Number of channels in the destination image (optional)

    # 8. Thresholding: 
        cv2.threshold(src, thresh, maxVal, type[, dst])
        - src: Input image
        - thresh: Threshold value
        - maxVal: Maximum value to use with the chosen thresholding type
        - type: Thresholding type (e.g., cv2.THRESH_BINARY)
        - dst: Output image (optional)

    # 9. Add Noise: 
        cv2.randn(dst, mean, stddev)
        - dst: Output array where the noise is added
        - mean: Mean of the Gaussian distribution (center of the noise)
        - stddev: Standard deviation of the Gaussian distribution (controls the spread of the noise)
    """)
