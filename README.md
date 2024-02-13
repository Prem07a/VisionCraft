<p align="center">
  <img src="./VC_logo.png" alt="VisionCraft Logo" width="700"/>
</p>


# VisionCraft - A Custom Image Processing Library

VisionCraft is a Python library designed for image processing, providing a collection of functions for filtering, processing, and transformations. It offers a set of modules to address various aspects of image manipulation.

## Table of Contents

- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
- [vision/](#vision)
   - [filter.py](#filterpy)
   - [processing.py](#processingpy)
   - [transform.py](#transformpy)
   - [utils.py](#utilspy)
- [craft/](#craft)
- [Project Status and Future Development](#project-status-and-future-development)
- [Author](#author)
- [License](#license)

## Directory Structure

```
VisionCraft/
│
├── vision/
│   ├── __init__.py
│   ├── filter.py
│   ├── processing.py
│   ├── transform.py
│   ├── utils.py
│
├── craft/
│   # Under Development
│   ├── __init__.py
│   
│   
│
README.md
|
./requirements.txt
```

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/Prem07a/VisionCraft.git
```


1. **Create a Virtual Environment:**
   ```bash
   python -m venv env
   ```
   This command creates a virtual environment named `env`.

2. **Activate the Virtual Environment (Optional, Depending on OS):**
   - **On Windows:**
     ```bash
     .\env\Scripts\activate
     ```
   - **On macOS and Linux:**
     ```bash
     source env/bin/activate
     ```
   Activating the virtual environment isolates your Python environment, ensuring that packages are installed only within this environment.

3. **Install Dependencies from requirements.txt:**
   ```bash
   pip install -r requirements.txt
   ```
   This command installs the Python packages listed in the `requirements.txt` file. These packages are necessary for your project and may include libraries, frameworks, or tools required for running the code.

After following these steps, your virtual environment should be set up, activated, and the project dependencies installed. You can then proceed with running your Python scripts or working on your VisionCraft project within this isolated environment.

## Usage

Import VisionCraft modules into your Python script or Jupyter notebook:

```python
# Example: Importing filter functions
from VisionCraft.vision.filter import boxFilter

# Example: Importing processing functions
from VisionCraft.vision.processing import imgNegative

# Example: Importing transform functions
from VisionCraft.vision.transform import grayLevelSlicing

# Example: Importing utility functions
from VisionCraft.vision.utils import imshow
```

## `vision/`

### 1. `filter.py`

**Module for Image Filtering Operations**

This module (`filter.py`) focuses on various image filtering operations. Image filtering is a fundamental technique in image processing that helps enhance or suppress specific features within an image. The module currently includes the following function:

#### Functions:

- **`boxFilter`**
  - Description: Applies a box filter to the input image, which is a simple and commonly used filter for smoothing or blurring images.
  - Parameters:
    - `img`: Input image as a NumPy array.
    - `path`: Path to an image file. Either `img` or `path` should be provided.
    - `filter_size`: Size of the filter (odd number).
    - `show`: Boolean indicating whether to display the original and filtered images.
    - `height`: Height of the displayed images (for visualization).
    - `width`: Width of the displayed images (for visualization).
  - Returns: Filtered image as a NumPy array.

- **`weightedAvgFilter`**
  - *Description*: Applies a 3x3 weighted average filter to the input image, offering a balanced smoothing effect.
  - *Parameters*:
    - `img` (np.ndarray): Input image as a NumPy array.
    - `path` (str): Path to an image file. Either `img` or `path` should be provided.
    - `show` (bool): Boolean indicating whether to display the original and filtered images.
    - `height` (int): Height of the displayed images (for visualization).
    - `width` (int): Width of the displayed images (for visualization).
  - *Returns*: np.ndarray - Filtered image.

- **`medianFilter`**
  - *Description*: Applies a median filter to the input image, effectively reducing noise and preserving edges.
  - *Parameters*:
    - `img` (np.ndarray): Input image as a NumPy array.
    - `path` (str): Path to an image file. Either `img` or `path` should be provided.
    - `filter_size` (int): Size of the median filter (odd number recommended).
    - `show` (bool): Boolean indicating whether to display the original and filtered images.
    - `height` (int): Height of the displayed images (for visualization).
    - `width` (int): Width of the displayed images (for visualization).
  - *Returns*: np.ndarray - Filtered image.

### 2. `processing.py`

**Module for Image Processing Operations**

This module (`processing.py`) encompasses various image processing operations. Image processing involves manipulating an image to achieve a desired result. The module currently includes the following functions:

#### Functions:

- **`imgNegative`**
  - *Description*: Computes the negative of an input image, enhancing the visibility of details and features.
  - *Parameters*:
    - `img` (np.ndarray): Input image as a NumPy array.
    - `path` (str): Path to an image file. Either `img` or `path` should be provided.
    - `show` (bool): Boolean indicating whether to display the original and negated images.
    - `height` (int): Height of the displayed images (for visualization).
    - `width` (int): Width of the displayed images (for visualization).
  - *Returns*: Negated image as a NumPy array.

- **`imgLog`**
  - *Description*: Applies a logarithmic transformation to the input image, enhancing low-intensity details.
  - *Parameters*:
    - `img` (np.ndarray): Input image as a NumPy array.
    - `path` (str): Path to an image file. Either `img` or `path` should be provided.
    - `show` (bool): Boolean indicating whether to display the original and transformed images.
    - `height` (int): Height of the displayed images (for visualization).
    - `width` (int): Width of the displayed images (for visualization).
  - *Returns*: Transformed image as a NumPy array.

- **`powerLaw`**
  - *Description*: Applies a power-law transformation to the input image, adjusting the intensity values.
  - *Parameters*:
    - `img` (np.ndarray): Input image as a NumPy array.
    - `path` (str): Path to an image file. Either `img` or `path` should be provided.
    - `height` (int): Height of the displayed images (for visualization).
    - `width` (int): Width of the displayed images (for visualization).
    - `show` (bool): Boolean indicating whether to display the original and transformed images.
    - `gamma` (float): Gamma parameter for the power-law transformation.
  - *Returns*: Transformed image as a NumPy array.

- **`flipImg`**
  - *Description*: Flips the input image both vertically and horizontally.
  - *Parameters*:
    - `img` (np.ndarray): Input image as a NumPy array.
    - `path` (str): Path to an image file. Either `img` or `path` should be provided.
    - `show` (bool): Boolean indicating whether to display the original and flipped images.
    - `height` (int): Height of the displayed images (for visualization).
    - `width` (int): Width of the displayed images (for visualization).
  - *Returns*: Tuple[np.ndarray, np.ndarray] -> Tuple containing horizontally and vertically flipped images as NumPy arrays.


### 3. `transform.py`

**Module for Image Transformation Operations**

This module (`transform.py`) provides functions for image transformations, including gray level slicing, bit-plane slicing, contrast stretching, and histogram equalization.

#### Functions:

- **`grayLevelSlicing`**
  - *Description*: Performs gray level slicing on the input image, highlighting a specific range of gray levels.
  - *Parameters*:
    - `img` (np.ndarray): Input image as a NumPy array.
    - `path` (str): Path to an image file. Either `img` or `path` should be provided.
    - `lower` (int): Lower bound for gray level slicing.
    - `upper` (int): Upper bound for gray level slicing.
    - `bg` (bool): If True, retains the background in the sliced image.
    - `THRESHOLD` (int): Threshold value for gray level slicing.
    - `show` (bool): Boolean indicating whether to display the original and sliced images.
    - `height` (int): Height of the displayed images (for visualization).
    - `width` (int): Width of the displayed images (for visualization).
  - *Returns*: The sliced image as a NumPy array.

- **`bitPlaneSlicing`**
  - *Description*: Performs bit-plane slicing on the input image, extracting information at different bit levels.
  - *Parameters*:
    - `img` (np.ndarray): Input image as a NumPy array.
    - `path` (str): Path to an image file. Either `img` or `path` should be provided.
    - `show` (bool): If True, displays the original and bit-plane sliced images.
    - `height` (int): Height of the displayed images (for visualization).
    - `width` (int): Width of the displayed images (for visualization).
  - *Returns*: List containing bit-plane sliced images as NumPy arrays.

- **`contrastStretching`**
  - *Description*: Adjusts the contrast of the input image using stretching techniques.
  - *Parameters*:
    - `img` (np.ndarray): Input image as a NumPy array.
    - `path` (str): Path to an image file. Either `img` or `path` should be provided.
    - `s1` (int): Stretching value for the lower limit.
    - `s2` (int): Stretching value for the upper limit.
    - `r1` (int): Original lower limit.
    - `r2` (int): Original upper limit.
    - `L` (int): Maximum gray level.
    - `show` (bool): Boolean indicating whether to display the original and contrast-stretched images.
    - `height` (int): Height of the displayed images (for visualization).
    - `width` (int): Width of the displayed images (for visualization).
  - *Returns*: The contrast-stretched image as a NumPy array.

- **`histogramEquilization`**
  - *Description*: Performs histogram equalization on the input image, enhancing the overall contrast.
  - *Parameters*:
    - `img` (np.ndarray): Input image as a NumPy array.
    - `path` (str): Path to an image file. Either `img` or `path` should be provided.
    - `show` (bool): If True, displays the original and equalized images.
    - `height` (int): Height of the displayed images (for visualization).
    - `width` (int): Width of the displayed images (for visualization).
    - `eq_table` (bool): If True, returns the equalization table along with the equalized image.
  - *Returns*: The equalized image as a NumPy array. If `eq_table` is True, returns the equalization table as well.

### 4. `utils.py`

**Module for Utility Functions Related to Image Processing**

This module (`utils.py`) provides utility functions for displaying images and plotting various transformations.

#### Functions:

- **`imShow`**
  - *Description*: Display an image using Matplotlib.
  - *Parameters*:
    - `title` (str, optional): Title of the displayed image.
    - `image` (np.ndarray, optional): Image as a NumPy array.
    - `path` (str, optional): Path to the image file. If provided, 'image' parameter is ignored.
    - `subplot` (bool, optional): If True, displays the image as a subplot.
    - `row` (int, optional): Row position for the subplot.
    - `col` (int, optional): Column position for the subplot.
    - `num` (int, optional): Number of the subplot.
  - *Returns*: None
  - *Note*: If 'path' is provided but the image is not found, a message is printed, and None is returned.

- **`plotLogTransform`**
  - *Description*: Visualize logarithmic transformations and their inverses.
  - *Parameters*:
    - `height` (int, optional): Height of the Matplotlib figure.
    - `width` (int, optional): Width of the Matplotlib figure.
  - *Returns*: None

- **`plotPowerLaw`**
  - *Description*: Visualize power-law transformations with different gamma values.
  - *Parameters*:
    - `height` (int, optional): Height of the Matplotlib figure.
    - `width` (int, optional): Width of the Matplotlib figure.
  - *Returns*: None


##  `craft /`

**COMING SOON**

## Project Status and Future Development

- **Early Development Stage:** This version of VisionCraft is currently in its early stages of development. It is actively being developed, and changes are expected as new features and improvements are introduced.

- **Regular Updates:** Expect regular updates, new methods, and bug fixes to be consistently released to enhance the library's capabilities. We are committed to improving and expanding VisionCraft to meet the evolving needs of the image processing community.

- **Upcoming Feature: `Craft`:** The introduction of the `Craft` is anticipated shortly. This powerful feature will provide advanced capabilities for neural network construction, expanding the toolkit's functionality and opening up new possibilities for image processing tasks.

We appreciate your interest in VisionCraft and invite you to stay tuned for exciting developments!

## Author

<img src="https://i.ibb.co/GWDGXJM/Prem-Ganesh-Gaikwad-1.png" alt="Prem-Ganesh-Gaikwad-1" border="0" style="width: 30%; height: auto;">


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

