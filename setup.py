from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.1'
DESCRIPTION = 'Computer Vision Library to help do CV much faster'
LONG_DESCRIPTION = 'Simplifying computer vision tasks with Python. Visualize, transform, and analyze images effortlessly. Ideal for students, researchers, and developers. Enhance your computer vision projects!'

# Setting up
setup(
    name="VisionCraft",
    version=VERSION,
    author="Prem Gaikwad",
    author_email="premgaikwad7a@gmail.com",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['opencv-python', 'numpy', 'pandas', 'matplotlib'],
    keywords=['python', 'CNN', 'cv'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)