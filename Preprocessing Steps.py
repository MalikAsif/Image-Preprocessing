# """
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import estimate_sigma
import os
from PIL import Image, ImageEnhance, ImageFilter

# Plotting Setting
w = 10
h = 10
fig = plt.figure(figsize=[10, 10])
columns = 4
# rows = 5

# prep (x,y) for extra plotting
xs = np.linspace(0, 5, 2, endpoint=False)  # from 0 to 2pi
# ys = np.zeros(2)#np.abs(xs)  # absolute of sine

# ax enables access to manipulate each of subplots
ax = []

structs = []


class img_data:
    def __init__(self, img_name, noise, filter_name, noise_value_after, contrast_value, con_after_enh):
        self.img_name = img_name
        self.noise = noise
        self.filter_name = filter_name
        self.noise_value_after = noise_value_after
        self.contrast_value = contrast_value
        self.con_after_enh = con_after_enh


def signaltonoise(a, axis=None, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    # If axis is equal to None, the array is first ravel'd. If axis is an
    #         integer, this is the axis over which to operate. Default is 0.
    # Degrees of freedom correction for standard deviation. Default is 0.
    sd = a.std(axis=axis, ddof=ddof)
    sigma = estimate_sigma(a, multichannel=False, average_sigmas=True)
    # The mean to standard deviation ratio(s) along `axis`, or 0 where the standard deviation is 0.
    return np.where(sd == 0, 0, m / sd)


def noiseEstimation():
    for i in range(1318):
        rows = 10

        org_img = i + 1
        print(org_img)
        exist = os.path.exists(
            "/Users/adilyousuf/Documents/NCAI/DIP - First Paper/Assignment 6/assets/Preprocessed Data/data/train/Malignant/{0}.png".format(
                i + 1))
        if not exist:
            continue

        url = "/Users/adilyousuf/Documents/NCAI/DIP - First Paper/Assignment 6/assets/Preprocessed Data/data/train/Malignant/{0}.png".format(
            i + 1)
        print(url)
        img = cv2.imread(url, 0)
        value = signaltonoise(img)
        structs.append(img_data(org_img, value, "", 0, 0, 0))
        print(value)

        """Apply Filters"""

        """Gaussian Filter"""
        filtering = cv2.GaussianBlur(img, (5, 5), 0)
        value = signaltonoise(filtering)
        print("GaussianBlur:", value)
        img_clahe = filtering

        """Bilateral Filter"""
        filtering = cv2.bilateralFilter(img, 9, 75, 75)
        if value > signaltonoise(filtering):
            value = signaltonoise(filtering)
            img_clahe = filtering
        print("bilateralFilter:", signaltonoise(filtering))

        filtering = cv2.medianBlur(img, 3)
        if value > signaltonoise(filtering):
            value = signaltonoise(filtering)
            img_clahe = filtering
        print("medianBlur:", signaltonoise(filtering))

        kernel = np.ones((3, 3), np.float32) / 9
        filtering = cv2.filter2D(img, -1, kernel)
        if value > signaltonoise(filtering):
            value = signaltonoise(filtering)
            img_clahe = filtering
        print("filter2D:", signaltonoise(filtering))

        """Applying CLAHE"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
        out = clahe.apply(img_clahe)

        """#Adding Sharpening"""
        out = Image.fromarray(out)
        out = out.filter(ImageFilter.SHARPEN)

        new_img = "/Users/adilyousuf/Documents/NCAI/DIP - First Paper/Assignment 6/assets/Preprocessed Data/2/train/Malignant/{0}.png".format(
            i + 1)
        sharpen2 = np.array(out)
        cv2.imwrite(new_img, sharpen2)


noiseEstimation()
for i in structs:
    print("*************************************************")
    print("Image Name: ", i.img_name,
          ",\nNoise: ", i.noise,
          ",\nFilter Name: ", i.filter_name,
          ",\nNoise Value After: ", i.noise_value_after)
    # , ", Contrast Value: ", i.contrast_value, ", Contrast After Enhancement: ", i.con_after_enh)
    print("*************************************************")
    print("*************************************************")
    print("*************************************************")

# plotting
"""
https://scikit-image.org/docs/dev/auto_examples/filters/plot_rank_mean.html
"""
