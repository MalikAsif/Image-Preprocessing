from PIL import Image, ImageEnhance
from math import sqrt
import sys
from numpy import array
import cv2
import math
import numpy as np

"""
https://www.researchgate.net/post/Can_entropy_of_contrast_enhanced_image_be_greater_than_the_original_image
https://dsp.stackexchange.com/questions/3309/measuring-the-contrast-of-an-image

Rank Zero
https://emsis.eu/olh/HTML/topics_idh_filt_rank.html
http://www.numerical-tours.com/matlab/denoisingadv_7_rankfilters/
"""

mu1 = 10
sigma1 = 10

# url = "/Users/adilyousuf/Documents/NCAI/DIP/Assignment 6/assets/Total_Pectoral_data/Images/1.png"
# img = cv2.imread(url, 0)
# s1 = np.random.normal(mu1, sigma1, 100000)
# print(s1)
# hist1 = np.histogram(img, bins=50, range=(-10,10), density=True)
# data = hist1[0]
# ent = -(data*np.ma.log(np.abs(data))).sum()
# print(ent)
#
# s1 = np.random.normal(mu1, sigma1, 100000)
# hist1 = np.histogram(img, bins=50, range=(-10, 10), density=True)
# ent = 0
# for i in hist1[0]:
#     ent -= i * math.log(abs(i))
# print(ent)


# def entropy(hist, bit_instead_of_nat=False):
#     """
#     given a list of positive values as a histogram drawn from any information source,
#     returns the entropy of its probability density function. Usage example:
#       hist = [513, 487] # we tossed a coin 1000 times and this is our histogram
#       print entropy(hist, True)  # The result is approximately 1 bit
#       hist = [-1, 10, 10]; hist = [0] # this kind of things will trigger the warning
#     """
#     h = np.asarray(hist, dtype=np.float64)
#     print(h)
#     if h.sum() <= 0 or (h < 0).any():
#         print("[entropy] WARNING, malformed/empty input %s. Returning None." % str(hist))
#         return None
#     h = h / h.sum()
#     log_fn = np.ma.log2 if bit_instead_of_nat else np.ma.log
#     return -(h * log_fn(h)).sum()
#
# s1 = np.random.normal(mu1, sigma1, 100000)
# hist1 = np.histogram(s1, bins=50, range=(-10, 10), density=True)
# print(entropy(hist1))

url = "/Users/adilyousuf/Documents/NCAI/DIP/Assignment 6/assets/DDSM/1.png"
# print(url)
img = cv2.imread(url, 0)

img = Image.fromarray(img)

hist1 = np.histogram(img, bins=50, range=(0, 1), density=True)
data = hist1[0]
ent = -(data * np.ma.log(np.abs(data))).sum()
print("Before CLAHE: ", ent)

#Increase Contrast
def calculate_brightness(image):
    greyscale_image = image.convert('L')
    histogram = greyscale_image.histogram()
    pixels = sum(histogram)
    brightness = scale = len(histogram)

    for index in range(0, scale):
        ratio = histogram[index] / pixels
        brightness += ratio * (-scale + index)

    return 1 if brightness == 255 else brightness / scale

# url = "/Users/adilyousuf/Documents/NCAI/DIP/Assignment 6/assets/DDSM/1.png"
# # print(url)
# img = cv2.imread(url, 0)
#
# img = Image.fromarray(img)
# image = Image.open("/Users/adilyousuf/Documents/NCAI/DIP/Assignment 6/assets/DDSM/1.png")
print("%s" % (calculate_brightness(img)))
print(ImageEnhance.Contrast(img).enhance(3.0))
img = ImageEnhance.Contrast(img).enhance(3.0)
img = ImageEnhance.Brightness(img).enhance(3.0)
print("%s" % (calculate_brightness(img)))

hist1 = np.histogram(img, bins=50, range=(0, 1), density=True)
data = hist1[0]
ent = -(data * np.ma.log(np.abs(data))).sum()
print("After CLAHE: ", ent)

"""
imag = Image.open("/Users/adilyousuf/Documents/NCAI/DIP/Assignment 6/assets/noise_2_poisson.png")
# Convert the image te RGB if it is a .gif for example
imag = imag.convert('RGB')
# coordinates of the pixel
X, Y = 10, 10
# Get RGB
pixelRGB = imag.getpixel((X, Y))
R, G, B = pixelRGB

print("R: ", R, ", G: ", G, ", B: ", B)

brightness = sum([R, G, B]) / 3
print("brightness: ", brightness)

# Standard
# LuminanceA = (0.2126 * R) + (0.7152 * G) + (0.0722 * B)
LuminanceA = (0.2126 * R + 0.7152 * G + 0.0722 * B)
print("LuminanceA: ", LuminanceA)
# Percieved A
# LuminanceB = (0.299 * R + 0.587 * G + 0.114 * B)
LuminanceB = (0.299 * R + 0.587 * G + 0.114 * B)
print("LuminanceB: ", LuminanceB)
# Perceived B, slower to calculate
LuminanceC = sqrt(0.299 * (R ** 2) + 0.587 * (G ** 2) + 0.114 * (B ** 2))
# LuminanceC = sqrt(0.299 * R ^ 2 + 0.587 * G ^ 2 + 0.114 * B ^ 2)
print("LuminanceC: ", LuminanceC)

print("Luminance: ", LuminanceB - LuminanceA / LuminanceB + LuminanceA)
"""
"""






# image = cv2.UMat(image)
img = array(img)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
img = clahe.apply(img)

img = Image.fromarray(img)
# image = Image.open("/Users/adilyousuf/Documents/NCAI/DIP/Assignment 6/assets/noise_2_poisson.png")
print("%s" % (calculate_brightness(img)))

if __name__ == '__main__':
    for file in sys.argv[1:]:
        image = Image.open(file)
        print("%s\t%s" % (file, calculate_brightness(image)))

"""