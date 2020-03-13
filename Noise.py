# """
import cv2
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import estimate_sigma
from numpy import array
import os

img_path1 = '/Users/adilyousuf/Documents/NCAI/DIP/Assignment 6/assets/gdorleans_noise_poisson.png'  # '/Users/adilyousuf/Documents/NCAI/DIP/Assignment 4/Histogram_Equalization/assets/low_contrast_result.png'  # 5.401869530505883
img_path2 = '/Users/adilyousuf/Documents/NCAI/DIP/Assignment 6/assets/noise_2_poisson.png'  # '/Users/adilyousuf/Documents/NCAI/DIP/Assignment 4/Histogram_Equalization/assets/low_contrast.png'  # 9.751941155343086
img_path3 = '/Users/adilyousuf/Documents/NCAI/DIP/Assignment 6/assets/photogravure2_poisson.png'  # '/Users/adilyousuf/Documents/NCAI/DIP/Assignment 4/Histogram_Equalization/Ass_05/assets/grey/lena.png'  #

"""
m/sd
5.401869530505883
9.751941155343086
2.9492450488650688

m/sigma
28.538806677417046
72.77306469392597
55.84269080683082
"""

# lena =                          '/Users/adilyousuf/Documents/NCAI/DIP/Assignment 6/assets/Color/lena.png'
# lena_gaussian_filter =          '/Users/adilyousuf/Documents/NCAI/DIP/Assignment 6/assets/Color/lena_gaussian_filter.png'
# lena_gaussian_noise =           '/Users/adilyousuf/Documents/NCAI/DIP/Assignment 6/assets/Color/lena_gaussian_noise.png'
# lena_gaussian_noise_filter =    '/Users/adilyousuf/Documents/NCAI/DIP/Assignment 6/assets/Color/lena_gaussian_noise_filter.png'

lena = '/Users/adilyousuf/Documents/NCAI/DIP/Assignment 6/assets/Grey/lena.png'
lena_gaussian_filter = '/Users/adilyousuf/Documents/NCAI/DIP/Assignment 6/assets/Grey/lena_gaussian_filter.png'
lena_gaussian_noise = '/Users/adilyousuf/Documents/NCAI/DIP/Assignment 6/assets/Grey/lena_gaussian_noise.png'
lena_gaussian_noise_filter = '/Users/adilyousuf/Documents/NCAI/DIP/Assignment 6/assets/Grey/lena_gaussian_noise_filter.png'

"""
#Color

m/sd
Original
2.9492450488650688
Filter on Original
2.6246159627232064
Noise Original
2.5816025892696217
Filter Noise Original
2.6439223201901143

m/sigma
Original
55.84269080683082
Filter on Original
125.85485310951628
Noise Original
40.90017403835715
Filter Noise Original
71.08960158288625

#Grey

m/sd
Original
2.9492450488650688
Filter on Original
2.9946403944255287
Noise Original
3.0328252734665813
Filter Noise Original
3.0936075239326

m/sigma
Original
55.84269080683082
Filter on Original
155.5075594462127
Noise Original
24.480402221384832
Filter Noise Original
100.6198805021206
"""

# Plotting Setting
w = 10
h = 10
fig = plt.figure(figsize=[10, 10])
columns = 3
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


def addImage(i, img, rows, title):
    # print("rows:", rows, "columns:", columns, "i:", i)
    fig.add_subplot(rows, columns, i)
    plt.axis("off")
    plt.title(title)
    # plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def noiseEstimation():
    rows = 3  # len(imgArr)
    for i in range(rows):

        # org_img = os.path.basename(i+1)
        # print(org_img)
        # img_name = os.path.basename(i+1)
        org_img = i + 1  # img_name.replace(".png", "")
        print(org_img)
        url = "/Users/adilyousuf/Documents/NCAI/DIP/Assignment 6/assets/DDSM/{0}.png".format(i + 1)
        print(url)
        img = cv2.imread(url, 0)
        #         img = Image.open(url)
        value = signaltonoise(img)
        structs.append(img_data(org_img, value, "", 0, 0, 0))
        print(value)

        #         print("line: 144", "rows:", rows, "columns:", columns, "i:", 2 * i)

        """START"""

        """Applying CLAHE"""
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
        # out = clahe.apply(img)
        # org_img = "Histogram Equalization"
        # addImage(columns * i + 1, out, rows, org_img)

        addImage(columns * i + 1, img, rows, i+1)

        """END"""

        """Apply Filters"""

        """Gaussian Filter"""
        filtering = cv2.GaussianBlur(img, (5, 5), 0)
        # org_img = org_img + "_Gaussian"
        # img_path = "/Users/adilyousuf/Documents/NCAI/DIP/Assignment 6/assets/DDSM/" + img_name.replace(".png",
        #                                                                                                "") + "_gaussian.png"
        # print(img_path)
        # cv2.imwrite(img_path, filtering)
        value = signaltonoise(filtering)
        print("GaussianBlur:", value)
        img_clahe = filtering
        structs[i].filter_name = "GaussianBlur"
        structs[i].noise_value_after = value
        # addImage(5 * i+1, filtering , rows)

        """Bilateral Filter"""
        filtering = cv2.bilateralFilter(img, 9, 75, 75)
        # img_path = "/Users/adilyousuf/Documents/NCAI/DIP/Assignment 6/assets/Grey/" + img_name.replace(".png",
        #                                                                                                "") + "_bilateral.png"
        # print(img_path)
        # cv2.imwrite(img_path, filtering)
        if value < signaltonoise(filtering):
            value = signaltonoise(filtering)
            img_clahe = filtering
            # org_img = org_img + "_Bilateral"
            structs[i].filter_name = "bilateralFilter"
            structs[i].noise_value_after = value
        # value = signaltonoise(filtering) if value < signaltonoise(filtering) else value
        print("bilateralFilter:", signaltonoise(filtering))
        # addImage(5 * i+2, filtering, rows)

        """Median Filter"""
        filtering = cv2.medianBlur(img, 3)
        # img_path = "/Users/adilyousuf/Documents/NCAI/DIP/Assignment 6/assets/Grey/" + img_name.replace(".png",
        #                                                                                                "") + "_median.png"
        # print(img_path)
        # cv2.imwrite(img_path, filtering)
        if value < signaltonoise(filtering):
            value = signaltonoise(filtering)
            img_clahe = filtering
            # org_img = org_img + "_Medium"
            structs[i].filter_name = "medianBlur"
            structs[i].noise_value_after = value
        # value = signaltonoise(filtering)
        print("medianBlur:", signaltonoise(filtering))

        """START"""

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
        out = clahe.apply(filtering)
        org_img = "Median Histogram Equalization"
        addImage(columns * i + 2, out, rows, org_img)


        # addImage(columns * i + 2, filtering, rows, "Median FIlter")
        """END"""
        # addImage(5 * i+3, filtering, rows)

        # processed_image = cv2.meanBlur(img, 3)
        # 2D Convolution ( Image Filtering )
        kernel = np.ones((3, 3), np.float32) / 9
        filtering = cv2.filter2D(img, -1, kernel)
        # img_path = "/Users/adilyousuf/Documents/NCAI/DIP/Assignment 6/assets/Grey/" + img_name.replace(".png",
        #                                                                                                "") + "_filter2D.png"
        # print(img_path)
        # cv2.imwrite(img_path, filtering)
        if value < signaltonoise(filtering):
            value = signaltonoise(filtering)
            img_clahe = filtering
            # org_img = org_img + "_Filter2D"
            structs[i].filter_name = "filter2D"
            structs[i].noise_value_after = value
        # value = signaltonoise(filtering)
        print("filter2D:", signaltonoise(filtering))
        # addImage(5 * i+4, filtering, rows)

        """#Adding Image After Filtering"""
        # addImage(columns * i + 3, img_clahe, rows, structs[i].filter_name)

#         """#Adding Sharpening"""
#         blurred_f = ndimage.gaussian_filter(img_clahe, 3)
#
#         filter_blurred_f = ndimage.gaussian_filter(blurred_f, 1)
#
#         alpha = 30
#         img_clahe = blurred_f + alpha * (blurred_f - filter_blurred_f)
#
#         addImage(columns * i + 3, img_clahe, rows, structs[i].filter_name)
#
        """Applying CLAHE"""
        """START"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
        out = clahe.apply(img_clahe)
        org_img = structs[i].filter_name, "Histogram Equalization"  # org_img + "_Hst.Eq"
#         org_img = org_img if i == 0 else ""
#
#
# #         new_img = "/Users/adilyousuf/Documents/NCAI/DIP/Assignment 6/assets/InBreast/new/{0}.png".format(i+1)
# #         org_img = array(org_img)
# #         cv2.imwrite(new_img, out)
        addImage(columns * i + 3, out, rows, org_img)
        """END"""



noiseEstimation()
plt.show()
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


# noiseEstimation([lena, lena_gaussian_filter, lena_gaussian_noise, lena_gaussian_noise_filter])
# noiseEstimation([img_path1, img_path2, img_path3])  #
# noiseEstimation([lena, lena_gaussian_filter, lena_gaussian_noise, lena_gaussian_noise_filter], "/Users/adilyousuf/Documents/NCAI/DIP/Assignment 6/assets/Grey/lena_sp.jpg"])

# plotting
"""
https://scikit-image.org/docs/dev/auto_examples/filters/plot_rank_mean.html
"""
