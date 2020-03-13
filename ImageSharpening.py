import cv2
from scipy import stats, ndimage
import numpy as np, array
import matplotlib.pyplot as plt
from skimage.restoration import estimate_sigma
import os
from PIL import Image, ImageEnhance, ImageFilter

# Plotting Setting
w = 10
h = 10
fig = plt.figure(figsize=[40, 40])
columns = 5
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
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BAYER_BG2GRAY))#(img, cmap=plt.cm.gray)#(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.imshow(img, cmap=plt.cm.gray)


def noiseEstimation():
    rows = 1
    for i in range(rows):

        # org_img = os.path.basename(i+1)
        # print(org_img)
        # img_name = os.path.basename(i + 1)
        org_img = i + 4  # img_name.replace(".png", "")
        print(org_img)
        url = "/Users/adilyousuf/Documents/NCAI/DIP/Assignment 6/ImagesForPaper/{0}.png".format(
            i + 4)
        print(url)
        img = cv2.imread(url, 0)
        #         img = Image.open(url)
        value = signaltonoise(img)
        structs.append(img_data(org_img, value, "", 0, 0, 0))
        print(value)

        #         print("line: 144", "rows:", rows, "columns:", columns, "i:", 2 * i)
        addImage(columns * i + 1, img, rows, "Ori")

        """Apply Filters"""

        """Gaussian Filter"""
        filtering = cv2.GaussianBlur(img, (5, 5), 0)
        # org_img = org_img + "_Gaussian"
        img_path = url.replace(".png", "") + "_gaussian.png" #"/Users/adilyousuf/Documents/NCAI/DIP/Assignment 6/ImagesForPaper/" + img_name
        # print(img_path)
        cv2.imwrite(img_path, filtering)
        value = signaltonoise(filtering)
        print("GaussianBlur:", value)
        img_clahe = filtering
        structs[i].filter_name = "GaussianBlur"
        structs[i].noise_value_after = value
        addImage(columns * i + 2, filtering, rows, "G_F")

        # """Bilateral Filter"""
        # filtering = cv2.bilateralFilter(img, 9, 75, 75)
        # # img_path = "/Users/adilyousuf/Documents/NCAI/DIP/Assignment 6/assets/Grey/" + img_name.replace(".png",
        # #                                                                                                "") + "_bilateral.png"
        # # print(img_path)
        # # cv2.imwrite(img_path, filtering)
        # if value > signaltonoise(filtering):
        #     value = signaltonoise(filtering)
        #     img_clahe = filtering
        #     # org_img = org_img + "_Bilateral"
        #     structs[i].filter_name = "bilateralFilter"
        #     structs[i].noise_value_after = value
        # # value = signaltonoise(filtering) if value > signaltonoise(filtering) else value
        # print("bilateralFilter:", signaltonoise(filtering))
        # # addImage(5 * i+2, filtering, rows)

        filtering = cv2.medianBlur(img, 3)
        img_path = url.replace(".png", "") + "_median.png" #"/Users/adilyousuf/Documents/NCAI/DIP/Assignment 6/assets/Results/" + img_name
        # print(img_path)
        cv2.imwrite(img_path, filtering)
        if value > signaltonoise(filtering):
            value = signaltonoise(filtering)
            img_clahe = filtering
            # org_img = org_img + "_Medium"
            structs[i].filter_name = "medianBlur"
            structs[i].noise_value_after = value
        # value = signaltonoise(filtering)
        print("medianBlur:", signaltonoise(filtering))
        addImage(columns * i + 3, filtering, rows, "M_F")

        # # processed_image = cv2.meanBlur(img, 3)
        # # 2D Convolution ( Image Filtering )
        # kernel = np.ones((3, 3), np.float32) / 9
        # filtering = cv2.filter2D(img, -1, kernel)
        # # img_path = "/Users/adilyousuf/Documents/NCAI/DIP/Assignment 6/assets/Grey/" + img_name.replace(".png",
        # #                                                                                                "") + "_filter2D.png"
        # # print(img_path)
        # # cv2.imwrite(img_path, filtering)
        # if value > signaltonoise(filtering):
        #     value = signaltonoise(filtering)
        #     img_clahe = filtering
        #     # org_img = org_img + "_Filter2D"
        #     structs[i].filter_name = "filter2D"
        #     structs[i].noise_value_after = value
        # # value = signaltonoise(filtering)
        # print("filter2D:", signaltonoise(filtering))
        # # addImage(5 * i+4, filtering, rows)

        """#Adding Image After Filtering"""
        # addImage(columns * i + 2, img_clahe, rows, structs[i].filter_name)

        """#Adding Sharpening"""
        # blurred_f = ndimage.gaussian_filter(out, 3)
        #
        # filter_blurred_f = ndimage.gaussian_filter(blurred_f, 1)
        #
        # alpha = 13
        # img_clahe = blurred_f + alpha * (blurred_f - filter_blurred_f)

        # img1 = Image.fromarray(img_clahe)
        # img1 = img1.filter(ImageFilter.SHARPEN)
        #
        # addImage(columns * i + 3, img1, rows, "Sharp")

        # hist1 = np.histogram(img_clahe, bins=50, range=(-10, 10), density=True)
        # data = hist1[0]
        # ent = -(data * np.ma.log(np.abs(data))).sum()
        # print("Before CLAHE: ", ent)

        out = img_clahe
        #
        # max_channels = np.amax([np.amax(img[:, 0])])
        #
        # print("B CLAHE MAX: ", max_channels)
        # min_channels = np.amin([np.amin(img[:, 0])])
        #
        # print("B CLAHE MIN: ", min_channels)
        # print("Before COntrast: ", (max_channels - min_channels) / (max_channels + min_channels))

        # show = True
        # if ent >= 1:
        #     show = False

        # while ent < 1:
        """Applying CLAHE"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
        out = clahe.apply(out)
        img_path = "/Users/adilyousuf/Documents/NCAI/DIP/Assignment 6/ImagesForPaper/{0}_CLAHE.png".format(i + 4)
        print(img_path)
        cv2.imwrite(img_path, out)
        # org_img = "Histogram Equalization"  # org_img + "_Hst.Eq"
        # hist1 = np.histogram(out, bins=50, range=(-10, 10), density=True)
        # data = hist1[0]
        # ent = -(data * np.ma.log(np.abs(data))).sum()
        # print("After CLAHE: ", ent)

        # max_channels = np.amax([np.amax(img[:, 0])])

        # print("After CLAHE MAX: ", max_channels)
        # min_channels = np.amin([np.amin(img[:, 0])])
        #
        # print("After CLAHE MIN: ", min_channels)
        # print("After COntrast: ", (max_channels - min_channels) / (max_channels + min_channels))

        # org_img = org_img if i == 0 else ""
        # org_img = array(org_img)
        #         cv2.imwrite(new_img, out)
        # if show:
        addImage(columns * i + 4, out, rows, "CLAHE")

        """#Adding Sharpening"""
        # blurred_f = ndimage.gaussian_filter(out, 3)
        # filter_blurred_f = ndimage.gaussian_filter(blurred_f, 1)
        # alpha = 20
        # img_clahe = blurred_f + alpha * (blurred_f - filter_blurred_f)
        # sharpen = Image.fromarray(img_clahe)
        #
        # addImage(columns * i + 4, sharpen, rows, "Sharp-1")

        out = Image.fromarray(out)
        out = out.filter(ImageFilter.SHARPEN)
        # addImage(columns * i + 5, out, rows, "Sharp-2")

        new_img = "/Users/adilyousuf/Documents/NCAI/DIP/Assignment 6/ImagesForPaper/{0}_Sharpening.png".format(i + 4)
        sharpen2 = np.array(out)
        addImage(columns * i + 5, sharpen2, rows, "Sharpening")
        cv2.imwrite(new_img, sharpen2)

        """Morphological Transformations"""

        # Taking a matrix of size 5 as the kernel
        kernel = np.ones((15, 15), np.uint8)
        # The first parameter is the original image,
        # kernel is the matrix with which image is
        # convolved and third parameter is the number
        # of iterations, which will determine how much
        # you want to erode/dilate a given image.
        # img_erosion = cv2.erode(sharpen2, kernel, iterations=1)
        # new_img = "/Users/adilyousuf/Documents/NCAI/DIP/Assignment 6/assets/Total_Processed_Pectoral_data/Total_Processed_Pectoral_data_Erosion/{0}.png".format(
        #     i + 1)
        # cv2.imwrite(new_img, img_erosion)
        # addImage(columns * i + 3, img_erosion, rows, "img_erosion")
        # img_dilation = cv2.dilate(sharpen2, kernel, iterations=1)
        # new_img = "/Users/adilyousuf/Documents/NCAI/DIP/Assignment 6/assets/Total_Processed_Pectoral_data/Total_Processed_Pectoral_data_Dilation/{0}.png".format(
        #     i + 1)
        # cv2.imwrite(new_img, img_dilation)
        # addImage(columns * i + 4, img_dilation, rows, "img_dilation")
        # opening = cv2.morphologyEx(sharpen2, cv2.MORPH_OPEN, kernel)
        # addImage(columns * i + 5, opening, rows, "img_opening")
        # closing = cv2.morphologyEx(sharpen2, cv2.MORPH_CLOSE, kernel)
        # addImage(columns * i + 6, closing, rows, "img_closing")
        # gradient = cv2.morphologyEx(sharpen2, cv2.MORPH_GRADIENT, kernel)
        # addImage(columns * i + 7, gradient, rows, "gradient")
        # tophat = cv2.morphologyEx(sharpen2, cv2.MORPH_TOPHAT, kernel)
        # addImage(columns * i + 8, tophat, rows, "tophat")
        # blackhat = cv2.morphologyEx(sharpen2, cv2.MORPH_BLACKHAT, kernel)
        # addImage(columns * i + 9, blackhat, rows, "blackhat")


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
