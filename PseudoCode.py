import cv2
from scipy import stats, ndimage
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import estimate_sigma
import os
from PIL import Image, ImageEnhance, ImageFilter

# Plotting Setting
w = 10
h = 10
fig = plt.figure(figsize=[40, 40])
columns = 1
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
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    #     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BAYER_BG2GRAY))#(img, cmap=plt.cm.gray)#(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.imshow(img, cmap=plt.cm.gray)

def noiseEstimation():
    for i in range(1):
        rows = 1  # len(imgArr)

        # org_img = os.path.basename(i+1)
        # print(org_img)
        # img_name = "25.png"#os.path.basename(i+1)
        # org_img = i + 1  # img_name.replace(".png", "")
        # print(org_img)
        url = "/Users/adilyousuf/Documents/NCAI/DIP/Assignment 6/assets/InBreast/{0}.png".format(i + 1)
        print(url)
        img = cv2.imread(url, 0)
        #img = Image.open(url)
        value = signaltonoise(img)
        structs.append(img_data(org_img, value, "", 0, 0, 0))
        print(value)

        #print("line: 144", "rows:", rows, "columns:", columns, "i:", 2 * i)
        # addImage(columns * i + 1, img, rows, "")

        """Apply Filters"""

        """Gaussian Filter"""
        filtering = cv2.GaussianBlur(img, (5, 5), 0)
        # org_img = org_img + "_Gaussian"
        img_path = url.replace(".png", "") + "_g.png" #"/Users/adilyousuf/Documents/NCAI/DIP/Assignment 6/ImagesForPaper/" + img_name
        # print(img_path)
        cv2.imwrite(img_path, filtering)
        value = signaltonoise(filtering)
        print("GaussianBlur SNR:", value)
        img_clahe = filtering
        structs[i].filter_name = "GaussianBlur"
        structs[i].noise_value_after = value
        #addImage(columns * i + 2, filtering, rows, "b")

        """Bilateral Filter"""
        filtering = cv2.bilateralFilter(img, 9, 75, 75)
        img_path = url.replace(".png", "") + "_b.png" #"/Users/adilyousuf/Documents/NCAI/DIP/Assignment 6/assets/Results/" + img_name
        # print(img_path)
        cv2.imwrite(img_path, filtering)
        if value > signaltonoise(filtering):
            value = signaltonoise(filtering)
            img_clahe = filtering
            # org_img = org_img + "_Bilateral"
            structs[i].filter_name = "bilateralFilter"
            structs[i].noise_value_after = value
        # value = signaltonoise(filtering) if value > signaltonoise(filtering) else value
        print("bilateralFilter:", signaltonoise(filtering))
        #addImage(columns * i + 3, filtering, rows, "c")

        """Median Filter"""
        filtering = cv2.medianBlur(img, 3)
        img_path = "/Users/adilyousuf/Documents/NCAI/DIP/Assignment 6/assets/Results/" + img_name.replace(".png",
                                                                                                       "") + "_m.png"
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
        #addImage(columns * i + 4, filtering, rows, "d")

        # processed_image = cv2.meanBlur(img, 3)
        # 2D Convolution ( Image Filtering )
        kernel = np.ones((3, 3), np.float32) / 9
        filtering = cv2.filter2D(img, -1, kernel)
        img_path = "/Users/adilyousuf/Documents/NCAI/DIP/Assignment 6/assets/Results/" + img_name.replace(".png",
                                                                                                       "") + "_f2D.png"
        # print(img_path)
        cv2.imwrite(img_path, filtering)
        if value > signaltonoise(filtering):
            value = signaltonoise(filtering)
            img_clahe = filtering
            # org_img = org_img + "_Filter2D"
            structs[i].filter_name = "filter2D"
            structs[i].noise_value_after = value
        # value = signaltonoise(filtering)
        print("filter2D:", signaltonoise(filtering))
        #addImage(columns * i + 5, filtering, rows, "e")

        """#Adding Image After Filtering"""
        # addImage(columns * i + 2, img_clahe, rows, structs[i].filter_name)

        """Applying CLAHE"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
        out = clahe.apply(img_clahe)
        org_img = "Histogram Equalization"  # org_img + "_Hst.Eq"
        org_img = org_img if i == 0 else ""
        new_img = "/Users/adilyousuf/Documents/NCAI/DIP/Assignment 6/assets/Results/25_m_CLAHE.png"
        # org_img = array(org_img)
        cv2.imwrite(new_img, out)
        addImage(columns * i + 1, out, rows, "")

        """#Adding Sharpening"""
        out = Image.fromarray(out)
        out = out.filter(ImageFilter.SHARPEN)
        # addImage(columns * i + 1, out, rows, "")

        new_img = "/Users/adilyousuf/Documents/NCAI/DIP/Assignment 6/assets/Results/25_m_sharpen.png"
        sharpen2 = np.array(out)
        # addImage(columns * i + 1, sharpen2, rows, "")
        cv2.imwrite(new_img, sharpen2)


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

"""

class img_data:
    def __init__(self, img_name, noise, filter_name, noise_value_after, contrast_value, con_after_enh):
        self.img_name = img_name
        self.noise = noise
        self.filter_name = filter_name
        self.noise_value_after = noise_value_after
        self.contrast_value = contrast_value
        self.con_after_enh = con_after_enh


structs = [img_data("lena", 5.943, "gaussian", 9.342, 0, 1.2342)]
structs.append(img_data("lena", 3.943, "mean", 7.342, 6.342, 3.2342))

structs[0].filter_name = "Median"
structs[0].contrast_value = 4.543

for i in structs:
    print("Image Name: ", i.img_name, ", Noise: ", i.noise, ", Filter Name: ", i.filter_name, ", Noise Value After: ", i.noise_value_after, ", Contrast Value: ", i.contrast_value, ", Contrast After Enhancement: ", i.con_after_enh)


# float getContrast() {
#     float maxLum = 0
#     float minLum = 10000
#     for (int i=0; i<img.pixels.length; i++) {
#       float r = img.pixels[i] >> 16 & 0xFF
#       float g = img.pixels[i] >> 8 & 0xFF
#       float b = img.pixels[i] & 0xFF
#       float brightness = (0.2126 * r) + (0.7152 * g) + (0.0722 * b)
#       if (brightness > maxLum) maxLum = brightness
#       if (brightness < minLum) minLum = brightness
#     }
#     return (maxLum - minLum)/(maxLum + minLum)
#   }
"""