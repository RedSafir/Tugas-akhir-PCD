import numpy as np
import imutils
import cv2
import copy
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

lower = {'red':(166, 84, 141), 'green':(66, 122, 129), 'blue':(97, 100, 117), 'yellow':(23, 59, 119), 'orange':(0, 50, 80)} 
upper = {'red':(186,255,255), 'green':(86,255,255), 'blue':(117,255,255), 'yellow':(54,255,255), 'orange':(20,255,255)}
colors = {'red':(0,0,255), 'green':(0,255,0), 'blue':(255,0,0), 'yellow':(0, 255, 217), 'orange':(0,140,255)}

def equalzationHistogram(image):
    hist, bins = np.histogram(image.flatten(), 256, [0, 256]) #mengubah array img menjadi 1 dimensi
    cdf = hist.cumsum() # menentukan jumlah kumulatif array pada bagian tertentu
    cdf_normalized = cdf * hist.max() / cdf.max() # untuk normalisasi
    cdf_m = np.ma.masked_equal(cdf, 0) # memasking nilai array dengan yang di berikan
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min()) # melakukan perhitungan
    cdf = np.ma.filled(cdf_m, 0).astype('uint8') # mengisi array dengan nilai skalar
    image = cdf[image] # mengganti nilai array image menjadi nilai komulatif
    
    return image

def CannyClicked(img1, param1, param2):

    # Step 1: Noise Reduction dengan gaussian karnel
    gauss = (1.0 / 57) * np.array([[0, 1, 2, 1, 0],
                                    [1, 3, 5, 3, 1],
                                    [2, 5, 9, 5, 2],
                                    [1, 3, 5, 3, 1],
                                    [0, 1, 2, 1, 0]])
    img_out = cv2.filter2D(img1, -1, gauss)
    fig = plt.figure(figsize=(12, 12))

    # Step 2: Finding Gradient
    sobel_x = cv2.Sobel(img_out, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_out, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    theta = np.arctan2(sobel_y, sobel_x)

    # Step 3: Non-Maximum Suppression
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180
    Z = np.zeros(img1.shape, dtype=np.int32)
    H, W = img1.shape
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            try:
                q = 255
                r = 255
                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = mag[i, j + 1]
                    r = mag[i, j - 1]
                # angle 45
                elif (22.5 <= angle[i, j] < 67.5):
                    q = mag[i + 1, j - 1]
                    r = mag[i - 1, j + 1]
                # angle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    q = mag[i + 1, j]
                    r = mag[i - 1, j]
                # angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    q = mag[i - 1, j - 1]
                    r = mag[i + 1, j + 1]
                if (mag[i, j] >= q) and (mag[i, j] >= r):
                    Z[i, j] = mag[i, j]
                else:
                    Z[i, j] = 0
            except IndexError as e:
                pass
    img_N = Z.astype("uint8")

    # Step 4: Hysteresis Thresholding
    weak = param1
    strong = param2
    for i in np.arange(H):
        for j in np.arange(W):
            a = img_N.item(i, j)
            if (weak < a < strong):  # weak
                b = weak
            elif (a > strong):  # strong
                b = 255
            else:
                b = 0
            img_N.itemset((i, j), b)
    img_H1 = img_N.astype("uint8")

    # hysteresis Thresholding eliminasi titik tepi lemah jika tidak terhubung dengan tetangga tepi kuat
    strong = 255
    for i in range(1, H - 1):
        for j in range(1, W - 1):
            if (img_H1[i, j] == weak):
                try:
                    if ((img_H1[i + 1, j - 1] == strong) or
                            (img_H1[i + 1, j] == strong) or
                            (img_H1[i + 1, j + 1] == strong) or
                            (img_H1[i, j - 1] == strong) or
                            (img_H1[i, j + 1] == strong) or
                            (img_H1[i - 1, j - 1] == strong) or
                            (img_H1[i - 1, j] == strong) or
                            (img_H1[i - 1, j +   1] == strong)):
                        img_H1[i, j] = strong
                    else:
                        img_H1[i, j] = 0
                except IndexError as e:
                    print("error")
                    pass

    img_H2 = img_H1.astype("uint8")

    return img_H2

def closing(image):
    kernel = np.ones((9,9),np.uint8)

    mask = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    return mask
    
# Proses
image = cv2.imread("dumy-img-min.jpg")
# hsv = cv2.cvtColor(copy.copy(image), cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(copy.copy(image), cv2.COLOR_BGR2GRAY)
equal = equalzationHistogram(image)
canny = cv2.Canny(equal, 100, 200)
close = closing(canny)

# Tampilan
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axs[0].set_title('Gambar 1')

axs[1].imshow(close, cmap='gray')
axs[1].set_title('Gambar 2')

plt.tight_layout()
plt.show()

cv2.