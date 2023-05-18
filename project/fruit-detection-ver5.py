import numpy as np
import imutils
import cv2
import copy

lower = {'matang':(0, 50, 20), 'mentah':(30, 100, 100), 'setengah matang':(10, 100, 100)} 
upper = {'matang':(10,255,255), 'mentah':(80,255,255), 'setengah matang':(60,255,255)}
colors = {'matang':(0,0,255), 'mentah':(0,255,0), 'setengah matang':(0,140,255)}
target_width = 400
target_height = 400
largest_item = {}

# Started
image = cv2.imread("../imgs/tmt-setengah-1.jpg", cv2.IMREAD_COLOR)

# melakukan normalisasi
image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

image = cv2.resize(image, (target_width, target_height))
img_height, img_width, _ = image.shape
blurred = cv2.GaussianBlur(image, (11, 11), 0)
# Buat kernel sharpening
kernel = np.array([[-1, -1, -1],
                   [-1,  9, -1],
                   [-1, -1, -1]])

# Lakukan konvolusi dengan kernel
sharpened_image = cv2.filter2D(blurred, -1, kernel)

hsv = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2HSV)

# lingkaran sesuai warna
for key_color, value in upper.items():  
    # Inisialisasi Strel dengan menggunakan cv2.MORPH_CROSS, (9, 9)
    kernel = np.ones((9,9),np.uint8)
    # melakukan masking terhadap warna
    mask = cv2.inRange(hsv, lower[key_color], upper[key_color])
    # melakukan pre processing opening
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # melakukan pre processing closing
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # memunculkan mask di setiap warna
    if key_color == "matang" :
        cv2.imshow("matang", mask)
    elif key_color == "mentah":
        cv2.imshow("mentah", mask)
    elif key_color == "setengah matang":
        cv2.imshow("setengah matang", mask)

    # mencari conture di setiap warna
    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # bila conture pada warna tersebut di temukan
    if(len(cnts) > 0):
        # maka masukan ke dalam array associative
        largest_item[key_color] = max(cnts, key=cv2.contourArea)

try :
    # mencari conture terbesar pada array di setiap warna, dapatkan (key nya
    key = max(largest_item, key=lambda x: cv2.contourArea(largest_item[x]))
    # simpan 
    cnt_largest = largest_item[key]

    epsilon = 0.01 * cv2.arcLength(cnt_largest, True)
    approx = cv2.approxPolyDP(cnt_largest, epsilon, True)

    ((X, Y), radius) = cv2.minEnclosingCircle(approx)
    num_sides = len(approx)
    cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)

    if num_sides > 10:

        # text berdasarkan key yang di dapat
        if key == "matang" :
            text = "matang"
        elif key == "mentah":
            text = "mentah"
        elif key == "setengah matang":
            text = "setengah matang"

        # Menentukan titik tengah kontur untuk menyimpan text
        M = cv2.moments(cnt_largest)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        text_x = cX - text_width // 2
        text_y = cY + text_height // 2

        # gambar conture berserta dengan text
        cv2.circle(image, (int(X), int(Y)), int(radius), colors[key], 2)
        cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 255), 2)
    else:
        print("object bukanlah tomat")
except :
    print("tidak ada obejct")

cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()           
                          
                 
           
