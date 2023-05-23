import numpy as np
import imutils
import cv2
import copy

lower = {'matang':(0, 50, 20), 'mentah':(30, 100, 100), 'setengah matang':(20, 100, 100)} 
upper = {'matang':(10,255,255), 'mentah':(80,255,255), 'setengah matang':(60,255,255)}
colors = {'matang':(0,0,255), 'mentah':(0,255,0), 'setengah matang':(0,140,255)}
largest_item = {}

# Started
image = cv2.imread("../imgs/tmt-setengah-matang.jpg")
image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
img_height, img_width, _ = image.shape
blurred = cv2.GaussianBlur(image, (11, 11), 0)
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

# lingkaran sesuai warna
for key_color, value in upper.items():  
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
    largest_contour = max(largest_item, key=lambda x: cv2.contourArea(largest_item[x]))
    # simpan 
    cnt_largest = largest_item[largest_contour]
    ((X, Y), radius) = cv2.minEnclosingCircle(cnt_largest)

    # text berdasarkan key yang di dapat
    if largest_contour == "matang" :
        text = "matang"
    elif largest_contour == "mentah":
        text = "mentah"
    elif largest_contour == "setengah matang":
        text = "setengah matang"
    
    # Menentukan titik tengah kontur untuk menyimpan text
    M = cv2.moments(cnt_largest)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    text_x = cX - text_width // 2
    text_y = cY + text_height // 2

    # gambar conture berserta dengan text
    if radius > 0.5:
        cv2.circle(image, (int(X), int(Y)), int(radius), colors[largest_contour], 2)
        cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 255), 2)
except :
    print("Object tidak di temukan")

cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()           
                          
                 
           
