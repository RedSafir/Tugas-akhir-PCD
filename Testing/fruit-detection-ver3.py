import numpy as np
import imutils
import cv2
import copy

lower = {'red':(166, 84, 141), 'green':(36, 25, 25), 'yellow':(23, 59, 119), 'orange':(0, 50, 80)} 
upper = {'red':(186,255,255), 'green':(86,255,255), 'yellow':(54,255,255), 'orange':(20,255,255)}
colors = {'red':(0,0,255), 'green':(0,255,0),'yellow':(0, 255, 217), 'orange':(0,140,255)}

lower1 = np.array([0, 100, 20])
upper1 = np.array([10, 255, 255])

lower2 = np.array([160,100,20])
upper2 = np.array([179,255,255])

# Started
image = cv2.imread("../imgs/buah-tomat.jpg")
image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
img_height, img_width, _ = image.shape
blurred = cv2.GaussianBlur(image, (11, 11), 0)
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

# lingkaran sesuai warna
for key, value in upper.items():
    kernel = np.ones((9,9),np.uint8)
    mask = cv2.inRange(hsv, lower[key], upper[key])
    # melakukan pre processing opening
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # melakukan pre processing closing
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    if key == "red" :
        cv2.imshow("red", mask)
    elif key == "yellow":
        cv2.imshow("yellow", mask)
    elif key == "green":
        cv2.imshow("green", mask)
    elif key == "orange" :
        cv2.imshow("orange", mask)

    # mencari conture
    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, cnt in enumerate(cnts):
        # Mengabaikan kontur bingkai
        if cv2.contourArea(cnt) > img_width * img_height * 0.9:
            continue

        epsilon = 0.01 * cv2.arcLength(cnt, True) # menghitung panjang kurva
        approx = cv2.approxPolyDP(cnt, epsilon, True) # memperhalus kurva

        # Menentukan titik tengah kontur
        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        if len(approx) > 10:

            c = max(cnts, key=cv2.contourArea)
            ((X, Y), radius) = cv2.minEnclosingCircle(c)
            
            shape = "lingkaran"

            (text_width, text_height), _ = cv2.getTextSize(shape, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            text_x = cX - text_width // 2
            text_y = cY + text_height // 2

            text = 'Deteksi Objek'

            if radius > 0.5:
                cv2.circle(image, (int(X), int(Y)), int(radius), colors[key], 2)
                cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 0), 2)

cv2.imshow("image", image)
cv2.waitKey(0)
# cv2.destroyAllWindows()           
                          
                 
           
