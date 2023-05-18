import numpy as np
import imutils
import cv2
import copy

lower = {'matang':(0, 50, 20), 'mentah':(30, 100, 100), 'setengah matang':(20, 100, 100)} 
upper = {'matang':(10,255,255), 'mentah':(80,255,255), 'setengah matang':(60,255,255)}
colors = {'matang':(0,0,255), 'mentah':(0,255,0), 'setengah matang':(0,140,255)}
largest_item = {}

# Started
cam = cv2.VideoCapture(0)

while True :
    _, frame = cam.read()
    frame = imutils.resize(frame, width=800)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # lingkaran sesuai warna
    for key_color, value in upper.items():
        kernel = np.ones((9,9),np.uint8)
        mask = cv2.inRange(hsv, lower[key_color], upper[key_color])
        # melakukan pre processing opening
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # melakukan pre processing closing
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        if key_color == "matang" :
            cv2.imshow("matang", mask)
        elif key_color == "mentah":
            cv2.imshow("mentah", mask)
        elif key_color == "setengah matang":
            cv2.imshow("setengah matang", mask)

        # mencari conture
        cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if(len(cnts) > 0):
            largest_item[key_color] = max(cnts, key=cv2.contourArea)

    try :
        # mencari conture terbesar dengan perbandingan setiap warna
        largest_contour = max(largest_item, key=lambda x: cv2.contourArea(largest_item[x]))
        cnt_largest = largest_item[largest_contour]
        ((X, Y), radius) = cv2.minEnclosingCircle(cnt_largest)

        # Menentukan titik tengah kontur
        M = cv2.moments(cnt_largest)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # text berdasarkan key yang di dapat
        if largest_contour == "matang" :
            text = "matang"
        elif largest_contour == "mentah":
            text = "mentah"
        elif largest_contour == "setengah matang":
            text = "setengah matang"

        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        text_x = cX - text_width // 2
        text_y = cY + text_height // 2

        if radius > 0.5:
            cv2.circle(frame, (int(X), int(Y)), int(radius), colors[largest_contour], 2)
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 0), 2)
    except :
        print("Object tidak di temukan")

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()           
                          
                 
           
