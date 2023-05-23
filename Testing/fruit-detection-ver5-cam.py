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
cam = cv2.VideoCapture(0)

while True :
    _, frame = cam.read()
    frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
    frame = imutils.resize(frame, width=target_width, height=target_height)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    # Buat kernel sharpening
    kernel = np.array([ [-1, -1, -1],
                        [-1,  9, -1],
                        [-1, -1, -1]])
    # Lakukan konvolusi dengan kernel sharpening
    sharpened_image = cv2.filter2D(blurred, -1, kernel)
    # ubah menjad HSV
    hsv = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2HSV)

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
        key = max(largest_item, key=lambda x: cv2.contourArea(largest_item[x]))
        cnt_largest = largest_item[key]

        ((X, Y), radius) = cv2.minEnclosingCircle(cnt_largest)

        epsilon = 0.01 * cv2.arcLength(cnt_largest, True)
        approx = cv2.approxPolyDP(cnt_largest, epsilon, True)

        num_sides = len(approx)

        if num_sides > 10:

            # text berdasarkan key yang di dapat
            if key == "matang" :
                text = "matang"
            elif key == "mentah":
                text = "mentah"
            elif key == "setengah matang":
                text = "setengah matang"
            
            # Menentukan titik tengah kontur
            M = cv2.moments(cnt_largest)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            text_x = cX - text_width // 2
            text_y = cY + text_height // 2

            cv2.circle(frame, (int(X), int(Y)), int(radius), colors[key], 2)
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 255), 2)
   
    except :
        print("Object tidak di temukan")

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()           
                          
                 
           
