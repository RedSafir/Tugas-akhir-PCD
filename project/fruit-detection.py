import numpy as np
import imutils
import cv2

lower = {'red':(166, 84, 141), 'green':(66, 122, 129), 'yellow':(23, 59, 119), 'orange':(0, 50, 80)} 
upper = {'red':(186,255,255), 'green':(86,255,255), 'yellow':(54,255,255), 'orange':(20,255,255)}
 
colors = {'red':(0,0,255), 'green':(0,255,0),'yellow':(0, 255, 217), 'orange':(0,140,255)}


# Preprocessing
image = cv2.imread("dumy-img-min.jpg")
image = cv2.resize(image, None, fx=0.15, fy=0.15, interpolation=cv2.INTER_CUBIC)
frame = imutils.resize(image, width=800)
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

    # mencari conture
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((X, Y), radius) = cv2.minEnclosingCircle(c)
        
        height, width, _ = image.shape

        (x, y, w, h) = cv2.boundingRect(cnts[0])

        font_scale = min(w, h) 

        text = 'Deteksi Objek'

        if radius > 0.5:
            cv2.circle(image, (int(X), int(Y)), int(radius), colors[key], 2)
            cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, colors[key], 2)

cv2.imshow("image", image)
cv2.waitKey(0)
# cv2.destroyAllWindows()           
                          
                 
           
