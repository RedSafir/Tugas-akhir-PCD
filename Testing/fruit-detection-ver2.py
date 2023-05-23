import cv2
import numpy as np

# Finds face and return the original image and cropped face
def detect_face(image):
	fruit_cascade = cv2.CascadeClassifier('cascade.xml')
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = fruit_cascade.detectMultiScale(gray, 1.3, 5)

	if faces == ():
		return image
	for (x, y, w, h) in faces:
		cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)
		cropped_face = image[y:y+h, x:x+w]
		cropped_face = cv2.resize(cropped_face, (200,200))
		return cropped_face		

capture = cv2.imread("buah-tomat.jpg")
captue_rez = cv2.resize(capture, None, fx=0.5, fy=0.5,
				interpolation=cv2.INTER_LINEAR)
found_face = detect_face(captue_rez)	
cv2.imshow("fruit detection", found_face)
	    
capture.release()
cv2.destroyAllWindows()	