import sys
import cv2
from PyQt5 import QtCore,QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
import numpy as np
import math 
import tkinter as tk
from tkinter import filedialog
import copy
import imutils

class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi("GUI.ui", self)
        self.Image = None

        # Action Button
        self.actionFoto.triggered.connect(self.mendeteksiTomat)
        self.actionCamera.triggered.connect(self.mendeteksiTomatCam)
        self.actionSave.triggered.connect(self.save)

    """
    INISIASI FUNCTION

    """
    def normalisasiCitra(self, image):
        # Konversi citra ke float32
        image_float = image.astype(np.float32)

        # Normalisasi citra
        normalized_image = cv2.normalize(image_float, None, 0, 255, cv2.NORM_MINMAX)

        # Konversi citra kembali ke tipe data uint8
        normalized_image = normalized_image.astype(np.uint8)

        return normalized_image
    
    def math_konvolusi(self, arrycitra, arrykarnel):
        # baca ukuran dimensi citra
        H_citra, W_citra = arrycitra.shape

        # baca ukuran dimensi karnel
        H_karnel, W_karnel = arrykarnel.shape

        # meenutukan titik tengah
        H = H_karnel // 2
        W = W_karnel // 2   

        out = np.zeros((H_citra, W_citra))

        # menggeser karnel konvolusi
        for i in range(H + 1, H_citra - H):
            for j in range(W + 1, W_citra - W):
                sum = 0
                for k in range(-H, H):
                    for l in range(-W, W):
                        citra_value = arrycitra[i + k, j + l]
                        kernel_value = arrykarnel[H + k, W + l]
                        sum += citra_value * kernel_value
                    out[i, j] = copy.copy(sum)
        
        return out

    def gaussianfilter(self, image):   
        # buat kernel
        KERNEL = (1.0 / 345) * np.array([   [1, 5, 7, 5, 1],
                                            [5, 20, 33, 20, 5],
                                            [7, 33, 55, 33, 7],
                                            [5, 20, 33, 20, 5],
                                            [1, 5, 7, 5, 1]])
        
        # lakukan konvolusi dengan karnel dan image yang sudah di buat grey
        hasil = cv2.filter2D(image, -1,  KERNEL)

        return hasil

    def sharpening(self, image):
        KERNEL = np.array([ [-1, -1, -1],
                            [-1,  9, -1],
                            [-1, -1, -1]])
        hasil = cv2.filter2D(image, -1,  KERNEL)

        return hasil

    def dilasi(self, image, strel):
        try:
            # Mengkonversi citra menjadi grayscale
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Mengkonversi citra grayscale menjadi citra biner menggunakan cv2.threshold
            _, image = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
        except :
            pass

        # Dilasi
        img_dilated = cv2.dilate(image,strel)

        return img_dilated

    def erosi(self, image, strel):
        try :
            # Mengkonversi citra menjadi grayscale
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Mengkonversi citra grayscale menjadi citra biner menggunakan cv2.threshold
            _, image = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
        except:
            pass

        # Erosi
        img_erose = cv2.erode(image,strel)

        return img_erose

    def opening(self, image, strel):
        try:
            # Mengkonversi citra menjadi grayscale
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Mengkonversi citra grayscale menjadi citra biner menggunakan cv2.threshold
            _, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
        except:
            pass
        
        # MORPH OPEN untuk erosi -> dilasi
        img_erose = self.erosi(image, strel)
        img_open = self.dilasi(img_erose, strel)

        return img_open

    def closing(self, image, strel):
        try : 
            # Mengkonversi citra menjadi grayscale
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Mengkonversi citra grayscale menjadi citra biner menggunakan cv2.threshold
            _, image = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
        except: 
            pass

        # MORPH CLOSE untuk dilasi -> erosi
        img_dilas = self.dilasi(image, strel)
        img_open = self.erosi(img_dilas, strel)

        return img_open
    
    """
    FILE COMPOSER
    """
    def save(self):
        flname, filter = QFileDialog.getSaveFileName(self, "SaveFile", "C:\\", "Image Files (*.jpg)")
        if flname:
            cv2.imwrite(flname, self.Image)
        else :
            print("error")

    def open(self):
        filename = filedialog.askopenfilename()
        img = cv2.imread(filename, cv2.IMREAD_COLOR)
        self.Image = img
        return img
    
    """
    PROJECT START
    """
    def mendeteksiTomat(self):
        """
        INISIASI VARIABLE

        """
        lower = {'matang':(0, 70, 30), 'mentah':(30, 50, 50), 'setengah matang':(10, 50 , 50)} 
        upper = {'matang':(10,255,255), 'mentah':(50,255,255), 'setengah matang':(30,255,255)}
        colors = {'matang':(0,0,255), 'mentah':(0,255,0), 'setengah matang':(0,140,255)}
        target_width = 400
        target_height = 400
        largest_item = {}
        """
        TAHAP PRE-PROCESSING IMAGE

        Tahap preprocessing dalam pengolahan citra adalah serangkaian langkah atau operasi yang dilakukan 
        pada citra sebelum masuk ke tahap analisis atau pemrosesan lebih lanjut. Tujuannya adalah untuk 
        meningkatkan kualitas citra, menghilangkan noise atau gangguan, mengurangi variabilitas, 
        dan mempersiapkan citra untuk tahap selanjutnya dalam alur pengolahan citra.

        """

        # Started
        image = self.open()

        self.displayImage(image, 1)

        # lakukan histogram equalization
        image = self.normalisasiCitra(image)
        # lakukan resize
        image = cv2.resize(image, (target_width, target_height))
        # melakukan penghalusan citra
        blurred_img = self.gaussianfilter(image)
        # menlakukan penajaman citra
        shrap_img = self.sharpening(blurred_img) 

        self.displayImage(shrap_img, 2)

        """
        PROCESSING IMAGE

        """ 
        # mengubah array citra yang semula BGR menjadi HSV
        image_hsv = cv2.cvtColor(shrap_img, cv2.COLOR_BGR2HSV)
        # loop setiap warna
        for key_color, color in upper.items():
            # inisiasi STREL
            STREL = np.ones((9,9),np.uint8)
            # melakukan masking terhadap warna
            mask = cv2.inRange(image_hsv, lower[key_color], upper[key_color])
            # melakukan morfologi OPENING
            mask = self.opening(mask, STREL)
            # melakukan morfologi CLOSING
            mask = self.closing(mask, STREL)

            """
            Setelah setiap warna di masking, munculkan hasilnya
            """
            if key_color == "matang" :
                self.displayImage(mask, 3)
            elif key_color == "setengah matang":
                self.displayImage(mask, 4)
            elif key_color == "mentah":
                self.displayImage(mask, 5)

            """
            Cari Conture untuk setiap warna yang di temukan
            """
            # mencari conture di setiap warna
            cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # bila conture pada warna tersebut di temukan
            if(len(cnts) > 0):
                # maka masukan ke dalam array associative dan cari conture terbesarnya
                largest_item[key_color] = max(cnts, key=cv2.contourArea)

        try :
            # mencari conture terbesar pada array di setiap warna, dapatkan (key nya
            key = max(largest_item, key=lambda x: cv2.contourArea(largest_item[x]))
            # simpan 
            cnt_largest = largest_item[key]
            
            # perhalus bentuk dari conture 
            epsilon = 0.01 * cv2.arcLength(cnt_largest, True)
            approx = cv2.approxPolyDP(cnt_largest, epsilon, True)
            
            # cari titik tengahnya dari conture
            ((X, Y), radius) = cv2.minEnclosingCircle(approx)
            
            # gambarnya conture yang di tangkap
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
            
            # hitung sisinya, apabila kurang dari 10, maka bentuknya bukan bulat
            num_sides = len(approx)
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
                self.Image = image
            else:
                print("object tidak terdefinisi")
        except :
            print("tidak ada obejct")

        self.displayImage(image, 6)
    
    """
    MENDETEKSI MENGGUNAKAN CAMERA
    """
    def mendeteksiTomatCam(self):
        """
        INISIASI VARIABLE

        """
        lower = {'matang':(0, 70, 30), 'mentah':(30, 50, 50), 'setengah matang':(10, 50 , 50)} 
        upper = {'matang':(10,255,255), 'mentah':(50,255,255), 'setengah matang':(30,255,255)}
        colors = {'matang':(0,0,255), 'mentah':(0,255,0), 'setengah matang':(0,140,255)}
        target_width = 400
        target_height = 400
        largest_item = {}

        # Stared
        cam = cv2.VideoCapture(0)

        while True :
            _, image = cam.read()
            # self.displayImage(image, 1)
            cv2.imshow("ORI", image)

            # lakukan histogram equalization
            image = self.normalisasiCitra(image)
            # lakukan resize
            image = cv2.resize(image, (target_width, target_height))
            # melakukan penghalusan citra
            blurred_img = self.gaussianfilter(image)
            # menlakukan penajaman citra
            shrap_img = self.sharpening(blurred_img) 

            self.displayImage(shrap_img, 2)
            """
            PROCESSING IMAGE

            """ 
            # ubah menjad HSV
            image_hsv = cv2.cvtColor(shrap_img, cv2.COLOR_BGR2HSV)
            # lingkaran sesuai warna
            for key_color, value in upper.items():
                # inisiasi STREL
                STREL = np.ones((9,9),np.uint8)
                # melakukan masking terhadap warna
                mask = cv2.inRange(image_hsv, lower[key_color], upper[key_color])
                # melakukan morfologi OPENING
                mask = self.opening(mask, STREL)
                # melakukan morfologi CLOSING
                mask = self.closing(mask, STREL)

                """
                Setelah setiap warna di masking, munculkan hasilnya
                """
                if key_color == "matang" :
                    self.displayImage(mask, 3)
                elif key_color == "setengah matang":
                    self.displayImage(mask, 4)
                elif key_color == "mentah":
                    self.displayImage(mask, 5)

                """
                Cari Conture untuk setiap warna yang di temukan
                """
                # mencari conture di setiap warna
                cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # bila conture pada warna tersebut di temukan
                if(len(cnts) > 0):
                    # maka masukan ke dalam array associative dan cari conture terbesarnya
                    largest_item[key_color] = max(cnts, key=cv2.contourArea)

            try :
                # mencari conture terbesar pada array di setiap warna, dapatkan (key nya
                key = max(largest_item, key=lambda x: cv2.contourArea(largest_item[x]))
                # simpan 
                cnt_largest = largest_item[key]

                # perhalus bentuk dari conture 
                epsilon = 0.01 * cv2.arcLength(cnt_largest, True)
                approx = cv2.approxPolyDP(cnt_largest, epsilon, True)
                
                # cari titik tengahnya dari conture
                ((X, Y), radius) = cv2.minEnclosingCircle(approx)
                
                # gambarnya conture yang di tangkap
                cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
                
                # hitung sisinya, apabila kurang dari 10, maka bentuknya bukan bulat
                num_sides = len(approx)
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
                    self.Image = image
                else:
                    pass
            except :
                pass

            self.displayImage(image, 6)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    # mengatur gambar di windows
    def displayImage(self, Image, windows=1):
        qformat = QImage.Format_Indexed8

        if len(Image.shape)==3:
            if(Image.shape[2])==4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        
        img = QImage(Image, Image.shape[1], Image.shape[0], Image.strides[0], qformat)

        # karna data warnanya adalah bgr bukan rgb
        img = img.rgbSwapped()

        # secara default, dia akan menampilkan gambar pada label 1
        if windows==1:
            self.label.setPixmap(QPixmap.fromImage(img))
            # memposisikan gambar
            self.label.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
            self.label.setScaledContents(True)
        elif windows==2:
            self.label_2.setPixmap(QPixmap.fromImage(img))
            self.label_2.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
            self.label_2.setScaledContents(True)
        elif windows==3:
            self.label_3.setPixmap(QPixmap.fromImage(img))
            self.label_3.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
            self.label_3.setScaledContents(True)
        elif windows==4:
            self.label_7.setPixmap(QPixmap.fromImage(img))
            self.label_7.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
            self.label_7.setScaledContents(True)
        elif windows==5:
            self.label_4.setPixmap(QPixmap.fromImage(img))
            self.label_4.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
            self.label_4.setScaledContents(True)
        elif windows==6:
            self.label_8.setPixmap(QPixmap.fromImage(img))
            self.label_8.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
            self.label_8.setScaledContents(True)


# mempersiapkan tampilan widows
app = QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle("Alat Pendeteksi Buah")
window.show()
sys.exit(app.exec_())
