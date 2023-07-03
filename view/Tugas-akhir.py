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
        loadUi("GUI_4.ui", self)
        self.Image = None

        # Action Button
        self.actionFoto.triggered.connect(self.mendeteksiTomat)
        self.actionCamera.triggered.connect(self.mendeteksiTomatCam)
        self.actionSave.triggered.connect(self.save)

    """
    INISIASI FUNCTION

    """
    def normalize_image(self, image):
        # Menentukan nilai minimum dan maksimum piksel
        min_value = np.min(image)
        max_value = np.max(image)

        # Melakukan normalisasi pada setiap piksel
        normalized_image = (image - min_value) / (max_value - min_value) * 255

        # Mengubah tipe data menjadi uint8
        normalized_image = normalized_image.astype(np.uint8)

        return normalized_image
    
    def normalisasiCitra_Cam(self, image):
        # Konversi citra ke float32
        image_float = image.astype(np.float32)

        # Normalisasi citra
        normalized_image = cv2.normalize(image_float, None, 0, 255, cv2.NORM_MINMAX)

        # Konversi citra kembali ke tipe data uint8
        normalized_image = normalized_image.astype(np.uint8)

        return normalized_image
        
    def math_konvolusi(self, arrycitra, arrykarnel):
        # baca ukuran dimensi citra
        H_citra, W_citra = arrycitra.shape[:2]

        # baca ukuran dimensi karnel
        H_karnel, W_karnel = arrykarnel.shape[:2]

        # meenutukan titik tengah
        H = H_karnel // 2
        W = W_karnel // 2   

        out = np.zeros((H_citra, W_citra))

        # menggeser karnel konvolusi
        for i in range(H, H_citra - H):
            for j in range(W, W_citra - W):
                citra_values = arrycitra[i - H:i + H + 1, j - W:j + W + 1]
                sum = np.sum(citra_values * arrykarnel)
                out[i, j] = sum
                
        return out
    
    def math_konvolusi(self, image, kernel):
        # Ambil dimensi gambar
        height, width, channels = image.shape

        # Ambil dimensi kernel
        k_height, k_width = kernel.shape

        # Tentukan offset titik tengah kernel
        k_h_offset = k_height // 2
        k_w_offset = k_width // 2

        # Buat citra output dengan ukuran yang sama dengan citra input
        output_image = np.zeros_like(image, dtype=np.float32)

        # Looping melalui setiap piksel pada citra input
        for h in range(k_h_offset, height - k_h_offset):
            for w in range(k_w_offset, width - k_w_offset):
                # Looping melalui setiap saluran warna (B, G, R)
                for c in range(channels):
                    # Inisialisasi jumlah konvolusi
                    conv_sum = 0.0

                    # Looping melalui setiap elemen dalam kernel
                    for kh in range(k_height):
                        for kw in range(k_width):
                            # Ambil nilai piksel pada posisi yang sesuai dalam kernel dan citra input
                            pixel_value = image[h - k_h_offset + kh, w - k_w_offset + kw, c]
                            kernel_value = kernel[kh, kw]

                            # Hitung konvolusi untuk saluran warna saat ini
                            conv_sum += pixel_value * kernel_value

                    # Simpan hasil konvolusi pada saluran warna saat ini
                    output_image[h, w, c] = conv_sum

        # Batasi nilai piksel dalam rentang 0-255
        output_image = np.clip(output_image, 0, 255)

        # Konversi citra kembali ke tipe data uint8
        output_image = output_image.astype(np.uint8)

        return output_image

    def gaussianfilter(self, image):   
        # buat kernel
        KERNEL = (1.0 / 345) * np.array([   [1, 5,  7,  5,  1],
                                            [5, 20, 33, 20, 5],
                                            [7, 33, 55, 33, 7],
                                            [5, 20, 33, 20, 5],
                                            [1, 5,  7,   5, 1]])
        
        # lakukan konvolusi dengan karnel dan image yang sudah di buat grey
        hasil = self.math_konvolusi(image, KERNEL)

        return hasil

    def sharpening(self, image):

        KERNEL = np.array([ [-1, -1, -1],
                            [-1,  9, -1],
                            [-1, -1, -1]])
        
        hasil = self.math_konvolusi(image, KERNEL)

        return hasil
    
    def dilasi(self, image, kernel):
        # Ambil dimensi gambar
        height, width = image.shape

        # Ambil dimensi kernel
        k_height, k_width = kernel.shape

        # Tentukan offset titik tengah kernel
        k_h_offset = k_height // 2
        k_w_offset = k_width // 2

        # Buat citra output dengan ukuran yang sama dengan citra input
        output_image = np.zeros_like(image)

        # Looping melalui setiap piksel pada citra input
        for h in range(k_h_offset, height - k_h_offset):
            for w in range(k_w_offset, width - k_w_offset):
                # Periksa apakah ada piksel bernilai 255 dalam sekitar piksel saat ini
                if np.any(image[h - k_h_offset : h + k_h_offset + 1, w - k_w_offset : w + k_w_offset + 1] == 255):
                    # Set piksel saat ini pada citra output menjadi 255
                    output_image[h, w] = 255

        return output_image
    
    def erosi(self, image, strel):
        # mencari bentuk gambar
        m,n= image.shape #Show the image
        
        # mencari nilai tengah dari strel
        constant = (15-1)//2
        
        #Erosion without using inbuilt cv2 function for morphology
        imgErode= np.zeros((m,n), dtype=np.uint8)

        for i in range(constant, m-constant):
            for j in range(constant,n-constant):
                temp = image[i-constant:i+constant+1, j-constant:j+constant+1]
                product = temp*strel
                imgErode[i,j] = np.min(product)
        
        return imgErode
    
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

        self.displayImage(copy.deepcopy(image), 1)

        # lakukan histogram equalization
        image = self.normalize_image(image.copy())
        # lakukan resize
        image = cv2.resize(image, (target_width, target_height))
        # melakukan penghalusan citra
        blurred_img = self.gaussianfilter(image)
        # menlakukan penajaman citra
        shrap_img = self.sharpening(blurred_img) 

        self.displayImage(shrap_img.copy(), 2)

        """
        PROCESSING IMAGE

        """
        image_hsv = cv2.cvtColor(shrap_img.copy(), cv2.COLOR_BGR2HSV)
        # loop setiap warna
        for key_color, color in upper.items():
            # inisiasi STREL
            STREL = np.ones((15,15),np.uint8)
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
                if radius > 90:
                    # gambar conture berserta dengan text
                    cv2.circle(image, (int(X), int(Y)), int(radius), colors[key], 2)
                    cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 255), 2)
                    self.Image = image.copy()
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
        lower = {'matang':(0, 50, 50    ), 'mentah':(30, 50, 20), 'setengah matang':(10, 50 , 50)} 
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
            image = self.normalisasiCitra_Cam(image)
            # lakukan resize
            image = imutils.resize(image, width=target_width, height=target_height)
            # melakukan penghalusan citra
            blurred_img = cv2.GaussianBlur(image, (11, 11), 0)
            # menlakukan penajaman citra
            KERNEL = np.array([ [-1, -1, -1],
                                [-1,  9, -1],
                                [-1, -1, -1]])
            shrap_img = cv2.filter2D(blurred_img, -1, KERNEL) 

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
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, STREL)
                # melakukan morfologi CLOSING
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, STREL)

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

                    if radius > 90:
                        # gambar conture berserta dengan text
                        cv2.circle(image, (int(X), int(Y)), int(radius), colors[key], 2)
                        cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 255), 2)
                        self.displayImage(image, 6)
                    else :
                        pass
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
