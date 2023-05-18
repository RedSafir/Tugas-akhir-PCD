import numpy as np
import imutils
import cv2
import copy


"""
INISIASI VARIABLE

"""
lower = {'matang':(0, 50, 20), 'mentah':(30, 100, 100), 'setengah matang':(10, 100, 100)} 
upper = {'matang':(10,255,255), 'mentah':(80,255,255), 'setengah matang':(60,255,255)}
colors = {'matang':(0,0,255), 'mentah':(0,255,0), 'setengah matang':(0,140,255)}
target_width = 400
target_height = 400
largest_item = {}

"""
INISIASI FUNCTION

"""

# Normelize histogram
def normalisasiCitra(image):
    # Konversi citra ke float32
    image_float = image.astype(np.float32)

    # Normalisasi citra
    normalized_image = cv2.normalize(image_float, None, 0, 255, cv2.NORM_MINMAX)

    # Konversi citra kembali ke tipe data uint8
    normalized_image = normalized_image.astype(np.uint8)

    return normalized_image

def math_konvolusi(arrycitra, arrykarnel):
        # baca ukuran dimensi citra
        H_citra = arrycitra.shape[0]
        W_citra = arrycitra.shape[1]

        # baca ukuran dimensi karnel
        H_karnel = arrykarnel.shape[0]
        W_karnel = arrykarnel.shape[1]

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

def gaussianfilter(image):   
    # buat kernel
    KERNEL = (1.0 / 345) * np.array([   [1, 5, 7, 5, 1],
                                        [5, 20, 33, 20, 5],
                                        [7, 33, 55, 33, 7],
                                        [5, 20, 33, 20, 5],
                                        [1, 5, 7, 5, 1]])
    
    # lakukan konvolusi dengan karnel dan image yang sudah di buat grey
    hasil = cv2.filter2D(image, -1, KERNEL)

    return hasil

def sharpening(image):
    KERNEL = np.array([ [-1, -1, -1],
                        [-1,  9, -1],
                        [-1, -1, -1]])
    hasil = cv2.filter2D(image, -1,  KERNEL)

    return hasil

def dilasi(image, strel):
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

def erosi(image, strel):
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

def opening(image, strel):
    try:
        # Mengkonversi citra menjadi grayscale
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Mengkonversi citra grayscale menjadi citra biner menggunakan cv2.threshold
        _, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    except:
        pass
    
    # MORPH OPEN untuk erosi -> dilasi
    img_erose = erosi(image, strel)
    img_open = dilasi(img_erose, strel)

    return img_open

def closing(image, strel):
    try : 
        # Mengkonversi citra menjadi grayscale
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Mengkonversi citra grayscale menjadi citra biner menggunakan cv2.threshold
        _, image = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    except:
        pass

    # MORPH CLOSE untuk dilasi -> erosi
    img_dilas = dilasi(image, strel)
    img_open = erosi(img_dilas, strel)

    return img_open


"""
TAHAP PRE-PROCESSING IMAGE

Tahap preprocessing dalam pengolahan citra adalah serangkaian langkah atau operasi yang dilakukan 
pada citra sebelum masuk ke tahap analisis atau pemrosesan lebih lanjut. Tujuannya adalah untuk 
meningkatkan kualitas citra, menghilangkan noise atau gangguan, mengurangi variabilitas, 
dan mempersiapkan citra untuk tahap selanjutnya dalam alur pengolahan citra.

"""

# Started
image = cv2.imread("../imgs/tmt-setengah-1.jpg", cv2.IMREAD_COLOR)
# lakukan normalisasi citra
image = normalisasiCitra(image)
# lakukan resize
image = cv2.resize(image, (target_width, target_height))
# dapatkan ukuran dari gambar yang baru di resize
img_height, img_width, _ = image.shape
# melakukan penghalusan citra
blurred_img = gaussianfilter(image)
# menlakukan penajaman citra
shrap_img = sharpening(blurred_img)

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
    mask = opening(mask, STREL)
    # melakukan morfologi CLOSING
    mask = closing(mask, STREL)

    """
    Setelah setiap warna di masking, munculkan hasilnya
    """
    if key_color == "matang" :
        cv2.imshow("matang", mask)
    elif key_color == "mentah":
        cv2.imshow("mentah", mask)
    elif key_color == "setengah matang":
        cv2.imshow("setengah matang", mask)

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
    else:
        print("object tidak terdefinisi")
except :
    print("tidak ada obejct")

cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows() 