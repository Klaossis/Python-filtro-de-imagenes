import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PyQt5 import uic, QtGui, QtCore
from PyQt5.QtWidgets import QMainWindow, QApplication, QDialog, QLabel, QPushButton, QFileDialog, QLineEdit

from os import getcwd
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import Qt, pyqtSignal


#textopor = StringVar()
imagen = 0


class proyecto_pi(QMainWindow):

    def __init__(self):
        super().__init__()
        uic.loadUi("gui_app1.ui", self)
        self.setWindowTitle("Proyecto filtros")
        self.pushButton.clicked.connect(self.selecionarImagen)
        #self.pushButton_2.clicked.connect(lambda: self.label.clear())
        self.pushButton_3.clicked.connect(self.restaurar)
        self.pushButton_4.clicked.connect(self.descargarima)
        self.pushButton_5.clicked.connect(self.filbilateral)
        self.pushButton_6.clicked.connect(self.blanconegro)
        self.pushButton_7.clicked.connect(self.laplacian)
        self.pushButton_8.clicked.connect(self.correciongama)
        self.pushButton_9.clicked.connect(self.adaptativo)
        #self.pushButton_10.clicked.connect(self.histograma2)






        #buttonEliminar.clicked.connect(lambda: self.labelImagen.clear())

    #def blanco_negro(self):

#===================Descargar Imagen ====================================

    def descargarima(self):
        #global imagencv2
        nombre = self.lineEdit.text()
        #print(nombre + ".jpg")
        nombre = nombre + ".jpg"
        cv2.imwrite(nombre,imagencv2)



#====================== Selecionar Imagenes ==============================================

    def selecionarImagen(self):
        global imagencv2
        global imagen
        imagen, extension = QFileDialog.getOpenFileName(self, "Seleccionar imagen", getcwd(),
                                                        "Archivos de imagen (*.png *.jpg)",
                                                        options=QFileDialog.Options())
        print(imagen)

        if imagen:
            # Adaptar imagen
            pixmapImagen = QPixmap(imagen).scaled(400, 1250, Qt.KeepAspectRatio,
                                                  Qt.SmoothTransformation)

            # Mostrar imagen
            self.label.setPixmap(pixmapImagen)
            imagencv2=cv2.imread(imagen,0)
            #cv2.imshow('prueba de imagen1', imagencv2)
            #self.label_2.setPixmap(pixmapImagen)


#============= Filtro Bilateral ========

    def filbilateral(self):
        global imagencv2
        imagentemp = cv2.imread(imagen)
        imagencv2 = cv2.bilateralFilter(imagentemp, 20, sigmaColor=100, sigmaSpace=4000)
        self.image = imagencv2
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.label_2.setPixmap(QtGui.QPixmap.fromImage(self.image))
        #cv2.imshow('prueba de imagen', imagencv2)

#============= Blanco y Negro ==============

    def blanconegro(self):
        global imagencv2
        imagencv2 = cv2.imread(imagen,0)
        #cv2.imshow('prueba de imagen blanco negro', imagencv2)

        #self.image = cv2.imread(imagen,0)
        self.image = imagencv2
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_Indexed8).rgbSwapped()#RGB888).rgbSwapped()#BGR2RGB
        #self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGBA8888)#.rgbSwapped()
        self.label_2.setPixmap(QtGui.QPixmap.fromImage(self.image))

#=============

        #self.image = cv2.imread('placeholder4.PNG')
        #self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        #self.image_frame.setPixmap(QtGui.QPixmap.fromImage(self.image))

#============= Restauracion =============================

    def restaurar(self):
        global imagencv2
        imagencv2 = cv2.imread(imagen)
        imagencv2 = cv2.medianBlur(imagencv2,5)
        self.image = imagencv2
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.label_2.setPixmap(QtGui.QPixmap.fromImage(self.image))


#============= Laplacian ==================
    def laplacian(self):
        global imagencv2
        imagencv2 = cv2.imread(imagen)
        imagencv2 = cv2.Laplacian(imagencv2, cv2.CV_64F)
        cv2.imshow('prueba de imagen1', imagencv2)
        temp = imagencv2

        #self.image = temp
        #self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888)
        #self.label_2.setPixmap(QtGui.QPixmap.fromImage(self.image))

#============= Adaptativo ==============
    def adaptativo(self):
        global imagencv2
        imagencv2 = cv2.imread(imagen)
        imagencv2 = cv2.cvtColor(imagencv2,cv2.COLOR_BGR2GRAY)
        bw = cv2.adaptiveThreshold(imagencv2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        imagencv2 = bw
        #cv2.imshow('prueba de imagen1', bw)

        self.image = imagencv2
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_Indexed8).rgbSwapped()
        self.label_2.setPixmap(QtGui.QPixmap.fromImage(self.image))


#=============
    def correciongama(self):
        global imagencv2

        imagencv2 = cv2.imread(imagen)

        𝛾 = 2

        gamma = imagencv2 / 255
        gamma = gamma ** (1.0 / 𝛾)
        gamma = gamma * 255
        gamma = gamma.astype(np.dtype('uint8'))
        imagencv2 = gamma


        self.image = imagencv2
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.label_2.setPixmap(QtGui.QPixmap.fromImage(self.image))

#============= histograma
    def histograma(self):
        print("coco salado")
        lena = cv2.imread('lena.jpg')
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(lena, cv2.COLOR_BGR2RGB))

        #def channel_hist(channel):
        #    hist = np.zeros(256)
         #   for pixel in channel.flatten():
          #      hist[pixel] += 1
           # return hist

        #hist_b = channel_hist(lena[:, :, 0])
        #hist_g = channel_hist(lena[:, :, 1])
        #hist_r = channel_hist(lena[:, :, 2])



#=============
    def histograma2(self):

        imagencv2 = cv2.imread(imagen)
        lena = cv2.imread(imagen)
        print("lena salsa")

        hist_b = channel_hist(lena[:,:,0])
        hist_g = channel_hist(lena[:,:,1])
        hist_r = channel_hist(lena[:,:,2])
        print("lena salsa")
        plt.figure(figsize=(10, 10))

        cumsum_b = hist_b.cumsum()
        cumsum_g = hist_g.cumsum()
        cumsum_r = hist_r.cumsum()

        plt.plot(cumsum_b, color='b', label='blue')
        plt.plot(cumsum_g, color='g', label='green')
        plt.plot(cumsum_r, color='r', label='red')

        plt.title('Distribución acumulada histograma')
        plt.legend()

        cumsum_b = (cumsum_b - cumsum_b.min()) * 255 / (cumsum_b.max() - cumsum_b.min())
        cumsum_g = (cumsum_g - cumsum_g.min()) * 255 / (cumsum_g.max() - cumsum_g.min())
        cumsum_r = (cumsum_r - cumsum_r.min()) * 255 / (cumsum_r.max() - cumsum_r.min())

        lena_equalized_b = cumsum_b[lena[:, :, 0]]
        lena_equalized_g = cumsum_g[lena[:, :, 1]]
        lena_equalized_r = cumsum_r[lena[:, :, 2]]

        plt.figure(figsize=(10, 10))

        cumsum_b = lena_equalized_b.cumsum()
        cumsum_g = lena_equalized_g.cumsum()
        cumsum_r = lena_equalized_r.cumsum()

        plt.plot(cumsum_b, color='b', label='blue')
        plt.plot(cumsum_g, color='g', label='green')
        plt.plot(cumsum_r, color='r', label='red')

        plt.title('Distribución acumulada histograma')
        plt.legend()

        new_lena = np.empty(shape=lena.shape)

        new_lena[:, :, 0] = lena_equalized_b
        new_lena[:, :, 1] = lena_equalized_g
        new_lena[:, :, 2] = lena_equalized_r
        new_lena = new_lena.astype(np.dtype('uint8'))

        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(new_lena, cv2.COLOR_BGR2RGB))



#=============

    def channel_hist(channel):
        hist = np.zeros(256)
        for pixel in channel.flatten():
            hist[pixel] += 1
        return hist

#=============

if __name__== '__main__':
    app = QApplication(sys.argv)
    GUI = proyecto_pi()
    GUI.show()
    sys.exit(app.exec_( ))