import cv2
import numpy as np
import sys
# Oscar Flores NC = 1389
# Función auxiliar para mostrar y guardar evitando repetir código
def mostrar_y_guardar(titulo_ventana, nombre_archivo, img):
    cv2.imshow(titulo_ventana, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(nombre_archivo, img)

# Read the image from file
image = cv2.imread('koa.jpg')
# Display the image in a window
cv2.imshow('koa imagen original 1389', image)
# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
#print("-+-+-+-+-+-+-+-+-+-+-+-+-+-")

# 1. Grayscale
image = cv2.imread('koa.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
mostrar_y_guardar('Escala de grises 1389', 'gray_koa-1389.jpg', gray_image)
#print("-+-+-+-+-+-+-+-+-+-+-+-+-+-")

# 2. Resize
image = cv2.imread('koa.jpg')
resized_image = cv2.resize(image, (300, 200))
mostrar_y_guardar('Redimensionada 1389', 'resized_koa-1389.jpg', resized_image)
#print("-+-+-+-+-+-+-+-+-+-+-+-+-+-")

# 3. Rotate
image = cv2.imread('koa.jpg')
(h, w) = image.shape[:2]
center = (w // 2, h // 2)
matrix = cv2.getRotationMatrix2D(center, 90, 1.0)
rotated_image = cv2.warpAffine(image, matrix, (w, h))
mostrar_y_guardar('Rotada 90 grados 1389', 'rotated_koa-1389.jpg', rotated_image)
#print("-+-+-+-+-+-+-+-+-+-+-+-+-+-")

# 4. Gaussian Blur
image = cv2.imread('koa.jpg')
blurred_image = cv2.GaussianBlur(image, (15, 15), 0)
mostrar_y_guardar('Desenfoque Gaussiano 1389', 'blurred_koa-1389.jpg', blurred_image)
#print("-+-+-+-+-+-+-+-+-+-+-+-+-+-")

# 5. Canny Edges
image = cv2.imread('koa.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray_image, 100, 200)
mostrar_y_guardar('Bordes Canny 1389', 'delineado_koa-1389.jpg', edges)
#print("-+-+-+-+-+-+-+-+-+-+-+-+-+-")

# 6. Rectangle and Text
image = cv2.imread('koa.jpg')
cv2.rectangle(image, (50, 50), (200, 200), (255, 0, 0), 3)
cv2.putText(image, 'OpenCV', (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
mostrar_y_guardar('Rectangulo y Texto 1389', 'output_koa-1389.jpg', image)
#print("-+-+-+-+-+-+-+-+-+-+-+-+-+-")

# 7. Thresholding
image = cv2.imread('koa.jpg', 0)
_, thresh_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
mostrar_y_guardar('Umbralizacion 1389', 'threshold_koa-1389.jpg', thresh_image)
#print("-+-+-+-+-+-+-+-+-+-+-+-+-+-")

# 8. Contours
image = cv2.imread('koa.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray_image, 100, 200)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
mostrar_y_guardar('Contornos 1389', 'contours_koa-1389.jpg', image)
print("Programa realizado por Oscar Flores NC = 1389")