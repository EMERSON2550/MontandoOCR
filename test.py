import cv2
import numpy as np
import pytesseract

from models.text_read import ReadTextImage
img = cv2.imread('Curso_OCR/Aluas/116d7b41fff92f0a671320d0d0a3b80d.webp')
#cv2.imshow('janela', img)
#cv2.waitKey(0)
img_crop = ReadTextImage()
img_crop = img_crop.crop_img(img)
#cv2.imshow('janela' ,img_crop)
#cv2.waitKey(0)
print(help(img))
img_ero = ReadTextImage()
img_ero = img_ero.preprocess(img)
#cv2.imshow('erosao', img_ero)
#cv2.waitKey(0)

img_tex = ReadTextImage()
img_tex = img_tex.ocr_detector(img)
print(img_tex)
print(type(img_tex))
print(img_tex)

print(len(img_tex))

vali = ReadTextImage()
vali = vali.posprocess(img_tex)
print(vali)



image_to_tex = ReadTextImage()
#image_to_tex.image_to_text(img)



'''gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
#cv2.imshow("ero", erosao)
#cv2.waitKey(0)
val, thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)
#thresh[0:25,:] =0
erosao = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))# (gray, np.ones((3, 3), np.uint8))
config_tesseract = '--tessdata-dir tessdata --psm 7 -c tessedit_char_whitelist=0123456789'
resultado = pytesseract.image_to_string(erosao, lang="por", config=config_tesseract)
print(resultado)
cv2.imshow("resul",thresh)
cv2.waitKey(0)
# #print(type(resultado))'''