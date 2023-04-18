from logging import getLogger
import numpy as np
import cv2
import pytesseract

class ReadTextImage:
    def __init__(self):
        
        self.log = getLogger("OCR")
        self.config_tesseract = "--tessdata-dir tessdata --psm 7 -c tessedit_char_whitelist=0123456789"
        self.lang_tesseract = "por"
    
    
    def image_to_text(self, img: np.ndarray) -> str:
        """_summary_

        Args:
            img (np.ndarray): imagem da tag apos mografia

        Returns:
            str: texto com numero detectado ou str vazia
        """
       
        
        result = self.ocr_detector(img)
        cv2.imshow('mografia', img)
        cv2.waitKey(0)
        
        return print(result)
    
    
    def preprocess(self,img:np.ndarray) -> np.ndarray:
        """_summary_
            aplica crop (recorta a parte do numero da image)
            converte para escola de cinza
            aplica threashod adaptivo (numero branco fundo preto (para aplicar morfo))
            operação morfologica de abertura (erosão seguida de dilatação, usar cv2.morphologyEx) com kernel Wx4H (autura 4 vezes maior que a largura)
            invert image (numero preto fundo branco)
        Args:
            img (np.ndarray): imagem RGB

        Returns:
            np.ndarray: image com numero preto e fundo branco
        """
        img = self.crop_img(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        val, thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)
        erosao = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))# (gray, np.ones((3, 3), np.uint8))
        invert = 255 - erosao
        return invert
    def crop_img(self, img:np.ndarray) -> np.ndarray:
        img_arr = np.array(img)
        nova_img_crop = img_arr[0:80, 253:480]
        return nova_img_crop

    def ocr_detector(self, img: np.ndarray) -> str:
        """_summary_
            aplicar tesseract
        Args:
            img (np.ndarray): image com numero preto e fundo branco

        Returns:
            str: resultado do tesseract
        """
        img = self.preprocess(img)
        
        resultado = pytesseract.image_to_string(img, lang=self.lang_tesseract, config=self.config_tesseract)
        resultado = resultado.strip()
        return resultado
    
    def posprocess(self, text:str) -> str:
        """_summary_
            valida valor encontrado pelo ocr
            o resultado deve ser numero (usar isdigits())
        Args:
            text (str): texto capturado pelo ocr

        Returns:
            str: _description_
        """


        if text.isdigit() == True:
            print('valor numerico')
        elif len(text) == 0:
            print('valor nao encontrado')
        

        return text

if __name__ == "__main__":
    ocr = ReadTextImage()
