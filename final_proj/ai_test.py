import cv2
from demofunction import ocr

if __name__ == "__main__":
    img = cv2.imread('./ocrtemp/output.png')
    ocr = ocr(img)
    print(ocr)
