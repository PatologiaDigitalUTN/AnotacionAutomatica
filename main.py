import cv2
import os 
import numpy as np
import params as P
import functions as F

def processImage(file, showImage):
    imgOriginal =  cv2.imread(P.PATH + image)
    if not imgOriginal is None:
        imgCopy = imgOriginal.copy()
        
        imgCopy = F.highlighter(imgCopy, imgOriginal)
        imgCopy = imgOriginal - imgCopy
        
        imgGray = cv2.cvtColor(imgCopy, cv2.COLOR_BGR2GRAY)
        th, imgThresh = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        imgGaussian = cv2.GaussianBlur(imgThresh,P.GB_SIZE,0)

        closing = cv2.morphologyEx(imgGaussian, cv2.MORPH_CROSS, P.ME_KERNEL, iterations=P.ME_K)
        closing = cv2.erode(closing, P.E_SIZE, iterations=P.E_K)
        
        morph = cv2.Laplacian(closing, cv2.CV_64F)
        max=np.max(morph)
        div = max/float(255) 
        imgGaussian = np.uint8(np.round(morph / div))
        lstROIs = F.getROIs(imgGaussian)
        pos = F.getPos(imgOriginal.shape, lstROIs)
        print(pos)

        if showImage:
            imgROIs = F.drawROIs(imgOriginal, lstROIs)
            cv2.imshow(image, imgROIs)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

images = os.listdir(P.PATH)

for image in images:
    processImage(image, True)
