
import cv2
import numpy as np
import params as P
from Non_Max_Suppression import *


def getROIs(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
    areas = getFilter(contours)
    rectList = filterRects(contours, areas[0], areas[1])
    
    return rectList

def getFilter(contours):
    lst = []
    for cont in contours:
        area = cv2.contourArea(cont)
        if area >= P.MIN_AREA:
            lst.append(area)

    return np.percentile(lst, 50 - P.PERCENTILE_GAP), np.percentile(lst, 50 + P.PERCENTILE_GAP)

def filterRects(contours, minArea, maxArea):
    filtredCont = []
    lst = []
    for cont in contours:
        area = cv2.contourArea(cont)
        if area > minArea and area < maxArea:
            x,y,w,h = cv2.boundingRect(cont)
            lst.append([x,y, x+w, y+h])
    lista = np.array(lst)
    filtredCont = non_max_suppression_slow(lista, 0.3)

    return filtredCont

def highlighter(img, img2):
    thresh = P.HL_TRESH
    for i in range(P.HL_K):
        img = img2.copy()
        img[np.all(img > thresh, axis=2)] = 0
        img = img + img2
        thresh = thresh + P.HL_STEP
    return img

def drawROIs(img, ROIlist):
    for cont in ROIlist:
        cv2.rectangle(img, (cont[0],cont[1]),(cont[2],cont[3]), (0, 255, 255),1)
    return img

def getPos(imageSize, ROIlist):
    lstPos = []
    if len(ROIlist) > 0:
        for r in ROIlist:
            pos = calcPos(r, imageSize)
            lstPos.append(pos)
    return lstPos
            
def calcPos(ROI, imageSize):
    x = (ROI[0] - P.SXS) / P.SXS
    y = (ROI[1] - P.SXS) / P.SXS
    w = ROI[2] / imageSize[0]
    h = ROI[3] / imageSize[1]
    return [x, y, w, h]