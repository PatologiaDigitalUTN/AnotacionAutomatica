import numpy as np

#General
PATH = "images/"

#Highlighter
HL_TRESH = 50
HL_STEP = 10
HL_K = 7

#MorphologyEx
ME_KERNEL = np.ones((3, 3), np.uint8)
ME_K = 1 

#Gaussian Blur
GB_SIZE = (3,3)

#Erode
E_SIZE = (3,3)
E_K = 2

#Filtro de rectángulos
MIN_AREA = 50
PERCENTILE_GAP = 45

#Etiquetado
SXS = 40
