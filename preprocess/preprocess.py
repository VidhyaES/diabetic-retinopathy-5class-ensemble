import cv2
import numpy as np
import os

def crop_image_from_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > 10
    return img[np.ix_(mask.any(1), mask.any(0))]

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def preprocess_image(path):
    img = cv2.imread(path)
    img = crop_image_from_gray(img)
    img = cv2.resize(img, (224,224))
    img = apply_clahe(img)
    img = img / 255.0
    return img
