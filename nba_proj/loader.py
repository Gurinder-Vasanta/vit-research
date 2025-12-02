import cv2
import numpy as np

def preprocess_frame(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (768, 432), interpolation=cv2.INTER_AREA)
    return img