#from model import Net
import cv2
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt



transform = transforms.Compose([
    transforms.ToTensor(),
    ])

imgDir = "coco.jpg"

img = cv2.imread(imgDir)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

numpydata = np.asarray(img)

if __name__ == "__main__":
    transformed_img = transform(img)
    print(transformed_img.shape)

    
    
    


