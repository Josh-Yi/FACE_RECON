import cv2
import numpy
import numpy as np
import time

import torch
from tqdm import tqdm
from LeNet5 import LeNet5
cap = cv2.VideoCapture(0)
pix = 128
model = LeNet5(4)
model = torch.load('Model.pth')
model = model.double()
for i in range(2000):
    time.sleep(0.2)
    ret, frame = cap.read()
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = frame[:, 280:1000, :]
    frame = cv2.resize(frame, (pix, pix))
    # i += 1
    cv2.imshow("capture", frame)
    frame = frame.reshape(-1,3,pix, pix)
    frame = torch.from_numpy(frame)
    frame = frame.double()
    z = model(frame)
    # print(z)
    dic = {0:'Josh',1:'No one is here',2:'Yuanhao',3:'Xinyue'}
    # print(z)
    z=int(z.argmax(1))
    print(dic[z])

cap.release()
cv2.destroyAllWindows()