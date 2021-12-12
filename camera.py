import cv2
import numpy
import numpy as np
import time

import torch
from tqdm import tqdm
cap = cv2.VideoCapture(0)
pix = 128
imgs = numpy.empty((pix, pix, 3, 1))
i = 0
print('-'*66)
print(' '*24,'CHEESE 😁😁😁 ')
print('-'*66)
for i in tqdm(range(200)):
    time.sleep(0.05)
    ret, frame = cap.read()
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = frame[:, 280:1000, :]
    frame = cv2.resize(frame, (pix, pix))
    i += 1
    cv2.imshow("capture", frame)
    frame = frame.reshape(pix, pix, 3, 1)
    imgs = np.concatenate((imgs, frame), axis=3)
    # print(imgs.shape,end='')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
imgs = imgs[:, :, :, 1:]
print(imgs.shape)
np.save('imgs_1.npy',imgs)
cap.release()
cv2.destroyAllWindows()
