# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 20:46:44 2021

@author: niloy
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

#Specify the name/directory of the input image
INPUT_IMAGE = "image2.JPG"

#Read an input image 
img = cv2.imread(INPUT_IMAGE)

#Convert the rbg image to grayscale, hsv and lab format
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)

#Extract the separate channels 
Ib,Ig,Ir = cv2.split(img)
h,s,v = cv2.split(hsv)
L,A,B=cv2.split(lab)


V = np.zeros(Ib.shape, dtype = 'uint8')
S = np.zeros(Ib.shape)
C = np.zeros(Ib.shape)
F = np.zeros(Ib.shape)


#Calculate V which is the intensity matrix using Equation (1)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        
        V[i][j] = max(Ib[i][j], Ig[i][j], Ir[i][j])

#Calcuate S which is the saturation matrix using Equation (2)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        
        S[i][j] = (V[i][j] - min(Ib[i][j], Ig[i][j], Ir[i][j]))/V[i][j]


# compute minimum and maximum in 17x17 region using erode and dilate
kernel = np.ones((17,17),np.uint8)
Lmin = cv2.erode(L,kernel,iterations = 1, borderType = cv2.BORDER_REPLICATE )
Lmax = cv2.dilate(L,kernel,iterations = 1, borderType = cv2.BORDER_REPLICATE )

# convert min and max to floats
Lmin = Lmin.astype(np.float64) 
Lmax = Lmax.astype(np.float64) 

# compute local contrast
C = (Lmax-Lmin)/(Lmax+Lmin)



# compute  glare occurrence map Gphoto using Equation  (3)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):        
        F[i][j] = gray[i][j]*(1-S[i][j])*(1-C[i][j])
        
        
#Normalize Gmap in range (1,0)
normalizedImg = np.zeros(Ib.shape)
normalizedImg = cv2.normalize(F,  normalizedImg, 0, 1, cv2.NORM_MINMAX)

#Display the output Gmap
plt.imshow(normalizedImg, cmap='jet', interpolation='nearest')
plt.colorbar()
plt.show()

#Save image file
plt.savefig("Output of " + INPUT_IMAGE[:-4] + ".png")




