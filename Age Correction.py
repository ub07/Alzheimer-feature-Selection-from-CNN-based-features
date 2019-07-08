
import glob
import nibabel as nib
import os
import numpy as np
np.set_printoptions(precision=4, suppress=True)
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression


#Read Patients of group CN
data = pd.read_csv("ADNI1.csv")
data = data[data.Group == 3]
data2 = pd.read_csv("ADNI1_Screening_1.5T_5_26_2019.csv")
CN_Sub = data.Subject.unique()
age = data2[data2["Subject"].isin(CN_Sub)].Age
age = list(age)
img_list = []
for sub in CN_Sub:
  img = glob.glob(sub+".nii")
  img_list.append(img)
  
#Concated Array
img_data = np.empty([228,180,218,182])
for i in range(0,228):
  img = nib.load(img_list[i][0]).get_data()
  img_data[i,:,:,:] = img[:,:,:]
  
#Fitting Model  
w = np.empty([180,218,182])
for x in range(0,180):
  for y in range(0,218):
    for z in range(0,182):
      model = LinearRegression().fit(img_data[:,x,y,z].reshape(-1,1), np.array(age))
      w[x,y,z] = model.coef_                                  
  np.save("weights.npy",w) 
