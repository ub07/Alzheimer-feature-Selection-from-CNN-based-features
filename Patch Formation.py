!git clone https://github.com/dscarmo/e2dhipseg
!pip install sparse
!pip install dipy
import os
import random
os.chdir("e2dhipseg")

#Segmentation of Hippocampal region
!python3 run.py "Age Corrected Images" -dir

#Choose any one segmented Image(Preffered with most counts)
im_data = nib.load("002_S_0729.nii_voxelcount-8088_e2dhipmask.nii.gz").get_data()
index = np.argwhere(im_data)

index_list = []
index_list.append(list(random.choice(index)))
while(len(index_list) != 151):
  tindex = list(random.choice(index))
  count = 0
  for items in index_list:
    if((abs(np.sqrt((items[0]-tindex[0])**2 + (items[1]-tindex[1])**2 + (items[2]-tindex[2])**2) < 2))):
      break
    count = count+1  
  print(count,len(index_list))
  if(count == len(index_list)):
    index_list.append(tindex)
    
index_list.sort()
img_list = glob.glob("Age Corrected Images/*")
for img in img_list:
  data = nib.load(img).get_data()
  i = 1
  for ind in index_list:
    Zpr,Zpo = ind[0]-16,ind[0]+16
    Fpr,Fpo = ind[1]-16,ind[1]+16
    Spr,Spo = ind[1]-16,ind[1]+16
    Z,F,S = ind[0],ind[1],ind[2]
    Sim = data[Z,Fpr:Fpo,Spr:Spo]
    Cim = data[Zpr:Zpo,F,Spr:Spo]
    Tim = data[Zpr:Zpo,Fpr:Fpo,S]
    im =  np.stack((Tim,Cim,Sim),axis=2)
    cv2.imwrite(img+"."+str(i)+".jpeg",im)
    i = i+1
    



