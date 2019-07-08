!pip install dipy
!pip install nibabel
!pip install deepbrain
import glob
import nibabel as nib
from deepbrain import Extractor
import os
import numpy as np
np.set_printoptions(precision=4, suppress=True)
import matplotlib.pyplot as plt
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.imwarp import DiffeomorphicMap
from dipy.align.metrics import CCMetric
from dipy.viz import regtools
from dipy.align.imaffine import (AffineMap,MutualInformationMetric,AffineRegistration)
from dipy.align.transforms import (TranslationTransform3D,RigidTransform3D,AffineTransform3D)
import pandas as pd

#Read Image
template_img = nib.load("MNI152_T1_1mm_Brain.nii.gz")
template_data = template_img.get_data()
template_affine = template_img.affine
NIFTI_list = glob.glob("Images/*")
NIFTI_list.sort()


#Image Registration and Brain Extraction and normalization
for img in NIFTI_list:
  moving_img = nib.load(img)
  #Extract Brain
  moving_data = moving_img.get_data()
  ext = Extractor()
  prob = ext.run(moving_data)
  mask = prob>0.9
  moving_data= moving_data*mask
  moving_affine = moving_img.affine
  # The mismatch metric
  nbins = 32
  sampling_prop = None
  metric = MutualInformationMetric(nbins, sampling_prop)

  # The optimization strategy
  level_iters = [10, 10, 5]
  sigmas = [3.0, 1.0, 0.0]
  factors = [4, 2, 1]

  affreg = AffineRegistration(metric=metric,level_iters=level_iters,sigmas=sigmas,factors=factors)

  transform = TranslationTransform3D()
  params0 = None
  translation = affreg.optimize(template_data, moving_data, transform, params0,template_affine, moving_affine)

  transform = RigidTransform3D()
  rigid = affreg.optimize(template_data, moving_data, transform, params0,template_affine, moving_affine,starting_affine=translation.affine)

  transform = AffineTransform3D()
  affreg.level_iters = [1000, 1000, 100]
  affine = affreg.optimize(template_data, moving_data, transform, params0,template_affine, moving_affine,starting_affine=rigid.affine)

  # The mismatch metric
  metric = CCMetric(3)
  # The optimization strategy:
  level_iters = [10, 10, 5]

  # Registration object
  sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)
  mapping = sdr.optimize(template_data, moving_data, template_affine,moving_affine,affine.affine)
  warped_moving = mapping.transform(moving_data)
  fimg = nib.Nifti1Image(warped_moving, template_affine)
   
  #nyul based normalization
  nyul_normalize(NIFTI_list,standard_hist)
  
  
  




