import tensorflow as tf
import numpy as np
import glob
import nibabel as nib
import os
import numpy as np
np.set_printoptions(precision=4, suppress=True)
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold

# Number of Basis Function,feature group and size
l = 151
size = 1024
# Number of nodes in hidden layer
Zs = 10
nc = 2
e = tf.constant(np.power(3,1),shape=[l],dtype = tf.float32)
# Regularization rate
bta = 0.01


# declare the training data placeholder
# input X 151 Patches 
x = tf.placeholder(tf.float32,[None,l*size])
# Output data placeholder
y = tf.placeholder(tf.float32,[None,nc])
lr2 = tf.placeholder(tf.float32,[ ])
lr1 = tf.placeholder(tf.float32,[ ])

beta = tf.Variable(e,name = "beta")
W1 = tf.Variable(tf.compat.v1.random.uniform([l*size,Zs], minval = -.2,maxval = .2),name = "W1")
W2 = tf.Variable(tf.compat.v1.random.uniform([Zs,nc], minval = -.2,maxval = .2),name = "W2")

output_list = []
for i in range(0,l*size,size):
   output_list.append(x[:,i:i+size]*tf.exp(-1.0*(beta[i//size]**2)))
Z1 = tf.concat(output_list,axis=1)    
Z2 = tf.matmul(Z1, W1)
Z2 = tf.nn.sigmoid(Z2)
Z3 = tf.matmul(Z2, W2)

#Sample Loss
loss = tf.compat.v1.losses.sigmoid_cross_entropy(y,Z3)
regularizer = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)
loss = tf.reduce_mean(loss + bta*regularizer)

optimiser1 = tf.train.AdamOptimizer(learning_rate=lr1).minimize(loss,var_list=[W1,W2])
optimiser2 = tf.train.AdamOptimizer(learning_rate=lr2).minimize(loss,var_list=beta)
optimiser = tf.group(optimiser1, optimiser2)

Y = np.load("MCI(Output of patients).npy")
X = np.load("features.npy")

#feature normalization
for i in range(0,151*1024):
  if((np.max(X[:,i]) - np.min(X[:,i]))==0):
      continue
  X[:,i] = (X[:,i] - np.min(X[:,i]))/(np.max(X[:,i]) - np.min(X[:,i]))
  
k = []
kfold = KFold(n_splits=10, shuffle=True)
for train,test in kfold.split(X,Y):
  init = tf.global_variables_initializer()
  sess = tf.Session()
  sess.run(init)
  j=0
  for i in range(150):
    _ ,loss_value, beta6 = sess.run([optimiser,loss, beta],feed_dict={x: X[train], y: Y[train],lr2:0,lr1:0.2})
    j=j+1

  for i in range(250):
    _,loss_value, beta6,Yo = sess.run([optimiser,loss, beta,Z3],feed_dict={x: X[train], y: Y[train],lr2:0.2,lr1:0.02})
    j=j+1

  sess.close()
  
  ns = len(X[train])
  Yo = sigmoid(Yo)  
  Yout= predictor(Yo)
  Yout = Yout.astype(int)
  count =0  
  yt = Y[train]
  for i in range(0,ns):
    if(np.array_equal(Yout[i],yt[i])):
       count=count+1
  print("TrAccuracy: {:f}".format(count/ns))
  
  ns = len(X[test])
  Yo = sigmoid(Yo)  
  Yout= predictor(Yo)
  Yout = Yout.astype(int)
  count =0  
  yt = Y[test]
  for i in range(0,ns):
    if(np.array_equal(Yout[i],yt[i])):
       count=count+1
  print("TeAccuracy: {:f}".format(count/ns))
  
  k1 = np.where(np.exp(-1.0*(beta6**2))>01)
  k.append(k1)
  
Imp(Grouped)features = np.concat(k)



  
