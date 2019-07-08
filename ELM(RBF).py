import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import rbf_kernel
X3 = np.load("Selected Features.npy")
Y3 = np.load("MCI(Output)")

cvscores = []
kfold = KFold(n_splits=10, shuffle=True)
for train, test in kfold.split(X3, Y3):
  X_train = X3[train]
  X_test = X3[test]
  k = np.empty([len(X_train),len(X_test)])
  for i in range(0,len(X_train)):
    k[i] = rbf_kernel(X_test,X_train[i:i+1]).reshape([len(X_test)])
  omega = np.empty([len(X_train),len(X_train)])
  for i in range(0,len(X_train)):
    for j in range(0,len(X_train)):
      omega[i,j] = rbf_kernel(X_train[i:i+1],X_train[j:j+1])
  y  = np.matmul(np.matmul(np.transpose(k),np.transpose(omega+1)),Ym[train])    
  correct = 0
  total = y.shape[0]
  Yt = Ym[test]
  for i in range(total):
    predicted = np.argmax(y[i])
    test = np.argmax(Yt[i])
    correct = correct + (1 if predicted == test else 0)
  print('Accuracy: {:f}'.format(correct/total))
  cvscores.append((correct/total)*100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))  
