import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('https://sharon.srworkspace.com/ml/datasets/hw2/exams2.csv', header=None)
df.head(3)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

pd = df.to_numpy()
X = pd[:, :-1]
y = pd[:,-1]
y = np.where (y==1,1,0)

noise = np.random.normal(0, 1, X.shape)
X = X + noise

X_temp, X_test, y_temp, y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp,y_temp,test_size=0.2, stratify=y_temp,random_state=42)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

def sigmoid(z):
  return 1/(1+np.exp(-z))


def Logistic_Regression_via_GD(X, y, lr, lamda=0.0):
    n, d = X.shape
    w = np.zeros(d)
    b = 0
    tol = 1e-5
    n = X.shape[0]
    z = np.zeros(n) 
    p = np.zeros(n)   
    err = np.zeros(n)   
    losses = []
    
    for i in range(3000):    
      z = X@w + b
      p = sigmoid(z)
      err = p - y 
      if (lamda != 0): 
        loss = - (1.0/n) * np.sum(y*np.log(p+1e-15) + (1-y) * np.log(1-p+1e-15))
        reg  = lamda * np.sum(np.abs(w))
        loss_reg = loss + reg
        losses.append(loss_reg)

      w_gradient = (1.0/n) * X.T @ err +lamda * np.sign(w)
      b_gradient = (1.0/n) * np.sum(err)

      grad_norm = np.linalg.norm(w_gradient)      
      if grad_norm < tol and abs(b_gradient) < tol:
            break
      
      w-=lr*w_gradient
      b-=lr*b_gradient
    
    return w,b

import matplotlib.pyplot as plt

def plot(data, labels, w, bias):

    plt.scatter(data[:,0], data[:,1], c=labels)

    a, b, c = w[0], w[1], bias

    m = -a / b
    b = -c / b

    x = np.arange(np.min(data[:,0]), np.max(data[:,0]), 0.1)
    y = m * x + b

    plt.plot(x, y)
    plt.show()



def predict(w,b,x):
    return np.sign(np.dot(w, x) + b)

w, b = Logistic_Regression_via_GD(X_train, y_train, lr = 0.1)
plot(X_test, y_test, w, b)
preds = (sigmoid(X_test.dot(w) + b) >= 0.5).astype(int)
accuracy = np.mean(preds == y_test)
print(f"Test accuracy is {accuracy * 100}%")

lamdas = [0.0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0]
val_accuracies = []

for lam in lamdas:
    w, b = Logistic_Regression_via_GD(X_train, y_train,0.1,lam)
    preds_val = (sigmoid(X_val.dot(w) + b) >= 0.5).astype(int)
    acc = np.mean(preds_val == y_val)
    val_accuracies.append(acc)

# plot
plt.plot(lamdas, val_accuracies, marker='o')
plt.xlabel("lamda")
plt.ylabel("accuracy")
plt.show()
