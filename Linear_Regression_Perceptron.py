import numpy as np
import matplotlib.pyplot as plt

#@title load npy file

import requests
from io import BytesIO

def load_npy_file(url):
  response = requests.get(url)
  if response.status_code == 200:
    npy_data = np.load(BytesIO(response.content), allow_pickle=True).item()
    return npy_data
  else:
    return None
  
data_dict = load_npy_file('https://sharon.srworkspace.com/ml/datasets/hw2/perceptron_data.npy')

X_train = data_dict['X_train']
y_train = data_dict['y_train']
X_test = data_dict['X_test']
y_test = data_dict['y_test']




y_train = np.where(y_train == 0, -1, 1)
y_test = np.where(y_test == 0, -1, 1)

def perceptron(data, labels, batch_size):
    lr = 0.05
    bias = 0 
    losses = []

    weights = np.ones(data.shape[1])
    gradient_norm_threshold = 1e-3

    while True:
        predictions = np.sign(data @ weights + bias)
        predictions[predictions == 0] = -1

        misclassified_indices = np.where(predictions != labels)[0]
        current_loss = len(misclassified_indices)
        losses.append(current_loss)

        if current_loss == 0:
            break

        sample_indices = np.random.choice(misclassified_indices, size=min(batch_size, len(misclassified_indices)), replace=False)

        gradient = np.zeros_like(weights)
        gradient_b = 0
        for i in sample_indices:
            gradient += labels[i] * data[i]
            gradient_b += labels[i]

        gradient_norm = np.linalg.norm(gradient)
        if gradient_norm < gradient_norm_threshold:
            break

        weights += lr * gradient
        bias += lr * gradient_b

    print("Learned weights:", weights)
    return losses, weights, bias

_, w, bias = perceptron(X_train, y_train, batch_size=10)
print("WeightS:", w)
print("Bias:", bias)

#@title Ploting function
def plot(data, labels, w, bias):

    plt.scatter(data[:,0], data[:,1], c=labels)

    a, b, c = w[0], w[1], bias

    m = -a / b
    b = -c / b

    x = np.arange(0.2, 0.8, 0.1)
    y = m * x + b

    plt.plot(x, y)

    preds = np.sign(np.dot(data, w)+bias)
    acc = np.count_nonzero(labels == preds) / len(labels)
    plt.title(f"Accuracy on data is {acc}")

    plt.show()



batches = [1,10,50,100]
for bat in batches: 
    _, w, bias = perceptron(X_train, y_train, bat)
    plot(X_train, y_train, w, bias)

_, w, bias = perceptron(X_train, y_train, 1)
plot(X_test, y_test, w, bias)
