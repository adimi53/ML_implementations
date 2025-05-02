#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

df = pd.read_csv('https://sharon.srworkspace.com/ml/datasets/hw3/ENB2012_data.csv')
print(df.head(3))

# Implement here
from sklearn.model_selection import train_test_split
npdf = df.to_numpy()
x = npdf[:, :-1]
y = npdf[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=21)

results = {'Linear': [], 'Ridge': [], 'Lasso': []}
# Implement here Linear Only
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, Ridge, LinearRegression


from sklearn.preprocessing import StandardScaler

corr_df = df.iloc[:, :-2] 
plt.subplots(figsize=(10, 8))
sns.heatmap(corr_df.corr(), annot=True, cmap="RdYlGn")
plt.show()

#scale features,for Ridge,Lasso
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = LinearRegression()
model.fit (x_train_scaled,y_train)
y_train_pred = model.predict(x_train_scaled)
y_test_pred = model.predict(x_test_scaled)
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
results['Linear'].append(mse_train)
results['Linear'].append(mse_test)


######### RIDGE #########

alphas = np.arange(0.1, 1, 0.1)
mses_train = []
mses_test = []

for alpha in alphas:
    model = Ridge(alpha=alpha)
    model.fit(x_train_scaled, y_train)

    y_train_pred = model.predict(x_train_scaled)
    y_test_pred = model.predict(x_test_scaled)

    mses_train.append(mean_squared_error(y_train, y_train_pred))
    mses_test.append(mean_squared_error(y_test, y_test_pred))

best_ridge_train = min(mses_train)
best_ridge_test = mses_test[mses_train.index(best_ridge_train)]
results['Ridge'].append(best_ridge_train)
results['Ridge'].append(best_ridge_test)

plt.figure(figsize=(12,12))

plt.subplot(221)
plt.plot(alphas, mses_train, marker='o')
plt.title("train mse (ridge)")

plt.subplot(222)
plt.plot(alphas, mses_test, marker='o')
plt.title("test mse (ridge)")


######### LASSO #########

alphas = np.logspace(-4, -2, 20)
mses_train = []
mses_test = []

for alpha in alphas:
    model = Lasso(alpha=alpha, max_iter=5000)
    model.fit(x_train_scaled, y_train)

    y_train_pred = model.predict(x_train_scaled)
    y_test_pred = model.predict(x_test_scaled)

    mses_train.append(mean_squared_error(y_train, y_train_pred))
    mses_test.append(mean_squared_error(y_test, y_test_pred))

best_lasso_train = min(mses_train)
best_lasso_test = mses_test[mses_train.index(best_lasso_train)]
results['Lasso'].append(best_lasso_train)
results['Lasso'].append(best_lasso_test)

plt.subplot(223)
plt.plot(alphas, mses_train, marker='o')
plt.title("train mse (lasso)")

plt.subplot(224)
plt.plot(alphas, mses_test, marker='o')
plt.title("test mse (lasso)")

df = pd.DataFrame(results, columns=['Linear', 'Ridge', 'Lasso'], index=['train', 'test'])
print(df)