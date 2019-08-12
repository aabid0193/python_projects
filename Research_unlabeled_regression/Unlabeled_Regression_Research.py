import numpy as np
from sklearn import datasets, linear_model

b_1 = np.random.randn(5,1)
b_2 = np.random.randn(5,1)
iteration = 500
data_y = np.random.random_integers(500, size=(300, 1))
data_x = np.random.random_integers(500, size=(300, 5))

def em(b_1,b_2,iteration, data_y, data_x):
  for t in range(1, iteration+1):
    J_1=[]
    J_2=[]
    for i in range(len(data_y)):
      if abs(data_y[i]-(np.dot(data_x[i],b_1))) < abs(data_y[i]-(np.dot(data_x[i],b_2))):
        J_1.append(i)
      else:
        J_2.append(i)
  return J_1, J_2

J_1, J_2 = em(b_1, b_2, iteration, data_y, data_x)
#y_a = data_y[J_1] #set(J_1).intersection(set(data_y))
#x_a = data_x[J_2] #set(J_2).intersection(set(data_x))

len(J_1), len(J_2)

lr1=linear_model.LinearRegression()
L1 = lr1.fit(data_x[J_1], data_y[J_1])

L.coef_

lr2 = linear_model.LinearRegression()
L2 = lr2.fit(data_x[J_2], data_y[J_2])
L2.coef_
