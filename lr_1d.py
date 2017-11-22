import numpy as np
import matplotlib.pyplot as plt


#load data
X=[]
Y=[]

for line in open('data_1d.csv'):
    x,y = line.split(',')
    X.append(float(x))
    Y.append(float(y))


#turn into numpy arrays
X = np.array(X)
Y = np.array(Y)

#plot data
plt.scatter(X,Y)
plt.show()


#apply equations we learned to calculate a and b
denominator = X.dot(X) - X.mean() * X.sum()
a = (X.dot(Y) - Y.mean()*X.sum())/denominator
b = (Y.mean()*X.dot(X) - X.mean()* X.dot(Y))/ denominator

#calculate predicted y
yhat = a*X + b

plt.scatter(X, Y)
plt.plot(X, yhat)
plt.show()

#calculate r-squared
d1 = Y -yhat
d2 = Y- Y.mean()
r2 = 1- d1.dot(d1)/d2.dot(d2)
print(" the r squared is:", r2)