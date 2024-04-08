import numpy as np
from numpy import genfromtxt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import cv2
# 1、读取数据
data = np.genfromtxt('machine_learning/job.csv', delimiter=',')
x_data=data[1:,1:-1]
y_data=data[1:,-1]
# 2、模型建立
model=linear_model.LinearRegression()
model.fit(x_data,y_data)
# 3、多项式回归
poly_reg=PolynomialFeatures(degree=3)# degree为多项式的最高阶数

x_poly=poly_reg.fit_transform(x_data)# 特征转换，多项式拟合的自变量要求进行此过程

lin_reg=linear_model.LinearRegression()
lin_reg.fit(x_poly,y_data)
plt.plot(x_data,y_data,'b.')
x_test=np.linspace(1,10,100)
x_test=x_test[:,np.newaxis]
plt.plot(x_test,lin_reg.predict(poly_reg.fit_transform(x_test)),c='r')
plt.show()