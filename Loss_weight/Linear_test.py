#coding:utf-8
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
y_ture = pd.read_csv('../data/20180128.csv')['y']
y_pred = pd.read_csv('../result/pred_linear.csv')['y_pred']
# MSE可以评价数据的变化程度，MSE的值越小，说明预测模型描述实验数据具有更好的精确度。
# MAE：平均绝对误差是绝对误差的平均值
# 平均绝对误差能更好地反映预测值误差的实际情况.
print('线性模型mse:{}'.format(mean_squared_error(y_ture,y_pred)))
print('线性模型mae:{}'.format(mean_absolute_error(y_ture,y_pred)))
