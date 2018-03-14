#coding:utf-8 
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
y_ture = pd.read_csv('../data/20180128.csv')['y']
y_pred = pd.read_csv('../result/pred_dnn.csv')['y_pred']

print('深度模型mse:{}'.format(mean_squared_error(y_ture,y_pred)))
print('深度模型mae:{}'.format(mean_absolute_error(y_ture,y_pred)))
