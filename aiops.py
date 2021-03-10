import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM # 导入OneClassSVM
from mpl_toolkits.mplot3d import Axes3D # 导入3D样式库
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
matplotlib.use('TkAgg')

# 数据定义
data1 = pd.read_csv("0304-St1.csv")
m_data = data1['mrt']
sr_data = data1['sr']
length = len(m_data)
tmp_data = data1['timestamp']
x_data = range(len(m_data))
print(np.mean(m_data), np.var(m_data), np.std(m_data))

#  折线图
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=np.arange(0, length), y=m_data, mode='lines', name='平均响应时间', marker={"line": {"color": "red"}}))
fig1.show()
fig_html = "./" + '0304-St1' + ".html"
pio.write_html(fig1, file=fig_html)

'''
#  四分位
def sfw(lis):
    if len(lis) % 2 == 0:
        mid = float((lis[len(lis) / 2] + lis[len(lis) / 2 - 1])) / 2
    else:
        mid = lis[len(lis) / 2]
    q1 = 1 + (float(len(lis)) - 1) * 1 / 4
    q3 = 1 + (float(len(lis)) - 1) * 3 / 4
    q4 = q3 + 1.5 * (q3 - q1)
    q5 = q1 - 1.5 * (q3 - q1)
    for i in range(len(lis)):
        if lis[i] > 0.5 * q3+1.5 * (q3-q1) or lis[i] < 0.5 * q1 - 1.5 * (q3 - q1):
            print(i)
# 检测异常
def anomaly(lis, t2):
    for i in range(len(t2)):
        if t2[i] < 100:
            print(i）
'''
