import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# 数据定义

data = pd.read_csv("kpi_0303.csv")
tc_data = data['tc']
sr_data = data['sr']
m_data = data['mrt']
tmp_data = data['timestamp']
t = []  # 存储不同的数据
s = []  # 存储数据索引
w = []

def num_ST(N_st):
    a = data[data.tc == N_st].index.tolist()
    s.append(a)
    tmp = []
    tmp.append(tmp_data[a])
    b = []
    for i in range(len(a)):
        b.append(m_data[a[i]])
    print(np.mean(b), np.var(b), np.std(b))
    print(tmp)
    t.append(b)
    return b, tmp
num_ST('ServiceTest1')
num_ST('ServiceTest2')
num_ST('ServiceTest3')
num_ST('ServiceTest4')
num_ST('ServiceTest5')
num_ST('ServiceTest6')
num_ST('ServiceTest7')
num_ST('ServiceTest8')
num_ST('ServiceTest9')
num_ST('ServiceTest10')
num_ST('ServiceTest11')

fig = make_subplots(rows=11, cols=1, shared_xaxes=True)
fig.add_trace(go.Scatter(x=np.arange(0, len(t[0])), y=t[0], mode='lines', name='ST1', marker={"line": {"color": "blue"}}), row=1, col=1)
fig.add_trace(go.Scatter(x=np.arange(0, len(t[1])), y=t[1], mode='lines', name='ST2', marker={"line": {"color": "blue"}}), row=2, col=1)
fig.add_trace(go.Scatter(x=np.arange(0, len(t[2])), y=t[2], mode='lines', name='ST3', marker={"line": {"color": "blue"}}), row=3, col=1)
fig.add_trace(go.Scatter(x=np.arange(0, len(t[3])), y=t[3], mode='lines', name='ST4', marker={"line": {"color": "blue"}}), row=4, col=1)
fig.add_trace(go.Scatter(x=np.arange(0, len(t[4])), y=t[4], mode='lines', name='ST5', marker={"line": {"color": "blue"}}), row=5, col=1)
fig.add_trace(go.Scatter(x=np.arange(0, len(t[5])), y=t[5], mode='lines', name='ST6', marker={"line": {"color": "blue"}}), row=6, col=1)
fig.add_trace(go.Scatter(x=np.arange(0, len(t[6])), y=t[6], mode='lines', name='ST7', marker={"line": {"color": "blue"}}), row=7, col=1)
fig.add_trace(go.Scatter(x=np.arange(0, len(t[7])), y=t[7], mode='lines', name='ST8', marker={"line": {"color": "blue"}}), row=8, col=1)
fig.add_trace(go.Scatter(x=np.arange(0, len(t[8])), y=t[8], mode='lines', name='ST9', marker={"line": {"color": "blue"}}), row=9, col=1)
fig.add_trace(go.Scatter(x=np.arange(0, len(t[9])), y=t[9], mode='lines', name='ST10', marker={"line": {"color": "blue"}}), row=10, col=1)
fig.add_trace(go.Scatter(x=np.arange(0, len(t[10])), y=t[10], mode='lines', name='ST11', marker={"line": {"color": "blue"}}), row=11, col=1)
fig_html = "./" + 'ST_NUM_0303' + ".html"
pio.write_html(fig, file=fig_html)
fig.show()

'''
def ano(num, lis):

    lis = np.array(lis)
    p1 = 25
    p2 = 50
    p3 = 75
    q1 = np.percentile(lis, p1)
    q2 = np.percentile(lis, p2)
    q3 = np.percentile(lis, p3)
    Low = p1 - 1.5 * (p3 - p1)
    Up = p3 + 1.5 * (p3 - p1)

    for i in range(len(lis)):
        if sr_data[s[num-1][i]] < 100:
            print(s[num-1][i])
        elif t[num-1][i] > 1000:
            print(s[num-1][i])

ano(1, t[0])
'''

