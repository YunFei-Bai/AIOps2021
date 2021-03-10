import numpy as np
import csv
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

data = pd.read_csv("kpi_0301.csv")
tc_data = data['tc']
sr_data = data['sr']
m_data = data['mrt']
tmp_data = data['timestamp']
t = []  # 存储不同服务器的数据
s = []  # 存储数据索引
w = ["timestamp", "rr", "sr", "count", "mrt", "tc"]

def num_ST(N_st):
    a = data[data.tc == N_st].index.tolist()
    s.append(a)
    b = []
    for i in range(len(a)):
        b.append(m_data[a[i]])
    print(np.mean(b), np.var(b), np.std(b))
    t.append(b)
    return b
num_ST('388b83e4b434a13b679b0bad237fb5fabe1a0c54')
num_ST('f952e21004726cf7d2b2c5aa400a566e2b2a6aae')
num_ST('aaa51527e6254b04dd1d25004e6da3e6db651e4e')
'''
fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
fig.add_trace(go.Scatter(x=np.arange(0, len(t[0])), y=t[0], mode='lines', name='ST1', marker={"line": {"color": "blue"}}), row=1, col=1)
fig.add_trace(go.Scatter(x=np.arange(0, len(t[1])), y=t[1], mode='lines', name='ST2', marker={"line": {"color": "blue"}}), row=2, col=1)
fig.add_trace(go.Scatter(x=np.arange(0, len(t[2])), y=t[2], mode='lines', name='ST3', marker={"line": {"color": "blue"}}), row=3, col=1)
fig_html = "./" + 'A_ST_NUM_0301' + ".html"
pio.write_html(fig, file=fig_html)
fig.show()
'''