# 北京交通大学
# 17271059 白云飞
# https://plotly.com/python/subplots/ 官方文档
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('TkAgg')
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
data = pd.read_csv("kpi_0129.csv")
tdata = data['mrt']
rd = sm.tsa.seasonal_decompose(tdata.values, period=11)
length = len(tdata)
print(rd.seasonal, rd.trend, rd.resid)
# shared_xaxes 就可以实现同时控制多个子图的功能

fig = make_subplots(rows=4, cols=1, shared_xaxes=True)
fig.add_trace(go.Scatter(x=np.arange(0, length), y=tdata, mode='lines', name='原始数据', marker={"line": {"color": "blue"}}), row=1, col=1)
fig.add_trace(go.Scatter(x=np.arange(0, length), y=rd.seasonal, mode='lines', name='季节项', marker={"line": {"color": "red"}}), row=2, col=1)
fig.add_trace(go.Scatter(x=np.arange(0, length), y=rd.trend, mode='lines', name='趋势项', marker={"line": {"color": "green"}}), row=3, col=1)
fig.add_trace(go.Scatter(x=np.arange(0, length), y=rd.resid, mode='lines', name='残差项', marker={"line": {"color": "black"}}), row=4, col=1)
fig_html = "./" + 'ssss' + ".html"
pio.write_html(fig, file=fig_html)
fig.show()
