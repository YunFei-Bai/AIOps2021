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
data = pd.read_csv("kpi_0304.csv")
data1 = int(data['timestamp'])
for i in range(len(data1)-1):
    if data[i+1] - data[i] > 60:
        print(i)

