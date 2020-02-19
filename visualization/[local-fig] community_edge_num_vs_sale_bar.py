# -*- coding: utf-8 -*-
"""
@author: zgz
"""

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt


truncated = 30
ceil = 10000

res = [[] for _ in range(truncated)]
with open('../data/community_size_and_edge_vs_sale','r') as file:
    for line in file:
        tem = eval(line)
        size = tem[0][0]
        if size>50 and size<=80:
            edge = tem[0][1]
            if edge<truncated:
                sale = tem[1]
                res[edge].append(sale)

res = [np.mean(u) for u in res]

x = [i+100 for i in range(truncated)]
y = res

# 调整x，y轴的label距离坐标轴的距离
mpl.rcParams['xtick.major.pad'] = 10
# 调整字体为type 1 font
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
# 作图尺寸
plt.figure(figsize=(12,9))
# 默认的像素：[6.0,4.0]，分辨率为100，图片尺寸为 600&400
plt.rcParams['savefig.dpi'] = 100 #图片像素
plt.rcParams['figure.dpi'] = 100 #分辨率
# 调整坐标轴边框的粗细
plt.rcParams['axes.linewidth'] = 2
# plt.grid()

plt.bar(x, y, edgecolor=['black']*truncated, color=r'darkorange', linewidth=2, label=r'cdf')

# plt.rc('text',usetex=True)
# plt.rc('font',family='serif',serif='serif')

ax=plt.gca()
# ax.set_xlabel(r'\textbf{Spatial Error/Km}',fontsize=28)
# ax.set_ylabel(r'\textbf{CDF of Check-ins/\%}',fontsize=28)
ax.set_xlabel(r'#Edge inside a Community',fontsize=36)
ax.set_ylabel(r'Agent Value',fontsize=36)
# 调整坐标轴取值范围
ax.set_xlim(99.5, 129.5)
# ax.set_ylim(0,1.05)
# 坐标轴的坐标文字大小
ax.tick_params(labelsize=36)
# ax.semilogx()
# ax.semilogy()
#plt.xticks([0.1, 1, 10],
#          [r'\boldmath $10^{-1}$', r'\boldmath $10^{0}$', r'\boldmath $10^{1}$'])
#plt.yticks([0, 0.25, 0.5, 0.75, 1],
#          [r'\boldmath $0.00$', r'\boldmath $0.25$', r'\boldmath $0.50$', r'\boldmath $0.75$', r'\boldmath $1.00$'])
plt.tight_layout()
plt.show()