# -*- coding: utf-8 -*-
"""
@author: zgz
"""

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.patches as patches


##################################### part 1 community size and edges VS sale #####################################

res = np.zeros((3000, 300))
with open('../data/community_size_and_edge_vs_sale','r') as file:
    for line in file:
        tem = eval(line)
        size = tem[0][0]
        edge = tem[0][1]
        sale = tem[1]
        res[edge][size] = sale
res = res + 0.1

res = res[:300, :200]
res[res>40000]=0

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

plt.pcolor(res,cmap=plt.cm.hot,norm=mpl.colors.Normalize(vmin=np.min(res),vmax=np.max(res)))

cbar = plt.colorbar()
cbar.ax.tick_params(labelsize = 36)

currentAxis = plt.gca()  # 获取当前子图
rect=patches.Rectangle((95, 0),10,300,linewidth=4,edgecolor='darkorange',facecolor='none',ls='--')
currentAxis.add_patch(rect)


ax=plt.gca()
ax.set_xlim(0,200)
ax.set_ylim(0,300)
ax.set_xlabel(r'Community Size',fontsize=36)
ax.set_ylabel(r'#Edges inside a Community',fontsize=36)
# ax.set_xticks(np.array(range(9))*window_invite)
# ax.set_yticks(np.array(range(9))*window_activity*2)
# ax.set_xticklabels(['0','25','50','75','100','125','150','175','200'])
# ax.set_yticklabels(['0','50','100','150','200','250','300','350','400'])
ax.tick_params(labelsize=36)
plt.tight_layout()
plt.show()