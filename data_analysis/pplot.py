#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 19:44:04 2020

@author: yuanyuan
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind,norm,f
import numpy as np
data = pd.read_csv('result_agent_community.csv',index_col=0).reset_index()
#data = data[data['node_a']>2]

def Describe(data,columns):
    des = data[columns].describe(percentiles=[0.25*(i+1) for i in range(3)])
    l = [float(des['25%']),float(des['50%']),float(des['75%'])]
    return l

def node_divide(data,index,l,columns):
    if index==1:
        return data[(data[columns]<=l[0])]
    elif index==2:
        return data[(data[columns]>l[0]) & (data[columns]<=l[1])]
    elif index==3:
        return data[(data[columns]>l[1]) & (data[columns]<=l[2])]
    elif index==4:
        return data[(data[columns]>l[2])]
    else:
        return data

net = ['tri_a', 'cen_a', 'cluster_a', 'node_a',
       'edge_a','tri_c', 'cen_c', 'cluster_c', 'node_c', 'edge_c']
l_node = Describe(data,'node_c')
x = []
x_std = []
x_t_test = []
for j in range(1):
    print('node_%d' % (j+4))
    for index,n in enumerate(net):
        print(n)
        data_n = node_divide(data,j+4,l_node,'node_c')
        temp = []
        temp_t_test = []
        std = []
        l_sale = Describe(data_n,'sales')
        for i in range(4):
            data_t = node_divide(data_n,i+1,l_sale,'sales')
            t_test_t = data_t[n].tolist()
            temp_t_test.append(t_test_t)
            temp.append(data_t[n].mean())
            std.append(data_t[n].std())
        print(temp)
        x.append(temp)
        x_t_test.append(temp_t_test)
        x_std.append(std)
name_list = ['<Q1','Q1~Q2','Q2~Q3','>Q3']

def pplot(num_list,y,xlabel,ylabel): 
    plt.bar(range(len(num_list)),y,tick_label=name_list,width = 0.5)   
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()
pplot(name_list,x[0],'sales','community_cluster_coef')


def ftest(s1,s2):
    '''F检验样本总体方差是否相等'''
    print("Null Hypothesis:var(s1)=var(s2)，α=0.05")
    F = np.var(s1)/np.var(s2)
    v1 = len(s1) - 1
    v2 = len(s2) - 1
    p_val = 1 - 2*abs(0.5-f.cdf(F,v1,v2))
    print(p_val)
    if p_val < 0.05:
        print("Reject the Null Hypothesis.")
        equal_var=False
    else:
        print("Accept the Null Hypothesis.")
        equal_var=True
    return equal_var
	 	
def ttest_ind_fun(s1,s2):
    '''t检验独立样本所代表的两个总体均值是否存在差异'''
    equal_var = ftest(s1,s2)
    print("Null Hypothesis:mean(s1)=mean(s2)，α=0.05")
    ttest,pval = ttest_ind(s1,s2,equal_var=equal_var)
    print(pval)
    if pval < 0.05:
        print("Reject the Null Hypothesis.")
    else:
        print("Accept the Null Hypothesis.")
    return pval

np.random.seed(42)
s1 = norm.rvs(loc=1,scale=1.0,size=20)
s2 = norm.rvs(loc=1.5,scale=0.5,size=20)
s3 = norm.rvs(loc=1.5,scale=0.5,size=25)

ttest_ind_fun(x_t_test[2][0],x_t_test[2][2])

