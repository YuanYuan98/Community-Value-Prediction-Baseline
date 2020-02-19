# -*- coding: utf-8 -*-
"""
@author: zgz
"""

'''
采样得到目标数据集
'''

import sys
import re
from pyspark import SparkContext
from pyspark.sql import SparkSession
from operator import add
import random
import pickle
from operator import add
from pyspark.sql.types import *

#调整系统编码为UTF-8
reload(sys)
sys.setdefaultencoding('utf-8')

spark = SparkSession\
        .builder\
        .appName("yy: data_preparation") \
        .enableHiveSupport() \
        .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel('ERROR')


user_community_partition = sc.pickleFile('hdfs://realtime-bigdata/qinghua/zhangguozhen/proj_customer_value/user_community_partition')

'''
    字段：uid(str), community_id(str)
	说明：
	1）community_id即agent_uid
	2）存在一个节点属于多个社群（e.g. ['123','2'], ['123',3']）;
	3）店主节点也属于自己的社群
'''

# community_memberlist = sc.pickleFile('hdfs://realtime-bigdata/qinghua/zhangguozhen/proj_customer_value/community_memberlist')
# '''
#     字段：agent(str), community_memberlist(list)
# '''

label = sc.pickleFile('hdfs://realtime-bigdata/qinghua/zhangguozhen/proj_customer_value/labels')
'''
    字段：agent_uid(str), 销售量(int), 销售额(int)
	说明：销售额是店主的价值
'''

edges = sc.pickleFile('hdfs://realtime-bigdata/qinghua/zhangguozhen/proj_customer_value/edges')
'''
    字段：uid(str), neighbor_uid(list)
	说明：无
'''

influencer_info = sc.pickleFile('hdfs://realtime-bigdata/qinghua/zhangguozhen/proj_customer_value/influencer_info')
'''
    字段：agent_uid(str), Community_size(int), edges_inside_community(int), agent_sale(int)
	说明：agent_sale不能直接用，需要用labels。
'''

influencer_info = influencer_info.filter(lambda x:x[1]>=40 and x[1]<=60).map(lambda x:(x[0],[x[1],x[2]]))

agent_sample = influencer_info.map(lambda x:x[0]).collect()
broadcast_agent = {}
for u in agent_sample:
    broadcast_agent[u]=1
broadcast_agent = sc.broadcast(broadcast_agent)

label = label.filter(lambda x:x[0] in broadcast_agent.value).map(lambda x:(x[0],[x[1],x[2]]))

influencer_info = influencer_info.join(label).map(lambda x:(x[0],x[1][0][0],x[1][0][1],x[1][1][0],x[1][1][1]))

# agent_uid(str), Community_size(int), edges_inside_community(int), 销售量(int), 销售额(int)


user_community_partition = user_community_partition.filter(lambda x:x[1] in broadcast_agent.value)

user_sample = list(set(user_community_partition.map(lambda x:x[0]).collect()))
broadcast_user={}
for u in user_sample:
    broadcast_user[u]=1
broadcast_user = sc.broadcast(broadcast_user)

user_community_list = user_community_partition.groupByKey().mapValues(list)

edges = edges.filter(lambda x:x[0] in broadcast_user.value)

edges = edges.flatMapValues(lambda x:x)

def edge_f(a,b):
    return (min(a,b),max(a,b))

edges = edges.filter(lambda x: x[1] in broadcast_user.value)
#edges = edges.map(lambda x:edge_f(x[0],x[1])).distinct()

edges1 = edges
edges2 = edges.map(lambda x:(x[1],x[0]))

edges1 = edges1.join(user_community_list).map(lambda x:((x[0],x[1][0]),x[1][1]))
edges2 = edges2.join(user_community_list).map(lambda x:((x[1][0],x[0]),x[1][1]))

def diff(x):
    a = set(x[0])
    b = set(x[1])
    return list(a.difference(b).union(b.difference(a)))

edges_c = edges1.join(edges2).map(lambda x:(1,diff(x[1]))).flatMapValues(lambda x:x)
edges_c = edges_c.map(lambda x:(x[1],x[0])).reduceByKey(add)

schema_edge = StructType([
    StructField("agent", StringType(), True),
    StructField("edges_outside_community", IntegerType(), True)])

edges_c = spark.createDataFrame(edges_c, schema_edge)

schema = StructType([
    StructField("agent", StringType(), True),
    StructField("community_size", IntegerType(), True),
    StructField("edges_inside_community", IntegerType(), True),
    StructField("sales_num", IntegerType(), True),
    StructField("sales_money", IntegerType(), True)])

influencer_info = spark.createDataFrame(influencer_info, schema)

influencer_info = influencer_info.join(edges_c,['agent'],'left').na.fill(0)

influencer_info = influencer_info.rdd.map(lambda x:(x.agent,x.community_size,x.edges_inside_community,x.sales_num,x.sales_money,int(x.edges_outside_community/2.0)))

df1,df2 = influencer_info.randomSplit([0.5,0.5],seed=1)

# df1.write.csv('hdfs://realtime-bigdata/qinghua/zhangguozhen/proj_customer_value/csvYY/community_info1',sep='\t')
# df2.write.csv('hdfs://realtime-bigdata/qinghua/zhangguozhen/proj_customer_value/csvYY/community_info2',sep='\t')

print(df1.collect())
# print(df2.collect())

sc.stop()
