from Data_loader import Data_loader
import csv
import numpy as np

Dataloader = Data_loader()
_,_,_,edgelist,_,_,_ = Dataloader.laod()
with open('edgelist.csv', 'w', newline='') as csvfile:
    writer  = csv.writer(csvfile,delimiter=' ')
    for row in edgelist:
        writer.writerow(row)

'''
deepwalk --input /data/mas/yuanyuan/community_value_prediction/edgelist.csv --representation-size 20 --workers 3 --output /data/mas/yuanyuan/community_value_prediction/node20.embeddings
'''

# feature+embedding
d = np.load('sample2_dataset_unc_norm.npy',allow_pickle=True)
feature = d[0]
fp = open('node.embeddings')
content = fp.readlines()
data = content[1:]
for index,x in enumerate(data):
    data[index] = x[:-1].split(' ')
    data[index] = list(map(float,data[index]))
    data[index] = [int(data[index][0]),data[index][1:]]
result = feature
for i in data:
    result[i[0]] += i[1]
result = np.array(result)
np.save('/data/mas/yuanyuan/community_value_prediction/feature_embedding.npy',result)