#Wais Patrick Assignment 4

#Test branch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sci

# Read in File and save number of cluster, and data to seperate variables for better handling
#---------------------------------------------------------------------------------------------------
df = pd.read_csv('input.csv',delimiter=';',header=None, decimal=",")
number_cluster = df.loc[0,0]
elements = df.loc[1,0]

array = df.to_numpy()
X = array[2:20,:]
num_elem = len(X)+1
#---------------------------------------------------------------------------------------------------

# Perform scatter plot of data array X
#---------------------------------------------------------------------------------------------------
labels = range(1,num_elem)
plt.figure(figsize=(11,8))
plt.subplots_adjust(bottom=0.1)
plt.scatter(X[:,0],X[:,1],label='Data points')
plt.title('Scatter Plot')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')


for label, x, y in zip(labels, X[:, 0], X[:, 1]):
    l1 = plt.annotate(
        label,
        xy=(x,y),xytext=(-3,3),
        textcoords='offset points', ha='right', va='bottom')


plt.legend(loc=1, shadow=bool)
#plt.show()
#---------------------------------------------------------------------------------------------------

# Calculate the cluster for each xy coordinate pair in the array X by using sklearn library
#---------------------------------------------------------------------------------------------------
from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=int(number_cluster),affinity='euclidean',linkage='ward')
cluster.fit_predict(X)
print(cluster.labels_)
#---------------------------------------------------------------------------------------------------

# Plot dendogram of data array X
#---------------------------------------------------------------------------------------------------
linked = sci.linkage(X,'single')
labelList = range(1,num_elem)
plt.figure(figsize=(10,7))
plt.title('Dendogram')
plt.xlabel('Number of data point')
plt.ylabel('Distance')

sci.dendrogram(linked,
               orientation='top',
               labels=labelList,
               distance_sort='descending',
               show_leaf_counts=True)

plt.show()
#---------------------------------------------------------------------------------------------------








