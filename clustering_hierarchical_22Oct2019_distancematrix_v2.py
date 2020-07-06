import csv
import importlib
importlib.import_module('mpl_toolkits.mplot3d').Axes3D
import statistics
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import glob
import plotly
# import plotly.plotly as py
# import plotly.graph_objs as go
import numpy as np
import os
from scipy import stats
import scipy.cluster.hierarchy as hac
#from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import scipy.signal as ss
from math import sqrt
# from dtaidistance import dtw
# from dtaidistance import dtw_visualisation as dtwvis
from fastdtw import fastdtw
#from dtaidistance import clustering
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
import openpyxl
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
from sklearn.cluster import AgglomerativeClustering
from sklearn.utils.validation import check_symmetric
sns.set(color_codes=True)

def my_metric(x, y):
    r = stats.pearsonr(x, y)[0]
    return 1 - r # correlation to distance: range 0 to 2

def euclid_dist(t1,t2):
    return sqrt(sum((t1-t2)**2))

def DTWDistance_fast(s1, s2):
    DTW={}
    w = 60
    w = max(w, abs(len(s1)-len(s2)))
    
    for i in range(-1,len(s1)):
        for j in range(-1,len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0
  
    for i in range(len(s1)):
        for j in range(max(0, i-w), min(len(s2), i+w)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
		
    return sqrt(DTW[len(s1)-1, len(s2)-1])

def my_fastdtw(sales1, sales2):
    return fastdtw(sales1,sales2)[0]

def DTWDistance(s1, s2):
    DTW={}

    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(len(s2)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

    return sqrt(DTW[len(s1)-1, len(s2)-1])


## Cross Correlation between 2 timeseries
def ccf(x, y):
    lag_max = 100
    result = ss.correlate(y - np.mean(y), x - np.mean(x), method='direct') / (np.std(y) * np.std(x) * len(y))
    length = (len(result) - 1) // 2
    lo = length - lag_max
    hi = length + (lag_max + 1)

    return result[lo:hi]

raw_data_all =[]

df_lanedetector = pd.read_csv("lanedetector.csv") 
array_laneDetector = df_lanedetector["laneDetector"].values
print (array_laneDetector)

df = pd.read_csv("clustering_allmondays.csv") 
df = df.replace(r'^\s*$', np.nan, regex=True)
#df = df.replace("", np.nan , inplace=True)
print (df)
#array_laneDetector = array_laneDetector.values
# df1 = df.drop('date_time', axis=1)
# p = df1.values
# #p1 = p.tolist()
# print (p)

df_pairwise_distances = pd.DataFrame(columns=array_laneDetector, index=array_laneDetector)
df_pairwise_elimiatedcounts = pd.DataFrame(columns=array_laneDetector, index=array_laneDetector)
# for each row index (lane detector time series) with each 
for i in range(len(df.index)):
    for j in range(len(df.index)):
        row_t1 = df.iloc[i]
        row_t1 = np.array(row_t1.to_numpy().ravel())
        print (row_t1)
        lanedetector_t1 = row_t1[0]
        print(lanedetector_t1)
        row_t1 = np.delete(row_t1, 0).astype(float)
        row_t2 = df.iloc[j]
        row_t2 = np.array(row_t2.to_numpy().ravel())
        print(row_t2)

        #Get lane detector names
        
        lanedetector_t2 = row_t2[0]
        print(lanedetector_t2)
        row_t2 = np.delete(row_t2, 0).astype(float)

        print (row_t1)
        print (row_t2)

        # get indexex where t1 has nan value
        new_row_t1 = row_t1[~np.isnan(row_t1)]
        new_row_t2 = row_t2[~np.isnan(row_t1)]
        
        # #eliminate t1 nan index values from t1 & t2
        # row_t1 = np.delete(row_t1, nan_indexex_t1)
        print (new_row_t1)
        print (len(new_row_t1))
        # row_t2 = np.delete(row_t2, nan_indexex_t1)
        
        print (new_row_t2)
        print (len(new_row_t2))

        # get indexex where t1 has nan value
        newer_row_t1 = new_row_t1[~np.isnan(new_row_t2)]
        newer_row_t2 = new_row_t2[~np.isnan(new_row_t2)]
        
        # #eliminate t1 nan index values from t1 & t2
        # row_t1 = np.delete(row_t1, nan_indexex_t1)
        print (newer_row_t1)
        print (len(newer_row_t1))
        # row_t2 = np.delete(row_t2, nan_indexex_t1)
        
        print (newer_row_t2)
        print (len(newer_row_t2))

       

        #get lengths of row t1 and row t2 to note how many eliminated
        bins_eliminated = 240*15 - len(newer_row_t1)

        #store distance in the pairwise distance dataframe
        df_pairwise_elimiatedcounts.iloc[i,j] = bins_eliminated

        #normalize row_t1 and row_t2 before taking distance
        znorm_row_t1 =  stats.zscore(newer_row_t1)
        znorm_row_t2 =  stats.zscore(newer_row_t2)

        #calculate distance 
        #dist = euclid_dist(newer_row_t1, newer_row_t2)
        #dist = euclid_dist(znorm_row_t1, znorm_row_t2)
        dist = my_fastdtw(znorm_row_t1, znorm_row_t2)

        #store distance in the pairwise distance dataframe
        if (i==j):
            df_pairwise_distances.iloc[i,j] = 0.0
        else:
            df_pairwise_distances.iloc[i,j] = dist

print (df_pairwise_distances)
print (df_pairwise_distances.isnull().sum().sum())
#df_pairwise_distances.to_csv("dist.csv",index=False)
pairwise_dist_mat = np.array(df_pairwise_distances.values)
#print (pairwise_dist_mat)
#print (pairwise_dist_mat.shape)
#dists = squareform(pairwise_dist_mat)
#print (dists)
# def check_symmetric(a, rtol=1e-05, atol=1e-08):
#     return np.allclose(a, a.T, rtol=rtol, atol=atol)

df_pairwise_distances.fillna(value=np.nan, inplace=True)
print (df_pairwise_distances)
#print (np.around(df_pairwise_distances.values, 3))
#check_symmetric(np.around(df_pairwise_distances.values, 3), rtol=1e-05, atol=1e-08)
path_todir = os.getcwd()
pairwise_dist_mat = df_pairwise_distances.values
#print(np.where(~np.allclose(pairwise_dist_mat, pairwise_dist_mat.T,rtol=1e-05, atol=1e-08)))
pairwise_distmat_repaired = check_symmetric(pairwise_dist_mat)    
print('max error: ', np.amax(np.abs(pairwise_dist_mat - pairwise_dist_mat.T)))        
print('max error repaired: ', np.amax(pairwise_distmat_repaired - pairwise_distmat_repaired.T)) 
print (np.around(pairwise_distmat_repaired))
#t = np.around(pairwise_distmat_repaired)
#pd.DataFrame(t).to_excel("round_dist.xlsx")
# pd.DataFrame(pairwise_distmat_repaired).to_excel(pltname+"_repaired.xlsx") 
# pd.DataFrame(pairwise_dist_mat).to_excel(pltname+".xlsx") 
#df_pairwise_distances.to_excel(pltname+".xlsx")
#os.system("pause")


plt.figure(figsize=(25,25)) 
sns.heatmap(df_pairwise_distances, cmap='Blues', linewidth=1)
#plt.show()
plt.savefig('allmondays_distancematrix.png')
#plt.savefig("distancematrix_euclid_normalized_"+pltname+".png") 


dists = squareform(pairwise_distmat_repaired)
#dist_cond = squareform(pairwise_distmat_repaired)
# print (dists)
n_clusters = 11
Z = hac.linkage(dists, 'average')
labels = hac.fcluster(Z, n_clusters, criterion='maxclust')
print (labels)

##Plot dendogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram - Average Linkage')
plt.xlabel('sample index')
plt.ylabel('distance')
hac.dendrogram(
    Z,
    labels = array_laneDetector,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.  # font size for the x axis labels
)

#pltname = fname[-9:-4]
#plt.show()
plt.savefig('allmondaytrend_dendogram.png')

# g = sns.clustermap(pp1, method='single', metric=my_fastdtw)
# plt.savefig(fname+'clustermap.png')

# model1 = clustering.Hierarchical(dtw.distance_matrix_fast, {})
# cluster_idx = model1.fit(p1)
# # Augment Hierarchical object to keep track of the full tree
# model2 = clustering.HierarchicalTree(model1)
# cluster_idx = model2.fit(series)
# model2.plot(fname+'.png')

# q = p.tolist()
# raw_data_all.append(q)

# print (raw_data_all)


# ## Hierarchical clustering 
# # Do the clustering


