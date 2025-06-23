import os
import pandas as pd
import pickle
import gzip
import pickle as cPickle
import numpy as np
import random
import glob
from collections import Counter
from sklearn import svm
from sklearn.svm import OneClassSVM
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score

from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tslearn.clustering import KShape
import umap

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import auc, precision_recall_curve
from sklearn.preprocessing import MinMaxScaler

same = []
opp = []
sameb = []
oppb = []

t_files = glob.glob("*.pkl")

for tfile in t_files:
  print(tfile)
  with open(tfile, 'rb') as file:
    windows = pickle.load(file)
  if tfile == ('chr21_Control.pkl'):
    keys = list(windows.keys())
    keys_sample = min(len(keys), 700)
    sel_keys = random.sample(keys, keys_sample)
    for key in sel_keys:
      sameb.append(windows[key]['forward_ipd'])
      oppb.append(windows[key]['reverse_ipd'])
  else:
    keys = list(windows.keys())
    keys_sample = min(len(keys), 100)
    sel_keys = random.sample(keys, keys_sample)
    for key in sel_keys:
      same_data = windows[key]['forward_ipd']
      opp_data = windows[key]['reverse_ipd']
      if len(same_data) > 100:
        same_data = same_data[:100]
      if len(opp_data) > 100:
        opp_data = opp_data[:100]
      same.append(same_data)
      opp.append(opp_data)

#non-b
same_df = pd.DataFrame(same)
opp_df = pd.DataFrame(opp)
combined = np.concatenate([same_df, opp_df], axis=1)

#b-dna
sameb_df = pd.DataFrame(sameb)
sameb_df.shape
oppb_df = pd.DataFrame(oppb)
comb_b = pd.concat([sameb_df, oppb_df], axis=1)
comb_b = pd.DataFrame(comb_b)

#combined
sample = np.concatenate([comb_b, combined], axis=0)
sample_df = pd.DataFrame(sample)
sample_df = sample_df.dropna()

#silhouette method to find k
ss = []
for i in range(2,6):
  kshape = KShape(n_clusters=i, random_state=42)
  label = kshape.fit_predict(sample_df)
  silhouette_avg = silhouette_score(sample_df, label)
  ss.append(silhouette_avg)

#kshape clustering
kshape = KShape(n_clusters =2, random_state =42)
label1 = kshape.fit_predict(sample_df)
sample_df['cluster']=label1
cluster_counts = sample_df['cluster'].value_counts().reset_index()
cluster_counts.columns = ['cluster_s', 'count']
silhouette = silhouette_score(sample_df, label1)

#kmeans clustering
k=2
sample_df.columns = sample_df.columns.astype(str)
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(sample_df)
label2 = kmeans.labels_
sample_df['cluster_m']=label2
clust_count = sample_df['cluster_m'].value_counts().reset_index()
clust_count.columns = ['cluster_m', 'count']
silhouette = silhouette_score(sample_df, label2)

#UMAP
X= sample_df.drop(columns = ['cluster'])
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embedding = reducer.fit_transform(X)
umap_df = pd.DataFrame(embedding, columns = ['umap1', 'umap2'])
umap_df['Cluster'] = label1

plt.figure(figsize=(10,8))
plt.scatter(umap_df['umap1'], umap_df['umap2'], c=umap_df['Cluster'], cmap='viridis', s=10)
plt.title('UMAP Clusters')
plt.colorbar()
plt.show()

#hyperparams for one class svm
X_train_p, X_test_p= train_test_split(comb_b, test_size=0.2, random_state=42)

gamma_grid = {'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
model = OneClassSVM(kernel = 'rbf')

grid_search = GridSearchCV(model, gamma_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train_p)

opt_gamma = grid_search.best_params_['gamma']
final_model = OneClassSVM(kernel='rbf', gamma =opt_gamma)
final_model.fit(X_train_p)

#testing gamma hyperparameter
gamma_val = [0.001, 0.01, 0.1, 1, 10, 100]
auc_prs = []
for gamma in gamma_val:
  model=OneClassSVM(kernel='rbf', gamma=gamma)
  model.fit(X_train_p)
  y_score = model.decision_function(X_test_p)
  precision, recall, _ = precision_recall_curve(np.ones_like(y_score),y_score)
  auc_pr = auc(recall, precision)
  auc_prs.append(auc_pr)

#one class svm
X_train, X_test= train_test_split(comb_b, test_size=0.2, random_state=42)
x = np.concatenate([X_test, combined], axis=0)
test = pd.DataFrame(x)

model = svm.OneClassSVM(kernel = 'rbf', nu=0.1, gamma=0.001)
model.fit(X_train)
y_pred = model.predict(test)
y_svm = [1 if label ==-1 else 0 for label in y_pred]

label_counts = Counter(y_svm)

X_train['y'] = 0
test['y'] = [1 if label == -1 else 0 for label in y_pred]

X_train = pd.DataFrame(X_train)
test = pd.DataFrame(test)

class_df = np.concatenate([X_train, test], axis=0)
class_df = pd.DataFrame(class_df)
names = list(class_df.columns)
names[-1] = 'y'
class_df.columns = names
label_counts = class_df['y'].value_counts()

#classification - logistic regression model
x = class_df.drop('y', axis=1)
y = class_df['y']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
logreg = LogisticRegression(max_iter=2000)
logreg.fit(x, y)
y_pred = logreg.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

#graphing evaluation metrics for different gamma values
hyperparams = ['0.001', '0.01', '0.1', '1', '10', '100']
precision_0 = [0.99,0.99, 0.78, 0.64, 0.64, 0.64 ]
precision_1 = [0.96,0.93, 0.78, 0.74, 0.74, 0.74]
recall_0 = [1.00, 0.99, 0.9, 0.54, 0.54, 0.54]
recall_1 = [0.89, 0.93, 0.57, 0.81, 0.81, 0.81]
f1_0 = [0.99, 0.99, 0.84, 0.59, 0.59, 0.59]
f1_1 = [0.93, 0.93, 0.66, 0.77, 0.77, 0.77]
accuracy = [0.9857, 0.9857, 0.7786,0.7071, 0.7071, 0.7071 ]
x = np.arange(len(hyperparams))
fig, ax = plt.subplots(figsize=(10,6))

ax.plot(x, precision_1, marker='o', label = 'Precision (Anomaly)')
ax.plot(x, recall_1, marker='o', label = 'Recall (Anomaly)')
ax.plot(x, f1_1, marker='o', label = 'F1-score (Anomaly)')

ax.set_xticks(x)
ax.set_xticklabels(hyperparams)
ax.set_xlabel('Gamma Value')
ax.set_ylabel('Score')
ax.set_title('Evaluation Metrics for Different Gamma Values')
ax.legend()

plt.grid(True)
plt.tight_layout()
plt.show()
