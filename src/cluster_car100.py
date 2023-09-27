# This script computes induced sub-contexts of the predicate scales
# using the cluster centers of k-medoid clustering
import pandas as pd
import plotly.express as px
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_samples, silhouette_score

# Scales the inter-ordinal and inter-ordinal tree scale using clustering to hopefully readable size

data = pd.read_csv("../data/rank_car.csv")
data_class = data.pop("class")
data = data[[c for c in data.columns if c != "class"]]

inter_scale = pd.read_csv("../output/rank_car100/RF/rank_car_interordinal-scale.csv",index_col=0)
inter_tree_scale = pd.read_csv("../output/rank_car100/RF/rank_car_product-scale.csv",index_col=0)

# Optics and dbscan do no work, since there is a sequence of all points each with distance 1

range_n_clusters = range(2,51)
silhouette_score_values = []
for n_clusters in range_n_clusters:
    clusterer = KMedoids(n_clusters=n_clusters, random_state=10,metric='hamming',init="random",method="pam")
    cluster_labels = clusterer.fit_predict(data)
    silhouette_avg = silhouette_score(data, cluster_labels,metric="hamming")
    silhouette_score_values.append(silhouette_avg)

results = pd.DataFrame(zip(range_n_clusters,silhouette_score_values),columns=["n_clusters","silhouette_coefficient"])
fig = px.scatter(results,x="n_clusters",y="silhouette_coefficient")
fig.show()
fig.write_html("../output/rank_car100/medoids_scores.html")
# 19 was best
clusterer = KMedoids(n_clusters=19, random_state=10,metric='hamming',init="random",method="pam")
clusterer.fit(data)
data.loc[clusterer.medoid_indices_].to_csv("../output/rank_car100/medoids.csv")
# 9 was ok
clusterer = KMedoids(n_clusters=9, random_state=10,metric='hamming',init="random",method="pam")
clusterer.fit(data)
data.loc[clusterer.medoid_indices_].to_csv("../output/rank_car100/medoids9.csv")
