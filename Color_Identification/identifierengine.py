from sklearn.cluster import KMeans

def detectcolors(max_clusters,data):
    inertia_metric = {}
    for clusters in range(1,max_clusters+1):
        kmeans  = KMeans(n_clusters = clusters,init = 'k-means++')
        kmeans.fit(data)
        inertia_metric[clusters] = kmeans.inertia_
    
    return inertia_metric

def bestclusternumber(inertia_metric,threshold):
    len_inertia_metric = len(inertia_metric)
    inertia_change = {}
    for i in range(1,len_inertia_metric):
        inertia_change[i] = abs((inertia_metric[i]-inertia_metric[i+1]))
        if(inertia_change[i]<=threshold):
            return i
    return len_inertia_metric


    
        

