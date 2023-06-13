import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

np.random.seed(874)

x = np.r_[np.random.normal(size=1000,loc=0,scale=1),
          np.random.normal(size=1000,loc=4,scale=1)]


y = np.r_[np.random.normal(size=1000,loc=10,scale=1),
          np.random.normal(size=1000,loc=10,scale=1)]

data = np.array([x,y])


"""
p = plt.subplot()
p.scatter(data[:,0],data[:,1],c="black",alpha=0.5)
p.set_aspect("equal")
plt.show()
"""
model= KMeans(2,init=centers)
fit2 = model.fit(data.T)
print(fit2.cluster_centers_)
fig, ax = plt.subplots(1,2,dpi=140,figsize=(8,4),sharex=True,sharey=True)
ax[0].set(xlabel="x",ylabel="y")
ax[0].scatter(x,y,c=fit2.labels_,marker='o',alpha=0.5,s=55,linewidths=.1,edgecolor="k",cmap="turbo")
ax[0].scatter(fit2.cluster_centers_[:,0],fit2.cluster_centers_[:,1],s=60, marker='D',c='pink')
ax[0].set_title("n_clusters=5")
#plt.savefig("kmeans_1_2.png",dpi=100)
plt.show()

"""
max_iter = 200
n_clusters = 2
clusters = np.random.randint(0,n_clusters,data.shape[0])

for _ in range(max_iter):
    centroids = np.array([data[clusters == n,:].mean(axis=0) for n in range(n_clusters)])
    new_clusters = np.array([np.linalg.norm(data-c,axis=1) for c in centroids]).argmin(axis = 0)

    for n in range(n_clusters):
        if not np.any(new_clusters == n):
            centroids[n] = data[np.random.choice(data.shape[0],1),:]

    if np.allclose(clusters,new_clusters):
        break
    clusters = new_clusters

p = plt.subplot()
p.scatter(data[clusters==0,0],data[clusters==0,1],c="red")
p.scatter(data[clusters==1,0],data[clusters==1,1],c="white",edgecolors="black")
p.scatter(centroids[:,0],centroids[:,1],c="orange",marker="s")

p.set_aspect("equal")
plt.show()

"""
n_clusters = 2



class Files:
    likes = None
    labels = None
    vectors = None # np.array([x1,x2,x3,...]) x1: np.array

class InstaGramFlies:
    n_iters = None
    n_clusters = None
    n_flies = None
    centers = None
    files = None #np.array(Files)


    def InstaGramFlies(self,n_iters,n_clusters,n_flies):
        self.n_iters = n_iters
        self.n_clusters = n_clusters
        self.n_files = n_files
        self.x = x
        self.y = y
        self.initFlies()
        self.CalcLikes()
    
    # 対象問題ごとに初期化
    def InitFiles(self):
        self.files = Flies()
        self.filies.vectors = np.zeros([n_flies,2])

    def CalcLikes(self):
        for i in range(self.n_clusters):
            self.flies.likes[i] = self.files.vectors[0]*self.files.vectors[1]

    def Clustering():
        if centers is None:
            model = KMeans(n_clusters,'k-means++')
        else:
            model = KMeans(n_clusters,'k-means++',init = self.centers)
        result = model.fit(vec_flies,init)
        self.flies.labels = result.labels_
        self.centers = result.cluster_centers_

    #def UpdateVectors
