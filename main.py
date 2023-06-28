import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class InstaGramFlies:
    n_iters = None
    n_clusters = None
    n_flies = None
    n_features = None

    centers = None
    # best fly in in each cluster
    best_fly_indices = None
    cluster_like_average = None
    center_dist_average = None

    likes = None
    labels = None
    # np.array([[pioneer rate, faddist rate, master rate],[...],...])
    strategies = None
    vectors = None  # np.array([x1,x2,x3,...]) x1: np.array

    def __init__(self, n_iters, n_clusters, n_flies):
        self.n_iters = n_iters
        self.n_clusters = n_clusters
        self.n_flies = n_flies
        self.InitFlies()
        self.EvaluateLikes()

    def InitFlies(self):
        self.vectors = np.random.uniform(0, 100, [self.n_flies, 2])
        self.strategies = np.zeros([self.n_flies, 3])
        for i in range(self.n_flies):
            randoms = np.random.uniform(1, 100, 3)
            for j in range(3):
                self.strategies[i][j] = randoms[j]/sum(randoms)
        self.likes = np.zeros(self.n_flies)
        self.n_features = len(self.vectors[0])
        self.best_fly_indices = np.zeros(self.n_clusters)

    def EvaluateLikes(self):
        for i in range(self.n_flies):
            self.likes[i] = self.vectors[i][0]*self.vectors[i][1]

    def Clustering(self):
        if self.centers is None:
            model = KMeans(n_clusters=self.n_clusters)
        else:
            model = KMeans(n_clusters=self.n_clusters, init=self.centers)
        result = model.fit(self.vectors)
        self.labels = result.labels_
        self.centers = result.cluster_centers_

        # best flies in each cluster
        best = np.zeros(self.n_clusters)
        self.cluster_like_average = np.zeros(self.n_clusters)
        for i in range(self.n_flies):
            label = self.labels[i]
            if (self.likes[i] > best[label]):
                best[label] = self.likes[i]
                self.best_fly_indices[label] = i

        # like average in each cluster
        for i in range(self.n_clusters):
            self.cluster_like_average[i] = np.mean(
                [self.likes[j] for j in range(self.n_flies) if self.labels[j] == i])

        # average dist between each cluster
        self.center_dist_average = 0
        for i in range(self.n_clusters):
            for j in range(i+1,self.n_clusters):
                self.center_dist_average += np.linalg.norm(self.centers[i]-self.centers[j])
        self.center_dist_average /= sum(range(1,self.n_clusters))

                

    def UpdateFlieVector(self):
        randoms = np.random.rand(self.n_flies)
        sum = 0
        for i in range(self.n_flies):
            action = 0
            for a in range(3):
                action = a
                sum = sum + self.strategies[i]
                if sum > randoms[i]:
                    break
            # pioneer
            if action == 0:
                self.vectors[i] = self.UpdatePioneer(self.vectors[i])
            # faddist
            if action == 1:
                self.vectors[i] = self.UpdateFaddist(self.vectors[i])
            # master
            if action == 2:
                self.vectors[i] = self.UpdateMaster(
                    self.vectors[i], self.labels[i])

    def UpdatePioneer(self, vector):
        dist = 0
        #TODO kokokara!!! averageを使ってUpdatePioneerを更新
        return 100

    def UpdateFaddist(self, vector):
        return 100

    def UpdateMaster(self, vector, label):
        center = self.centers[label]
        return 100


if __name__ == "__main__":
    n_clusters = 3
    n_flies = 10
    instaFlies = InstaGramFlies(100, n_clusters, n_flies)
    instaFlies.Clustering()
    print(instaFlies.centers,instaFlies.center_dist_average)
    plt.scatter(instaFlies.centers.T[0], instaFlies.centers.T[1], marker="*")
    plt.scatter(instaFlies.vectors.T[0], instaFlies.vectors.T[1], marker=".")
    plt.show()
