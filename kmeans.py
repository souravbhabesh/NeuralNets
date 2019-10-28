import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

def kmeans(data, k, iteration = 10):
    # Select k random points from the data as centroids
    k_random_pts = np.random.randint(data.shape[0], size=k)
    centroid = [data[i] for i in k_random_pts]
    print("Initial centroids are {0}".format(centroid))

    # Step 3: Assign all the points to the closest cluster centroid
    cluster_assignment = np.zeros(data.shape[0])
    for _iter in range(iteration):
        intra_cluster_dist = []
        for _centroid in centroid:
            # print((data- _centroid).shape)
            intra_cluster_dist.append(np.linalg.norm(data - _centroid, axis=1))
        intra_cluster_dist = np.array(intra_cluster_dist).transpose()
        # print("intra cluster shape ",intra_cluster_dist.shape)
        cluster_assignment = np.argmin(intra_cluster_dist, axis=1)
        # print(cluster_assignment.shape)

        # Step 4: Recompute the centroids of newly formed clusters
        clusters, cluster_counts = np.unique(cluster_assignment, return_inverse=False, return_counts=True)
        new_centroid = []
        for _col in range(x.shape[1]):
            new_centroid.append(np.bincount(cluster_assignment, weights=data[:, _col])/cluster_counts)
        new_centroid = np.asarray(new_centroid)
        new_centroid = new_centroid.transpose()
        for _k in range(k):
            centroid[_k] = new_centroid[_k]
        print("Updated centroids are {0} ".format(centroid))

    return cluster_assignment, centroid

if __name__ == "__main__":
    data = pd.read_csv('clustering.csv')
    # print(data.head())
    X = data[["LoanAmount", "ApplicantIncome"]]

    # K-means clustering
    x = np.asarray(X)
    print(x.shape)
    # print(x[:,0])

    cluster_assignment, centroid = kmeans(x, k=3)
    plt.scatter(X["ApplicantIncome"], X["LoanAmount"], c=cluster_assignment)
    plt.scatter(np.asarray(centroid)[:,1], np.asarray(centroid)[:,0], c='red')
    plt.xlabel('AnnualIncome')
    plt.ylabel('Loan Amount (In Thousands)')
    plt.show()