import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt


def dunnIndex(min_inter_cluster_distance, max_intra_cluster_distance):
    return min_inter_cluster_distance / max_intra_cluster_distance

def interClusterDistance(centroids):
    num_centroids = centroids.shape[0]
    min_inter_cluster_dist = 0
    for _c in range(num_centroids-1):
        for _cnext in range(_c+1,num_centroids):
            if np.linalg.norm(centroids[_c] - centroids[_cnext]) > min_inter_cluster_dist:
                min_inter_cluster_dist = np.linalg.norm(centroids[_c] - centroids[_cnext])
    return min_inter_cluster_dist

def inertia(data, centroid, cluster_assignment):
    inertia = 0
    for _cluster in np.unique(cluster_assignment):
        # Filter the data points that belong to the current cluster
        _data = data[cluster_assignment == _cluster]
        inertia = inertia + np.sum(np.linalg.norm((_data - centroid[_cluster]), axis=1))
    return inertia

def intraClusterDistance(data, centroid, cluster_assignment):
    max_intra_cluster_dist = 0
    for _cluster in np.unique(cluster_assignment):
        # Filter the data points that belong to the current cluster
        _data = data[cluster_assignment == _cluster]
        max_intra_cluster_dist = np.max(np.linalg.norm((_data - centroid[_cluster]), axis=1))
    return max_intra_cluster_dist

def kmeans(data, k, iteration=10):
    # Select k random points from the data as the initial i centroids
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

        # Calculating Dunn Index for current cluster assignments
        max_intra_cluster_distance = intraClusterDistance(data, centroid, cluster_assignment)
        min_inter_cluster_distance = interClusterDistance(np.asarray(centroid))
        dunn_index = dunnIndex(min_inter_cluster_distance, max_intra_cluster_distance)
        # print("Dunn index at iter {0} : {1}".format(_iter, dunn_index))

        # Calculating the Inertia for current cluster assignments
        inertia_cluster = inertia(data, centroid, cluster_assignment)
        print("Inertia at iter {0} : {1}".format(_iter, inertia_cluster))

        # Step 4: Recompute the centroids of newly formed clusters
        clusters, cluster_counts = np.unique(cluster_assignment, return_inverse=False, return_counts=True)
        new_centroid = []
        for _col in range(x.shape[1]):
            new_centroid.append(np.bincount(cluster_assignment, weights=data[:, _col]) / cluster_counts)
        new_centroid = np.asarray(new_centroid)
        new_centroid = new_centroid.transpose()
        for _k in range(k):
            centroid[_k] = new_centroid[_k]
        # print("Updated centroids are {0} ".format(centroid))

    return cluster_assignment, centroid, dunn_index, inertia_cluster


if __name__ == "__main__":
    data = pd.read_csv('clustering.csv')
    # print(data.head())
    X = data[["LoanAmount", "ApplicantIncome"]]

    # K-means clustering
    x = np.asarray(X)
    print(x.shape)
    # print(x[:,0])


    # Number of clusters versus Dunn index
    k_dunn = []
    k_inertia = []
    for k in range(2,7):
        cluster_assignment, centroid, dunn_index, inertia_cluster = kmeans(x, k=k, iteration=20)
        print("Cluster Assignment shape ",cluster_assignment.shape)
        k_dunn.append(np.array([k, dunn_index]))
        k_inertia.append(np.array([k, inertia_cluster]))
    k_dunn = np.asarray(k_dunn)
    k_inertia = np.asarray(k_inertia)

    # Plotting the Dunn index vs cluster size
    plt.figure(figsize=(12, 6))
    plt.plot(k_dunn[:,0], k_dunn[:,1], marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Dunn Index')
    plt.title('Dunn Index vs number of clusters')
    plt.show()

    # Plotting the Inertia versus cluster size
    plt.figure(figsize=(12, 6))
    plt.plot(k_inertia[:, 0], k_inertia[:, 1], marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Inertia vs number of clusters')
    plt.show()

    plt.scatter(X["ApplicantIncome"], X["LoanAmount"], c=cluster_assignment)
    plt.scatter(np.asarray(centroid)[:, 1], np.asarray(centroid)[:, 0], c='red')
    plt.xlabel('AnnualIncome')
    plt.ylabel('Loan Amount (In Thousands)')
    plt.show()
