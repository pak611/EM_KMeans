#%%

# write a k-means algorithm

# import libraries
import numpy as np
import os
import pandas as pd



def euclidean_distance(point, centroid):

    """ computes the euclidean distance 
    between a point and the centroid 
     """
    return np.sqrt(np.sum((point - centroid)**2))


def assign_label_cluster(distance, data_point, centroids): 
    min_idx = min(distance, key = distance.get) # get the minimum distance and assign to a centroid
    return [min_idx, data_point, centroids[min_idx]]



def new_centroids(cluster_label, centroids):
    """
    will return the average of the cluster label and
    the centroids 
    """
    return np.array(cluster_label + centroids)/2



def iterate_k_means(data_points, centroids, total_iteration):
    label = []
    cluster_label = []
    total_points = len(data_points)
    k = len(centroids)

    for iteration in range(0, total_iteration):
        for index_point in range(0, total_points):
            distance = {}
            for index_centroid in range(0, k): # measure the distance between each point and k centroids



                distance[index_centroid] = euclidean_distance(data_points[index_point], centroids[index_centroid])

                # assign to cluster with minimum distance
                label = assign_label_cluster(distance, data_points[index_point], centroids)

                # take the average of 

                # label[0] is  min_idx
                # label[1] is dat_point

                # so we're taking the new centroid to be the average
                # of the old centroid and the points within the cluster?
                #... but wasn't the old centroid based on the mean? 

                centroids[label[0]] = new_centroids(label[1], centroids[label[0]])

                # so we're adding more centroids.. and each subsequent
                # centroid is based on the data points that fall within 
                # the cluster and the centroid for the cluster
                


                if iteration == (total_iteration - 1):
                    cluster_label.append(label)

    # perform multiple iteratinos and choose the best of these

    return [cluster_label, centroids]



def print_label_data(result):
    print("Result of k-Means Clustering: \n")
    for data in result[0]:
        print("data point: {}".format(data[1]))
        print("cluster number: {} \n".format(data[0]))
    print("Last centroids position: \n {}".format(result[1]))


def create_centroids():
    centroids = []
    centroids.append([5.0, 0.0])
    centroids.append([45.0, 70.0])
    centroids.append([50.0, 90.0])
    return np.array(centroids)




#%%


if __name__ == "__main__":

    filename = os.path.dirname(__file__) + "/data.csv"
    data_points = np.genfromtxt(filename, delimiter=",")
    #data_points = pd.read_csv(filename)
    centroids = create_centroids()
    total_iteration = 100

    [cluster_label, new_centroids] = iterate_k_means(data_points, centroids, total_iteration)
    print_label_data([cluster_label, new_centroids])
    print()

