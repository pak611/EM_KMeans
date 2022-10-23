
# in this exercise, you will implement the K-means clustering algorithm and apply it
# to compress an image. 


# import libraries
import numpy as np
import os
import pandas as pd



class KMeansAlgorithm(object):

    def __init__(self, df, K):
        self.data = df.values
        self.x_label = df.columns[0]
        self.y_label = df.columns[1]
        self.K = K                      # num clusters
        self.m = self.data.shape[0]     # num training examples
        self.n = self.data.shape[1]     # num of features
        self.result = {}
        self.centroids = np.array([]).reshape(self.n, 0)

    def init_random_centroids(self, data, K):
        """
        Parameters
        ----------
        data : numpy.ndarray
            DataFrame of 2 features converted into a numpy array.
        K : TYPE
            DESCRIPTION.
        Returns
        -------
        numpy.ndarray
            Centroids will be a (n x K) dimensional matrix.
            Each column will be one centroid for one cluster.
        """
        temp_centroids = np.array([]).reshape(self.n, 0)
        for i in range(self.K):
            rand = rd.randint(0, self.m-1)
            temp_centroids = np.c_[temp_centroids, self.data[rand]]

        return temp_centroids


    def fit_model(self, num_iter):
        """
        Parameters
        ----------
        num_iter : int
            number of iterations until convergenc.
        Returns
        -------
        None.
        """

        # Initiate centroids randomly
        self.centroids = self.init_random_centroids(self.data, self.K)
        # Begin iterations to update centroids, compute and update Euclidean distances
        for i in range(num_iter):
            # First compute the Euclidean distances and store them in array
            EucDist = np.array([]).reshape(self.m, 0)
            for k in range(self.K):
                dist = np.sum((self.data - self.centroids[:,k])**2, axis=1)
                EucDist = np.c_[EucDist, dist]
            # take the min distance
            min_dist = np.argmin(EucDist, axis=1) + 1

            # Begin iterations
            soln_temp = {} # temp dict which stores solution for one iteration - Y

            for k in range(self.K):
                soln_temp[k+1] = np.array([]).reshape(self.n, 0)

            for i in range(self.m):
                # regroup the data points based on the cluster index
                soln_temp[min_dist[i]] = np.c_[soln_temp[min_dist[i]], self.data[i]]

            for k in range(self.K):
                soln_temp[k+1] = soln_temp[k+1].T

            # Updating centroids as the new mean for each cluster
            for k in range(self.K):
                self.centroids[:,k] = np.mean(soln_temp[k+1], axis=0)

            self.result = soln_temp

    def plot_kmeans(self):
        """
        Returns
        -------
        plot
            final plot showing k clusters color coded with centroids.
        """
        # create arrays for colors and labels based on specified K
        colors = ["#"+''.join([rd.choice('0123456789ABCDEF') for j in range(6)]) \
                  for i in range(self.K)]
        labels = ['cluster_' + str(i+1) for i in range(self.K)]

        fig1 = plt.figure(figsize=(5,5))
        ax1 = plt.subplot(111)
        # plot each cluster
        for k in range(self.K):
                ax1.scatter(self.result[k+1][:,0], self.result[k+1][:,1],
                                        c = colors[k], label = labels[k])
        # plot centroids
        ax1.scatter(self.centroids[0,:], self.centroids[1,:], #alpha=.5,
                                s = 300, c = 'lime', label = 'centroids')
        plt.xlabel(self.x_label) # first column of df
        plt.ylabel(self.y_label) # second column of df
        plt.title('Plot of K Means Clustering Algorithm')
        plt.legend()

        return plt.show(block=True)


    def predict(self):
        """
        Returns
        -------
        result
            minimum Euclidean distances from each centroid.
        centroids.T
            K centroids after n_iterations.
        """
        return self.result, self.centroids.T


    def plot_elbow(self):
        """
        Elbow Method:
        The elbow method will help us determine the optimal value for K.
        Steps:
        1) Use a range of K values to test which is optimal
        2) For each K value, calculate Within-Cluster-Sum-of-Squares (WCSS)
        3) Plot Num Clusters (K) x WCSS
        Returns
        -------
        plot
            elbow plot - k values vs wcss values to find optimal K value.
        """

        wcss_vals = np.array([])
        for k_val in range(1, self.K):
            results, centroids = self.predict()
            wcss=0
            for k in range(k_val):
                wcss += np.sum((results[k+1] - centroids[k,:])**2)
            wcss_vals = np.append(wcss_vals, wcss)
        # Plot K values vs WCSS values
        K_vals = np.arange(1, self.K)
        plt.plot(K_vals, wcss_vals)
        plt.xlabel('K Values')
        plt.ylabel('WCSS')
        plt.title('Elbow Method')

        return plt.show(block=True)




def main():

    df_kmeans = generate_data('data/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
    kmeans = KMeansAlgorithm(df_kmeans, K_clusters)
    kmeans.fit_model(num_iterations)
    kmeans.plot_kmeans()
    results, centroids = kmeans.predict()
    kmeans.plot_elbow()


if __name__ == '__main__':
    # Read in command line args
    try:
        K_clusters = int(sys.argv[1])
        num_iterations = int(sys.argv[2])
        main()
    except IndexError:
        raise IndexError('example/ python KMeansAlgorithm.py [K_value: [int], number of iterations: [int]]')



        