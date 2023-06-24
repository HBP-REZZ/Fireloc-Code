import datetime

from matplotlib.lines import Line2D

import time
import numpy as np
from matplotlib.patches import Circle
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
import hdbscan

FILE_PATH = "C:\\Users\\Hugo\\Desktop\\main_project\\data_file.txt"

import matplotlib.pyplot as plt


def generate_random_clusters(num_clusters, num_points_per_cluster, noise_level):
    clusters = []

    # Generate random clusters
    for _ in range(num_clusters):
        center = np.random.uniform(0, 300, size=2)  # Random center for each cluster
        cluster = center + np.random.randn(num_points_per_cluster, 2) * noise_level
        clusters.append(cluster)

    # Add some random noise points
    noise_points = np.random.uniform(0, 300, size=(int(num_points_per_cluster / 2), 2))
    clusters.append(noise_points)

    return clusters


def mrf_clustering(coordinates, threshold):
    # Convert coordinates to numpy array
    coordinates = np.array(coordinates)

    # Compute pairwise distances between coordinates
    distances = np.sqrt(((coordinates[:, None] - coordinates) ** 2).sum(axis=2))

    # Perform clustering using K-Means on the distances
    kmeans = KMeans(n_clusters=10, random_state=0).fit(distances)

    # Get cluster labels
    labels = kmeans.labels_

    # # Create clusters dictionary
    # clusters = {}
    # for i, label in enumerate(labels):
    #     point = tuple(coordinates[i].tolist())
    #     if label in clusters:
    #         clusters[label].append(point)
    #     else:
    #         clusters[label] = [point]
    return labels


def calculate_average_distance(points):
    # Convert points to numpy array
    points = np.array(points)

    # Calculate pairwise distances
    pairwise_distances = distance.cdist(points, points, metric='euclidean')

    # Exclude self-distances on the diagonal
    np.fill_diagonal(pairwise_distances, np.inf)

    # Calculate average distance
    average_distance = np.mean(pairwise_distances)

    return average_distance


def run_clustering_algorithms(points):
    algorithms = {
        'K-means': KMeans(n_clusters=10, n_init='auto'),
        'DBSCAN': DBSCAN(eps=10, min_samples=5),
        'OPTICS': OPTICS(eps=10, min_samples=5),
        'HDBSCAN': hdbscan.HDBSCAN(min_samples=5, min_cluster_size=3)
    }

    results = {}
    time_results = {}

    for algorithm_name, algorithm in algorithms.items():
        print(f"Running {algorithm_name} algorithm...")
        start_time = time.time()

        # Fit the algorithm to the data
        algorithm.fit(points)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{algorithm_name} algorithm executed in {execution_time:.4f} seconds.")
        time_results[algorithm_name] = execution_time
        # Save the cluster labels
        results[algorithm_name] = algorithm.labels_

    print("Running MRF Clustering algorithm...")
    start_time = time.time()

    # Perform MRF Clustering
    mrf_clusters = mrf_clustering(points, calculate_average_distance(points)/10)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"MRF Clustering algorithm executed in {execution_time:.4f} seconds.")
    time_results['MRF Clustering'] = execution_time
    # Save the cluster labels
    results['MRF Clustering'] = mrf_clusters

    # Plot the clustering results
    plot_clustering_results(points, results)
    plot_clustering_results2(points, results)

    return time_results


def run_multiple_times(num_times):
    num_clusters = 10
    points_per_cluster = 30
    noise_level = 20

    total_time_results = {}

    for i in range(num_times):
        separate_clusters = generate_random_clusters(num_clusters, points_per_cluster, noise_level)
        concat_clusters = np.concatenate(separate_clusters)
        time_results = run_clustering_algorithms(concat_clusters)

        # Add the current run's time results to the total
        for algorithm_name, execution_time in time_results.items():
            if algorithm_name in total_time_results:
                total_time_results[algorithm_name] += execution_time
            else:
                total_time_results[algorithm_name] = execution_time

    # Calculate the average runtime for each algorithm
    avg_time_results = {}

    for algorithm_name, total_execution_time in total_time_results.items():
        avg_execution_time = total_execution_time / num_times
        avg_time_results[algorithm_name] = avg_execution_time

    return avg_time_results


def plot_clustering_results(points, results):
    # Plotting parameters
    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)
    plot_num = 1

    # Iterate over each algorithm's results and plot them
    for algorithm_name, cluster_labels in results.items():
        # Create a subplot for the algorithm
        if plot_num == 5 or plot_num == 6:
            plt.subplot(2, 3, plot_num)
        else:
            plt.subplot(2, 3, plot_num)

        # Get a unique color for each cluster
        num_clusters = len(np.unique(cluster_labels))
        cmap = plt.cm.get_cmap('viridis', num_clusters)

        # Plot the points with the assigned labels
        cluster_alpha = max(0.5, 1.0 / len(points))
        plt.scatter(points[:, 0], points[:, 1], c=cluster_labels, cmap=cmap, alpha=cluster_alpha)

        # Set the subplot title and adjust the layout
        plt.title(algorithm_name)
        plt.axis('equal')
        plt.subplots_adjust(hspace=0.4)
        plot_num += 1
        #
         # Add legend for clusters
        # if plot_num == 5 or plot_num == 6:
        #     legend_handles = []
        #     for i in range(num_clusters):
        #         color = cmap(i)
        #         handle = Line2D([0], [0], marker='o', color=color, label=f'Cluster {i}', markersize=10)
        #         legend_handles.append(handle)
        #     plt.legend(handles=legend_handles, loc='center')

    # Show the plot
    plt.show()


def read_data_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()[1:]  # skip the first line

    # list of tuples & arrays
    data = []

    # go over each line and partition its data
    for line in lines:
        line = line.strip().split(';')
        submission_id = int(line[0])
        submission_date = datetime.datetime.strptime(line[1], '%d/%m/%Y %H:%M')
        user_id = int(line[2])
        user_rating = int(line[3])  # rating is both the quality and quantity of the submissions
        fire_verified = int(line[4])
        smoke_verified = int(line[5])
        lat = float(line[6])
        lon = float(line[7])
        text_district = line[8]
        text_parish = line[9]
        text_keywords = line[10]

        # join current iteration line into the input array
        data.append((submission_id, submission_date, user_id, user_rating, fire_verified, smoke_verified, lat, lon,
                     text_district, text_parish, text_keywords))

    return data


def write_coordinates_to_txt(coordinates, file_path):
    with open(file_path, 'w') as file:
        for x, y in coordinates:
            file.write(f"{x},{y}\n")
    print(f"Coordinates written to {file_path} successfully.")


def read_coordinates_from_txt(file_path):
    coordinates = []
    with open(file_path, 'r') as file:
        for line in file:
            x, y = line.strip().split(',')
            coordinates.append([float(x), float(y)])
    return np.array(coordinates)


def plot_clustering_results2(points, results):
    # Plotting parameters
    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)
    plot_num = 1

    # Dictionary to store the centroids
    centroids = {}

    # Iterate over each algorithm's results and plot the centroids
    for algorithm_name, cluster_labels in results.items():
        # Create a subplot for the algorithm
        if plot_num == 5 or plot_num == 6:
            plt.subplot(2, 3, plot_num)
        else:
            plt.subplot(2, 3, plot_num)

        # Get a unique color for each cluster
        num_clusters = len(np.unique(cluster_labels))
        cmap = plt.cm.get_cmap('viridis', num_clusters)

        # Get the unique labels and their corresponding centroids
        unique_labels = np.unique(cluster_labels)
        cluster_centroids = []

        # Iterate over each label and calculate the centroid
        for label in unique_labels:
            centroid = np.mean(points[cluster_labels == label], axis=0)
            cluster_centroids.append(centroid)

        # Store the centroids in the dictionary
        centroids[algorithm_name] = np.array(cluster_centroids)

        # Plot the centroids
        cluster_centroids = np.array(cluster_centroids)
        plt.scatter(cluster_centroids[:, 0], cluster_centroids[:, 1], c='red', marker='x')

        # Set the subplot title and adjust the layout
        plt.title(algorithm_name)
        plt.axis('equal')
        plt.subplots_adjust(hspace=0.4)
        plot_num += 1

    # Show the plot
    plt.show()

    return centroids

if __name__ == '__main__':
    # Example usage:
    num_clusters = 20
    num_points_per_cluster = 40
    noise_level = 10.0

    #separate_clusters = generate_random_clusters(num_clusters, num_points_per_cluster, noise_level)
    #concat_clusters = np.concatenate(separate_clusters)
    #time_results = run_clustering_algorithms(concat_clusters)

    data = read_coordinates_from_txt('C:\\Users\\Hugo\\Desktop\\algorithm_testing\\coordinates2.txt')
    run_clustering_algorithms(data)



