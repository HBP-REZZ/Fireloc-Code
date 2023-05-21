"""
#### Import Libs
"""
import random

import tensorflow as tf

keras = tf.keras
from folium.plugins import MarkerCluster
import folium

from collections import defaultdict

import datetime
import numpy as np
from sklearn.cluster import DBSCAN
import hdbscan

import matplotlib.pyplot as plt

"""
#### Define MACROS and Globals
"""

FILE_PATH = "C:\\Users\\Hugo\\Desktop\\main_project\\data_file2.txt"

# DBSCAN CLUSTERING
DBSCAN_DISTANCE = 0.01
DBSCAN_MIN_SIZE = 3

"""
#### Read input txt file. 
expected input format: sub_id;date/time;user_id;user_rating;fire_verified;smoke_verified;lat;lon;text_district;text_parish;text_keywords
expected output format: (0, datetime.datetime(2023, 3, 28, 10, 15), 1, 17, 1, 0, 40.151, -8.855, 'Coimbra', 'Figueira da Foz', 'Gasolina-Urbanizacao-Apartamentos'
calls encoder functions to create the binary array for keywords. keywords should be separated by - in the input
date ignores seconds
"""


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
        user_rating = int(line[3])
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


"""
#### Read input from console. 
expected input format: sub_id;date/time;user_id;user_rating;fire_verified;smoke_verified;lat;lon;text_district;text_parish;text_keywords
expected output format: (0, datetime.datetime(2023, 3, 28, 10, 15), 1, 17, 1, 0, 40.151, -8.855, 'Coimbra', 'Figueira da Foz', 'Gasolina-Urbanizacao-Apartamentos'
"""


def parse_manual_input(input_str):
    data = []

    line = input_str.strip().split(";")
    submission_id = int(line[0])
    submission_date = datetime.datetime.strptime(line[1], '%d/%m/%Y %H:%M')
    user_id = int(line[2])
    user_rating = int(line[3])
    fire_verified = int(line[4])
    smoke_verified = int(line[5])
    lat = float(line[6])
    lon = float(line[7])
    text_district = line[8]
    text_parish = line[9]
    text_keywords = line[10]

    data.append((submission_id, submission_date, user_id, user_rating, fire_verified, smoke_verified, lat, lon,
                 text_district, text_parish, text_keywords))

    return data


"""
#### Cluster Inputs based on similarity of each submission
# this receives the raw but formatted input from "read_input" and the encoding func, and applies DBSCAN clustering
# this should result in a rough estimation of which event each submission belongs to
# the expected output is a dictionary of the following format:
# { key : [ data point, data point, data point, ...]
# where the key is the cluster label, and the value is an array
# of data points of the same format as the output from read_input()

"""


# TODO H-DBSCAN
# TODO H-DBSCAN
# TODO H-DBSCAN
# extract coordinates and cluster them. Afterwards, handle the noise and return the results for data fusion
def apply_modified_DBSCAN_clustering(data):
    # get the x and y coordinates of the events
    coordinates = np.array([(d[6], d[7]) for d in data])

    # perform DBSCAN clustering
    db = DBSCAN(eps=DBSCAN_DISTANCE, min_samples=DBSCAN_MIN_SIZE).fit(coordinates)

    # get the labels assigned to each event by the clustering algorithm and
    # create a dictionary to store the data of each cluster along with their respective label
    clusters = {}
    for i, label in enumerate(db.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(data[i])

    clusters = handle_noise_DBSCAN(data, clusters, coordinates, db)

    return clusters


# if noise is within x2 DBSCAN distance, assign to the nearest cluster, otherwise create a 1member cluster
# TODO H-DBSCAN
# TODO what if a point joins an existing cluster, but then a point close to it forms a cluster? ex. 13 and 15
def handle_noise_DBSCAN(data, clusters, coordinates, db):
    # handle noise points
    for i in range(len(data)):
        if db.labels_[i] == -1:
            # get the non-noise clusters and their distances
            non_noise_indices = np.where(db.labels_ != -1)[0]
            non_noise_distances = np.linalg.norm(coordinates[non_noise_indices] - coordinates[i], axis=1)

            # if there's existing clusters within up to twice the DBSCAN distance then add noise to that cluster
            if any(non_noise_distances < 2 * DBSCAN_DISTANCE):
                closest_label = max(set(db.labels_[non_noise_indices][non_noise_distances < 2 * DBSCAN_DISTANCE]),
                                    key=list(
                                        db.labels_[non_noise_indices][non_noise_distances < 2 * DBSCAN_DISTANCE]).count)
                print("debug >> " + str(i) + "joined" + str(closest_label))
                # add current submissions data to current label
                clusters[closest_label].append(data[i])

            # otherwise this point is in a very distant area or the start of a new event, so create a 1-event cluster
            else:
                print("debug >> " + str(i) + "started a cluster")
                # create a new label
                new_label = max(clusters.keys()) + 1 if clusters else 0

                # add current submissions' data to current label
                clusters[new_label] = [data[i]]

    # remove noise points from clusters
    clusters.pop(-1, None)

    return clusters


# TODO H-DBSCAN UPDATE FOR NEW INPUTS
def update_HDBSCAN_clusters():
    # TODO AAAAAAAAAAAAAAAAAAAAAAAAAAA
    # TODO AAAAAAAAAAAAAAAAAAAAAAAAAAA
    # TODO AAAAAAAAAAAAAAAAAAAAAAAAAAA

    return 0


"""
#### Fuse all known data clusters
# apply_data_fusion() iterates over all clusters and sends individual clusters to the fusion function. 
Returns all fused clusters. 
"""


def apply_data_fusion(clusters):
    # Initialize an empty dictionary to store fused clusters
    # fused_clusters = {}
    fused_clusters = []
    # Loop over each cluster
    for cluster_id, cluster_members in clusters.items():
        fused_events = fuse_cluster_submissions(cluster_members)

        # Add fused event to dictionary of fused clusters
        # fused_clusters[cluster_id] = [fused_events]
        fused_clusters.append(fused_events)

    return fused_clusters


"""
#### Fuse individual members of a single cluster
# fuse_cluster_submissions() loads up all the data and creates an event dictionary.  
# strings are handled in separate functions handle_locations() and handle_keywords().
# centroids are handled in a separate function, big_circle_calculation2().
# from 1 cluster, 1 event is created. this event is returned to apply_data_fusion() and saved in an array.
"""


def fuse_cluster_submissions(cluster_members):
    # Initialize lists to hold fused data
    fused_dates = []
    fused_user_ids = []
    fused_sub_ids = []
    fused_ratings = []
    fused_latitudes = []
    fused_longitudes = []
    fused_districts = defaultdict(int)
    fused_parishes = defaultdict(int)
    fused_keywords = defaultdict(int)
    fused_fire_verified = 0
    fused_smoke_verified = 0

    # Loop over each submission in the cluster
    for submission in cluster_members:
        # Extract data from submission
        sub_id, date, user_id, user_rating, fire_verified, smoke_verified, latitude, longitude, district, parish, keywords = submission

        # Add numeric data to lists
        fused_dates.append(date)
        fused_user_ids.append(user_id)  # TODO right now, there are duplicate ids. is this of interest? is it preferable to have unique ids of the users that have contributed? or keep it this way to know how many times one has contributed?
        fused_sub_ids.append(sub_id)
        fused_ratings.append(user_rating)
        fused_latitudes.append(latitude)
        fused_longitudes.append(longitude)

        # fill in dictionaries with the number of occurrences of districts and parishes
        fused_districts, fused_parishes = handle_locations(district, parish, fused_districts, fused_parishes)

        # fill in dictionaries with the number of occurrences of keywords
        fused_keywords = handle_keywords(keywords, fused_keywords)

        # Update fire/smoke verified flags if one of the fusion members has positive flags
        fused_fire_verified |= fire_verified
        fused_smoke_verified |= smoke_verified

    # Calculate centroid of the coordinates in the numpy arrays
    centroid_latitude, centroid_longitude = big_circle_calculation2(fused_latitudes, fused_longitudes)

    # Create fused event
    fused_event = {
        'date_latest': max(fused_dates),
        'date_history': fused_dates,
        'user_ids': fused_user_ids,
        'sub_ids': fused_sub_ids,
        'rating': np.mean(fused_ratings),
        'fire_verified': fused_fire_verified,
        'smoke_verified': fused_smoke_verified,
        'latitude': centroid_latitude,
        'longitude': centroid_longitude,
        'districts': list(fused_districts.items()),
        'parishes': list(fused_parishes.items()),
        'keywords': list(fused_keywords.items())
    }

    return fused_event


"""
# handle_locations() and handle_keywords() go over all strings.
# for each non-empty string, a dictionary entry is created with the number of occurrences of said string
# these functions receive a dictionary entry that may or may not be empty, 
# and return an updated version of this same entry.
"""


def handle_locations(district, parish, fused_districts, fused_parishes):
    # Ignore empty entries and spaces
    district = district.strip()
    parish = parish.strip()

    # Count the districts and parishes that appear within submissions
    if district:
        # add +1 weight
        fused_districts[district] += 1

    if parish:
        # add +1 weight
        fused_parishes[parish] += 1

    return fused_districts, fused_parishes


def handle_keywords(keywords, fused_keywords):
    # keywords are split based on the "-" char to include keywords with spaces ex. "toxic gas".
    # empty " - " inputs are also handled.
    keywords = keywords.split("-")

    # since this may need to handle multiple strings
    for key in keywords:
        if key:
            # add +1 weight
            fused_keywords[key] += 1

    return fused_keywords


"""
# big_circle_calculation() and big_circle_calculation2() calculate the centroid of the input coordinates.
# big_circle_calculation() calcs standard average
# big_circle_calculation2() calcs using the geodesic algorithm (from an existing implementation)
"""


def big_circle_calculation(fused_latitudes, fused_longitudes):
    centroid_latitude = np.mean(fused_latitudes)
    centroid_longitude = np.mean(fused_longitudes)

    return centroid_latitude, centroid_longitude


def big_circle_calculation2(fused_latitudes, fused_longitudes):
    radius = 6371  # Radius of the earth in km

    lat_radians = np.radians(fused_latitudes)
    long_radians = np.radians(fused_longitudes)

    x = radius * np.cos(lat_radians) * np.cos(long_radians)
    y = radius * np.cos(lat_radians) * np.sin(long_radians)
    z = radius * np.sin(lat_radians)

    centroid_x = np.mean(x)
    centroid_y = np.mean(y)
    centroid_z = np.mean(z)

    centroid_long = np.arctan2(centroid_y, centroid_x)
    centroid_hyp = np.sqrt(centroid_x ** 2 + centroid_y ** 2)
    centroid_lat = np.arctan2(centroid_z, centroid_hyp)

    centroid_lat_degrees = np.degrees(centroid_lat)
    centroid_long_degrees = np.degrees(centroid_long)

    return centroid_lat_degrees, centroid_long_degrees


"""
#### Folium plot - plots based on coordinates only, prints event-specific information within the markers. 
"""


def plot_folium1(data, map_name):
    # Create a map centered at the first event in the list
    map_center = [data[0][6], data[0][7]]
    map_plot = folium.Map(location=map_center, zoom_start=6)

    # Create a marker cluster layer to add things to the map.
    # disableClusteringAtZoom -> at what zoom lvl events disperse
    marker_cluster = MarkerCluster(disableClusteringAtZoom=10).add_to(map_plot)

    # Iterate over the events and add markers to the map
    for event in data:
        # Extract the latitude, longitude, and text location from the event
        lat, lon, text_loc = event[6], event[7], event[9]

        # Create a popup message with the event data
        popup_text = f"Sub ID: {event[0]}<br>Date/Time: {event[1]}<br>User ID: {event[2]}<br>User rating: {event[3]}" \
                     f"<br>Fire verified: {event[4]}<br>Smoke verified: {event[5]}<br>District: {event[8]}" \
                     f"<br>Location: {event[9]}"

        # Add the marker to the marker cluster layer
        folium.Marker([lat, lon], popup=popup_text).add_to(marker_cluster)

    map_plot.save(map_name)

    return 0


def plot_folium2(data, map_name):
    # Create a map centered at the first event in the list
    map_center = [data[0].get('latitude'), data[0].get('longitude')]
    map_plot = folium.Map(location=map_center, zoom_start=6)

    # Create a marker cluster layer to add things to the map.
    # disableClusteringAtZoom -> at what zoom lvl events disperse
    marker_cluster = MarkerCluster(disableClusteringAtZoom=10).add_to(map_plot)

    # Iterate over the events and add markers to the map
    for event in data:
        # Extract the latitude, longitude, and text location from the event
        lat, lon, text_loc = event.get('latitude'), event.get('longitude'), event.get('location')

        # Create a popup message with the event data
        popup_text = f"Date/Time: {event.get('date_latest')}<br>User IDs: {event.get('user_ids')}<br>Submission IDs: {event.get('sub_ids')}" \
                     f"<br>Avg rating: {event.get('rating')}<br>Fire verified: {event.get('fire_verified')}<br>Smoke " \
                     f"verified: {event.get('smoke_verified')}" \
                     f"<br>District: {event.get('districts')}<br>Parish: {event.get('parishes')}"

        # Add keywords to the popup message
        keywords = event.get('keywords')
        if keywords:
            popup_text += '<br><br>Keywords:'
            for keyword, count in keywords:
                popup_text += f'<br>{keyword}: {count}'

        # Add the marker to the marker cluster layer
        folium.Marker([lat, lon], popup=popup_text).add_to(marker_cluster)

    map_plot.save(map_name)

    return 0


def plot_folium3(data, map_name):
    # Create a map centered at the first event in the list
    first_event = next(iter(data.values()))[0]  # get the first event in the first cluster
    map_center = [first_event[6], first_event[7]]
    map_plot = folium.Map(location=map_center, zoom_start=6)

    # Define a list of colors to use for the markers
    colors = ['blue', 'green', 'purple', 'orange']

    # Create a marker cluster layer to add things to the map.
    # disableClusteringAtZoom -> at what zoom lvl events disperse
    marker_cluster = MarkerCluster(disableClusteringAtZoom=10).add_to(map_plot)

    # Iterate over the clusters and events, and add markers to the map
    for i, (cluster_id, event_list) in enumerate(data.items()):
        for event in event_list:
            # Extract the latitude, longitude, and text location from the event
            lat, lon, text_loc = event[6], event[7], event[9]

            # Add the marker to the marker cluster layer, using the color for this cluster
            folium.Marker([lat, lon], icon=folium.Icon(color=colors[i % len(colors)])).add_to(marker_cluster)

    map_plot.save(map_name)

    return 0


def hdbscan_tester():
    # generate initial data
    data = np.random.randn(100, 2)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, gen_min_span_tree=True, prediction_data=True)
    clusterer.fit(data)

    print("Initial Clusters: ", clusterer.labels_)

    # Plot all the points
    plt.scatter(data[:, 0], data[:, 1])
    plt.ion()
    plt.show()
    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], c=clusterer.labels_, cmap='tab20')

    new_point_plot = ax.scatter([], [], c='black', marker='x')
    while True:
        # read in new point
        new_point_str = input("Enter new point as 'x,y': ")
        new_point = np.array(new_point_str.split(","), dtype=float)

        # predict the cluster label and strength for the new point
        labels, strengths = hdbscan.approximate_predict(clusterer, new_point.reshape(1, -1))
        print("New Point: ", new_point)
        print("Predicted Cluster Label: ", labels[0])
        print("Predicted Cluster Strength: ", strengths[0])

        # update the plot with the new point
        new_point_plot.set_offsets(new_point)
        new_point_plot.set_color(plt.cm.tab20(labels[0]))
        fig.canvas.draw()
        plt.pause(0.1)


if __name__ == '__main__':
    # hdbscan_tester()

    data_input = read_data_file(FILE_PATH)
    print(data_input[0])  # Debug Help

    plot_folium1(data_input, "fireloc_map_raw.html")

    clustered_data = apply_modified_DBSCAN_clustering(data_input)
    print(next(iter((clustered_data.items()))))  # print first key-value pair just for testing

    plot_folium3(clustered_data, "fireloc_map_clustered.html")

    fused_events = apply_data_fusion(clustered_data)
    plot_folium2(fused_events, "fireloc_map_fused.html")

"""
    while True:
        user_input = input("(Only one line at a time) >>  ")
        # Quit program
        if user_input == "quit" or user_input == "Quit" or user_input == "q" or user_input == "Q":
            break
        # Simulate real-time submissions
        else:
            # get random input
            line_data = parse_manual_input(user_input)
            print(line_data)

            # handle clustering for this input

            # handle fusion for this input

            # update maps

            # return to loop

    # plot_clusters2(clustered_data, fused_data)
"""
