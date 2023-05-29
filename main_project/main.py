"""
#### Import Libs
"""
import folium
from folium.plugins import MarkerCluster

from collections import defaultdict
import datetime
import numpy as np

import hdbscan

"""
#### Define MACROS and Globals
"""

FILE_PATH = "C:\\Users\\Hugo\\Desktop\\main_project\\data_file2.txt"

NOISE_DISTANCE_THRESHOLD = 1.5

KEYWORD_WEIGHTS = {
    # SIGNS
    "Fumo": 1,
    "Fogo": 2,
    # FUELS
    "Explosivo": 5,
    "Fertilizante": 4,
    "Quimico": 5,
    "Gas": 5,
    "Gasolina": 5,
    "Petroleo": 5,
    # HOUSES & PEOPLE
    "Urbana": 2,
    "Urbananizaçao": 2,
    "Casa": 2,
    "Populacao": 2,
    "Hospital": 3,
}

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


# write console input to data file. make sure there's an empty line in the end of the file
def write_data_file(file_path, data_string):
    with open(file_path, 'a') as f:
        f.write(data_string + "\n")


"""
#### Read input from console. 
expected input format: sub_id;date/time;user_id;user_rating;fire_verified;smoke_verified;lat;lon;text_district;text_parish;text_keywords
expected output format: (0, datetime.datetime(2023, 3, 28, 10, 15), 1, 17, 1, 0, 40.151, -8.855, 'Coimbra', 'Figueira da Foz', 'Gasolina-Urbanizacao-Apartamentos'
"""


def parse_manual_input(input_str):
    line = input_str.strip().split(";")

    if len(line) < 8:
        raise ValueError("DEBUG: Critical Manual Input Error -> missing critical fields of data")
    try:
        submission_id = int(line[0])
        submission_date = datetime.datetime.strptime(line[1], '%d/%m/%Y %H:%M')

        user_id = int(line[2])
        user_rating = int(line[3])

        fire_verified = int(line[4])
        if fire_verified not in [0, 1]:
            raise ValueError("Invalid fire_verified value. Must be 0 or 1.")

        smoke_verified = int(line[5])
        if smoke_verified not in [0, 1]:
            raise ValueError("Invalid smoke_verified value. Must be 0 or 1.")

        lat = float(line[6])
        if lat < 0:
            print("Warning: latitude value should be positive for current context (Portugal).")

        lon = float(line[7])
        if lon >= 0:
            print("Warning: longitude value should be negative for current context (Portugal).")

        text_district = line[8] if len(line) > 8 and line[8] != '' else ''
        text_parish = line[9] if len(line) > 9 and line[9] != '' else ''
        text_keywords = line[10] if len(line) > 10 and line[10] != '' else ''

    except ValueError:
        raise ValueError("DEBUG: Critical Manual Input Error -> incorrect/missing data formats")

    data_line = (submission_id, submission_date, user_id, user_rating, fire_verified, smoke_verified, lat, lon,
                 text_district, text_parish, text_keywords)

    return data_line


"""
#### Cluster Inputs based on similarity of each submission
# this receives the raw but formatted input from "read_input" and the encoding func, and applies clustering
# this should result in a rough estimation of which event each submission belongs to
# the expected output is a dictionary of the following format:
# { key : [ data point, data point, data point, ...]
# where the key is the cluster label, and the value is an array
# of data points of the same format as the output from read_input()
# it extract coordinates and cluster based on them.
# noise points will either be merged with existing clusters if within NOISE_DISTANCE_THRESHOLD, 
# or added to the noise label to await re-clustering once more points are added to the database
"""


def apply_clustering_algorithm(data):
    # get the x and y coordinates of the events
    coordinates = np.array([(d[6], d[7]) for d in data])

    # perform HDBSCAN clustering
    clusterer_instance = hdbscan.HDBSCAN(min_samples=1, min_cluster_size=2,
                                         metric='haversine', prediction_data=True, allow_single_cluster=True)
    clustering_results = clusterer_instance.fit(coordinates)

    # get the labels assigned to each event by the clustering algorithm and
    # create a dictionary to store the data of each cluster along with their respective label
    clusters = {}
    noise_points = []

    for i, label in enumerate(clustering_results.labels_):
        # if noise -> is too far to be assigned to a nearby cluster, add it to noise_points list
        if label == -1:
            # calculate the distances between the noise point and all other cluster centroids
            distances = np.array([
                haversine_distance(
                    coordinates[i][0], coordinates[i][1],
                    np.mean(coordinates[clustering_results.labels_ == lab], axis=0)[0],
                    np.mean(coordinates[clustering_results.labels_ == lab], axis=0)[1]
                )
                for lab in set(clustering_results.labels_) if lab != -1
            ])

            # if there are clusters within the threshold, assign the noise point to the nearest existing cluster
            if len(distances) > 0 and np.min(distances) <= NOISE_DISTANCE_THRESHOLD:
                nearest_cluster_label = list(set(clustering_results.labels_))[np.argmin(distances)]
                if nearest_cluster_label not in clusters:
                    clusters[nearest_cluster_label] = []
                clusters[nearest_cluster_label].append(data[i])
                clustering_results.labels_[i] = nearest_cluster_label

            else:
                noise_points.append(data[i])

        # else add found labels and their respective data to "clusters"
        else:
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(data[i])

    # assign noise_points to a separate noise label
    if noise_points:
        noise_label = -1
        clusters[noise_label] = noise_points

    return clusters, clusterer_instance


"""
update_hdbscan_approximate_predict will check if any new points belong to existing clusters. 
this is an extremely light operation. 
any points that dont belong to existing clusters will be labeled noise.
"""


def update_hdbscan_approximate_predict(data_point, clusterer_instance, clusters):
    label = -1
    # get the x and y coordinates of the new data point
    coordinates = np.array([(data_point[6], data_point[7])])

    # get the predicted labels for the new data point using approximate_predict method
    new_labels, strengths = hdbscan.approximate_predict(clusterer_instance, coordinates)

    # update the existing clusters' dictionary with the new labels and data
    for i, label in enumerate(new_labels):
        # if noise -> is too far to be assigned to a nearby cluster, this will be added to cluster queue as noise
        if label == -1:
            return clusters, label, False
        # else add found labels and their respective data to "clusters"
        else:
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(data_point)

    return clusters, label, True


"""
update_hdbscan_fit_predict will check if any new points belong AND OR create new clusters. 
this is an heavier operation. 
caution was needed when updating labels because fit_predict() will always start labels at 0 and these will conflict with
the old labels in clusters if not mapped properly
"""


def update_hdbscan_fit_predict(data, clusterer_instance, clusters):
    # include all existing noise points in the data to be clustered
    data = data + clusters.pop(-1, [])

    # get the x and y coordinates of the events, including the noise points
    coordinates = np.array([(d[6], d[7]) for d in data])

    # get the predicted labels for the new data points using fit_predict method
    new_labels = clusterer_instance.fit_predict(coordinates)

    # create a mapping between old and new labels, and update the clusters
    label_mapping = {}
    for i, label in enumerate(new_labels):
        # if the new label is -1 then add it to a -1 cluster
        if label == -1:
            if -1 in clusters:
                clusters[-1].append(data[i])
            else:
                clusters[-1] = [data[i]]
            # skip the rest of the loop if noise
            continue

        # there's a clustered point - unique -> create a new cluster
        if label not in label_mapping:
            # Calc a new unique label value
            new_label = len(label_mapping) + (max(clusters.keys()) + 1 if clusters else 0)
            label_mapping[label] = new_label

        # there's a clustered point - NOT unique -> use an existing cluster for it
        else:
            new_label = label_mapping[label]

        # UPDATE clusters dictionary with the correct key (new_label) and value (data[i])
        if new_label in clusters:
            clusters[new_label].append(data[i])
        else:
            clusters[new_label] = [data[i]]

    return clusters


"""
#### Fuse all known data clusters
# apply_data_fusion() iterates over all clusters and sends individual clusters to the fusion function. 
Returns all fused clusters. 
"""


def apply_data_fusion(clusters):
    # Initialize an empty dictionary to store fused clusters
    fused_clusters = {}

    # Loop over each cluster
    for cluster_id, cluster_members in clusters.items():
        # if handling the standard clusters
        if cluster_id != -1:
            events = fuse_cluster_submissions(cluster_id, cluster_members)

            # Add fused event to dictionary of fused clusters
            fused_clusters[cluster_id] = events

    # if handling the submissions labeled as noise
    handle_noise_submissions(clusters.get(-1), fused_clusters)

    return fused_clusters


def handle_noise_submissions(cluster_members, fused_clusters):
    # Loop over each submission in the cluster
    for i, submission in enumerate(cluster_members):
        sub_id, date, user_id, user_rating, fire_verified, smoke_verified, latitude, longitude, district, parish, keywords = submission

        fused_districts = defaultdict(lambda: {"counter": 0, "weight": 0})
        fused_parishes = defaultdict(lambda: {"counter": 0, "weight": 0})
        fused_keywords = defaultdict(lambda: {"counter": 0, "weight": 0})
        event_hazard_level = 0

        # fill in dictionaries with the number of occurrences of districts and parishes
        fused_districts, fused_parishes = handle_locations(district, parish, fused_districts, fused_parishes, user_rating)

        # fill in dictionaries with the number of occurrences of keywords vs their set weights
        fused_keywords = handle_keywords(keywords, fused_keywords, user_rating)

        # Create 1-submission event
        fused_event = {
            'event_id': -(i + 1),  # Assign negative ID for each noise event
            'date_latest': date,
            'date_age': datetime.datetime.now() - ((datetime.datetime.now() - date) % datetime.timedelta(minutes=1)),
            'date_history': date,
            'user_ids': user_id,
            'sub_ids': sub_id,
            'rating': user_rating,
            'fire_verified': fire_verified,
            'smoke_verified': smoke_verified,
            'latitude': latitude,
            'longitude': longitude,
            'districts': list(fused_districts.items()),
            'parishes': list(fused_parishes.items()),
            'keywords': list(fused_keywords.items())
        }

        # update fused clusters with noise events
        fused_clusters[fused_event.get('event_id')] = fused_event

    return fused_clusters


"""
#### Fuse individual members of a single cluster
# fuse_cluster_submissions() loads up all the data and creates an event dictionary.  
# strings are handled in separate functions handle_locations() and handle_keywords().
# centroids are handled in a separate function, big_circle_calculation2().
# from 1 cluster, 1 event is created. this event is returned to apply_data_fusion() and saved in an array.
# the cluster ID becomes the EVENT ID
"""


def fuse_cluster_submissions(cluster_id, cluster_members):
    # Initialize lists to hold fused data
    fused_dates = []
    fused_user_ids = []
    fused_sub_ids = []
    fused_ratings = []
    fused_latitudes = []
    fused_longitudes = []
    fused_districts = defaultdict(lambda: {"counter": 0, "weight": 0})
    fused_parishes = defaultdict(lambda: {"counter": 0, "weight": 0})
    fused_keywords = defaultdict(lambda: {"counter": 0, "weight": 0})
    fused_fire_verified = 0
    fused_smoke_verified = 0
    event_hazard_level = 0

    # Loop over each submission in the cluster
    for submission in cluster_members:
        # Extract data from submission
        sub_id, date, user_id, user_rating, fire_verified, smoke_verified, latitude, longitude, district, parish, keywords = submission

        # Add numeric data to lists
        fused_dates.append(date)  # for timeseries and time progression?
        fused_user_ids.append(user_id)  # ID info
        fused_sub_ids.append(sub_id)  # ID info
        fused_ratings.append(user_rating)  # This info is handed by fireloc
        fused_latitudes.append(latitude)  # This info is handed by fireloc
        fused_longitudes.append(longitude)  # This info is handed by fireloc

        # fill in dictionaries with the number of occurrences of districts and parishes
        fused_districts, fused_parishes = handle_locations(district, parish, fused_districts, fused_parishes, user_rating)

        # fill in dictionaries with the number of occurrences of keywords vs their set weights
        fused_keywords = handle_keywords(keywords, fused_keywords, user_rating)

        # Update fire/smoke verified flags if one of the fusion members has positive flags. This info is handed by fireloc
        fused_fire_verified |= fire_verified
        fused_smoke_verified |= smoke_verified

    # Calculate centroid of the coordinates in the numpy arrays
    centroid_latitude, centroid_longitude = haversine_centroid(fused_latitudes, fused_longitudes)

    # Create fused event
    fused_event = {
        'event_id': cluster_id,
        'date_latest': max(fused_dates),
        'date_age': datetime.datetime.now() - ((datetime.datetime.now() - min(fused_dates)) % datetime.timedelta(minutes=1)),
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
#### Fuse individual submission with an existing event that already went through the fusion procedure
"""


def apply_partial_data_fusion(new_data_point, fused_event):
    # Extract the data from the existing fused event
    fused_dates = fused_event['date_history']
    fused_user_ids = fused_event['user_ids']
    fused_sub_ids = fused_event['sub_ids']
    fused_ratings = fused_event['rating']
    fused_latitudes = fused_event['latitude']
    fused_longitudes = fused_event['longitude']
    fused_districts = fused_event['districts']
    fused_parishes = fused_event['parishes']
    fused_keywords = fused_event['keywords']
    fused_fire_verified = fused_event['fire_verified']
    fused_smoke_verified = fused_event['smoke_verified']

    # Extract the data from the new data point
    sub_id, date, user_id, user_rating, fire_verified, smoke_verified, latitude, longitude, district, parish, keywords = new_data_point

    # Data that is calculated using inbuilt numpy methods are simply appended into arrays
    fused_dates.append(date)
    fused_user_ids.append(user_id)
    fused_sub_ids.append(sub_id)
    fused_ratings.append(user_rating)
    fused_latitudes.append(latitude)
    fused_longitudes.append(longitude)

    # Update dictionaries with the number of occurrences of districts and parishes
    fused_districts, fused_parishes = handle_locations(district, parish, fused_districts, fused_parishes, user_rating)

    # Update dictionaries with the number of occurrences of keywords vs their set weights
    fused_keywords = handle_keywords(keywords, fused_keywords, user_rating)

    # Update fire/smoke verified flags
    fused_fire_verified |= fire_verified
    fused_smoke_verified |= smoke_verified

    # Calculate centroid of the updated coordinates
    centroid_latitude, centroid_longitude = haversine_centroid(fused_latitudes, fused_longitudes)

    # Create the updated fused event
    fused_event = {
        'event_id': fused_event['event_id'],  # event_id remains the same
        'date_latest': max(fused_dates),
        'date_age': datetime.datetime.now() - ((datetime.datetime.now() - min(fused_dates)) % datetime.timedelta(minutes=1)),
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


def handle_locations(district, parish, fused_districts, fused_parishes, user_rating):
    # Ignore empty entries and spaces
    district = district.strip()
    parish = parish.strip()

    # Count the districts and parishes that appear within submissions
    if district:
        # calculate counter & final weight
        if district in fused_districts:
            # increment counter and set weight calculation to counter value
            fused_districts[district]["counter"] += (user_rating / 20)
            fused_districts[district]["weight"] += 0
        else:
            # Create a new entry for the keyword
            fused_districts[district] = {
                "counter": user_rating / 20,
                "weight": 0
            }

    if parish:
        # calculate counter & final weight
        if parish in fused_parishes:
            # increment counter and set weight calculation to counter value
            fused_parishes[parish]["counter"] += (user_rating / 20)
            fused_parishes[parish]["weight"] += 0
        else:
            # Create a new entry for the keyword
            fused_parishes[parish] = {
                "counter": user_rating / 20,
                "weight": 0
            }

    # Calculate the total counter value for districts and parishes
    total_districts = sum(entry["counter"] for entry in fused_districts.values())
    total_parishes = sum(entry["counter"] for entry in fused_parishes.values())

    # Calculate the weights as a percentage of likelihood
    for district, data in fused_districts.items():
        data["weight"] = round((data["counter"] / total_districts) * 100, 2)

    for parish, data in fused_parishes.items():
        data["weight"] = round((data["counter"] / total_parishes) * 100, 2)

    return fused_districts, fused_parishes


def handle_keywords(keywords, fused_keywords, user_rating):
    # keywords are split based on the "-" char to include keywords with spaces ex. "toxic gas".
    # empty " - " inputs are also handled.
    keywords = keywords.split("-")

    # since this may need to handle multiple strings
    for keyword in keywords:
        # if there's a key
        if keyword:
            # if the key is an important default keyword calculate using the assigned weights
            if keyword in KEYWORD_WEIGHTS:
                # update counter
                fused_keywords[keyword]["counter"] += 1

                # recalculate weights
                hazard_weight = KEYWORD_WEIGHTS[keyword]
                calculated_weight = hazard_weight * (user_rating/20)

                # update weight calculation
                fused_keywords[keyword]["weight"] += calculated_weight

            # else if the key is a new/unknown/weightless word, increment the counter
            else:
                if keyword in fused_keywords:
                    # increment counter and set weight calculation to counter value
                    fused_keywords[keyword]["counter"] += 1
                    fused_keywords[keyword]["weight"] += (user_rating / 20)
                else:
                    # Create a new entry for the keyword
                    fused_keywords[keyword] = {
                        "counter": 1,
                        "weight": user_rating/20
                    }

    return fused_keywords


"""
# haversine_centroid() calcs the centroid of a cluster using the geodesic algorithm (from an existing implementation)
# haversine_distance() calcs the distance between 2 points using the geodesic algorithm (from an existing implementation)
"""


def haversine_centroid(_latitudes, _longitudes):
    radius = 6371  # Radius of the earth in km

    lat_radians = np.radians(_latitudes)
    long_radians = np.radians(_longitudes)

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


def haversine_distance(lat1, lon1, lat2, lon2):
    radius = 6371  # Radius of the earth in km

    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Calculate the differences
    d_lat = lat2 - lat1
    d_lon = lon2 - lon1

    # Apply the formula
    a = np.sin(d_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(d_lon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = radius * c

    return distance


"""
#### Folium plot - plots based on coordinates only, prints event-specific information within the markers. 
# plot_folium1() -> raw plot of all data points without any processing
# plot_folium2() -> plot all fused data points
# plot_folium3() -> plot all clustered data points
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
    # Create a map centered at the first event in the data
    first_event = next(iter(data.values()))
    map_center = [first_event.get('latitude'), first_event.get('longitude')]
    map_plot = folium.Map(location=map_center, zoom_start=6, max_width='100%')

    # Create a marker cluster layer to add things to the map.
    # disableClusteringAtZoom -> at what zoom lvl events disperse
    marker_cluster = MarkerCluster(disableClusteringAtZoom=10).add_to(map_plot)

    # Iterate over the events and add markers to the map
    for event in data.values():
        # Extract the latitude, longitude, and text location from the event
        lat, lon, text_loc = event.get('latitude'), event.get('longitude'), event.get('location')

        # Create a popup message with the event data
        # Format Age String
        incident_span = event.get('date_age')
        days = incident_span.day
        hours = incident_span.hour
        minutes = incident_span.minute

        # Create a formatted string for the incident span
        age_string = f"{days} days, {hours} hours, {minutes} minutes"

        # Create the popup message with the updated event data
        popup_text = f"Incident ID: {event.get('event_id')}<br>Latest Update: {event.get('date_latest')}<br>Incident Span: {age_string}<br>Contributor IDs: {event.get('user_ids')}<br>Contribution IDs: {event.get('sub_ids')}"

        # Display "Has Fire" and "Has Smoke"
        has_fire = "Confirmed" if event.get('fire_verified') else "Unknown"
        has_smoke = "Confirmed" if event.get('smoke_verified') else "Unknown"

        popup_text += f"<br>Fire: {has_fire}<br>Smoke: {has_smoke}"

        # Add district to the popup message
        districts = event.get('districts')
        if districts:
            popup_text += '<br><br>Incident District Probability:'
            for district, data in districts:
                weight = data["weight"]
                popup_text += f'<br>{district}: {weight}'

        # Add parish to the popup message
        parishes = event.get('parishes')
        if parishes:
            popup_text += '<br><br>Incident Parish Probability:'
            for parish, data in parishes:
                weight = data["weight"]
                popup_text += f'<br>{parish}: {weight}'

        # Add keywords to the popup message
        keywords = event.get('keywords')
        if keywords:
            popup_text += '<br><br>Possible Hazards:'
            for keyword, data in keywords:
                popup_text += f'<br>{keyword}: {data}'

        # set event icon colours
        if event.get('event_id') < 0:
            color = 'black'
        else:
            color = 'red'

        # Add the marker to the marker cluster layer
        popup = folium.Popup(popup_text, max_width=400)  # Adjust popup width
        folium.Marker([lat, lon], icon=folium.Icon(color=color), popup=popup).add_to(marker_cluster)

    # save map HTML
    map_plot.save(map_name)

    return 0


def plot_folium3(data, map_name):
    # Create a map centered at the first event in the list
    first_event = next(iter(data.values()))[0]  # get the first event in the first cluster
    map_center = [first_event[6], first_event[7]]
    map_plot = folium.Map(location=map_center, zoom_start=6)

    # Define a list of colors to use for the markers
    colors = [
        'orange', 'lightgray', 'lightblue', 'darkred', 'darkgreen', 'beige', 'pink', 'blue', 'cadetblue', 'darkpurple',
        'purple', 'lightgreen', 'gray', 'red', 'green', 'lightred', 'white', 'darkblue', 'black']

    # Create a marker cluster layer to add things to the map.
    # disableClusteringAtZoom -> at what zoom lvl events disperse
    marker_cluster = MarkerCluster(disableClusteringAtZoom=10).add_to(map_plot)

    # Iterate over the clusters and events, and add markers to the map
    for i, (cluster_id, event_list) in enumerate(data.items()):
        for event in event_list:
            # Extract the latitude, longitude, and text location from the event
            lat, lon, text_loc = event[6], event[7], event[9]

            popup_text = f"Submission IDs:: {event[0]}"
            # Determine the color for this marker based on the cluster ID
            if cluster_id == -1:
                color = 'black'
            else:
                color = colors[i % len(colors)]

            # Add the marker to the marker cluster layer, using the color for this cluster
            folium.Marker([lat, lon], popup=popup_text, icon=folium.Icon(color=color)).add_to(marker_cluster)

    map_plot.save(map_name)

    return 0


# debug function
def print_cluster_members(data):
    print("DEBUG CLUSTERS")
    for key in data:
        values = data[key]
        first_values = [v[0] for v in values]
        print(key, first_values)


if __name__ == '__main__':
    # Read data, plot all submissions in fireloc_map_raw
    data_input = read_data_file(FILE_PATH)
    plot_folium1(data_input, "fireloc_map_raw.html")

    # cluster data, plot all clusters in fireloc_map_clustered
    clustered_data, clusterer_instance = apply_clustering_algorithm(data_input)
    plot_folium3(clustered_data, "fireloc_map_clustered.html")

    # fuse data, plot all fused events in fireloc_map_fused
    fused_events = apply_data_fusion(clustered_data)
    plot_folium2(fused_events, "fireloc_map_fused.html")

    # print_cluster_members(clustered_data) # DEBUG

    data_queue = []
    while True:
        # Simulate real-time and/or manual submissions

        user_input = input("(Only one line at a time) >>  ")
        if user_input.lower() in {"quit", "q"}:
            # Quit program
            break

        else:
            # Insert inputs manually into current data

            # example input that creates a fire in Serra da Estrela:
            # 26;10/07/2023 09:30;23;17;1;0;40.334;-7.618;;;
            # 27;10/07/2023 10:00;24;18;0;1;40.335;-7.617;;;
            # 28;10/07/2023 10:30;25;20;1;0;40.336;-7.616;;;
            # 29;10/07/2023 11:00;26;15;0;1;40.318;-7.600;;;
            # 30;10/07/2023 11:30;27;19;1;0;40.350;-7.622;;;

            # example input that creates a fire cluster next to ID 25
            # 26;09/06/2023 10:00;22;17;0;1;40.185;-8.512;;;
            # 27;09/06/2023 10:00;22;17;0;1;40.185;-8.514;;;
            # 28;09/06/2023 10:00;22;17;0;1;40.187;-8.513;;;
            # 29;09/06/2023 10:00;22;17;0;1;40.186;-8.509;;;
            # 30;09/06/2023 10:00;22;17;0;1;40.186;-8.507;;;

            # example input that adds a fire submission next to cluster [13, 15]
            # 31;09/05/2023 16:30;69;1;1;1;40.239;-8.445;;;

            # parse and save new input
            parsed_input = parse_manual_input(user_input)
            # save current submissions (both original and manual inputs)
            # write_data_file(FILE_PATH, user_input)

            # quickly check if new point belongs to any existing cluster
            clustered_data, affected_label, found_existing_cluster = update_hdbscan_approximate_predict(parsed_input, clusterer_instance, clustered_data)
            if found_existing_cluster:
                # update cluster map
                plot_folium3(clustered_data, "fireloc_map_clustered.html")

                # update fused event
                fused_events[affected_label] = fuse_cluster_submissions(affected_label, clustered_data.get(affected_label))

                # update fused event map
                plot_folium2(fused_events, "fireloc_map_fused.html")

            else:
                # add point to clustering queue
                data_queue.append(parsed_input)

            # once queue is full, cluster the new points
            if len(data_queue) >= 5:
                clustered_data = update_hdbscan_fit_predict(data_queue, clusterer_instance, clustered_data)

                # print_cluster_members(clustered_data) # DEBUG
                # update cluster map
                plot_folium3(clustered_data, "fireloc_map_clustered.html")

                # re-apply data fusion to all clusters, replace old data
                fused_events = apply_data_fusion(clustered_data)
                # update fused event map
                plot_folium2(fused_events, "fireloc_map_fused.html")

                # reset queue
                data_queue = []
