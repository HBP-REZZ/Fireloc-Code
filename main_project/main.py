"""
#### Import Libs
"""

import time

import folium

from folium.plugins import MarkerCluster

from collections import defaultdict
import datetime
import numpy as np

import hdbscan


"""
#### Define MACROS and Globals
"""

FILE_PATH = "C:\\Users\\Hugo\\Desktop\\main_project\\october_fires.txt"  # data_file2  october_fires october_fires_first_100
ITER_FILE_PATH = "C:\\Users\\Hugo\\Desktop\\main_project\\october_fires.txt"

BATCH_SIZE = 50  # how many lines to read every batch

NOISE_DISTANCE_THRESHOLD = 1.5  # +NOISE_DISTANCE_THRESHOLD --> less noise but more isolated outliers will merge with normal clusters

DECAY_FACTOR = 0.7  # +DECAY_FACTOR --> weight will decrease faster --> 0.5 should result in 50% decay in 24h
MAXIMUM_DECAY = 0.2  # stops decay at 20%. dont use zero as a value

CUSTOM_DATE = datetime.datetime.strptime('05/06/2017 16:00', '%d/%m/%Y %H:%M')  # 01/06/2017 00:01 is the oldest october_fires_100 entry , 05/06/2017 15:34 is the earliest
USING_CUSTOM_DATE = False  # True False

ASSIGN_GREEN = False  # if you want to label events as inactive (colours them green)
REMOVE_GREEN = False  # if you want inactive events to be removed from the map
INACTIVITY_THRESHOLD = 1.5 * 7 * 24 * 60  # how long before an event is deemed inactive 1.5 weeks = 1.5 * 7 days x 24h x 60m

DATA_ITERATIONS = []

KEYWORD_WEIGHTS = {
    # SIGNS
    "fumo": 1,
    "fogo": 2,
    "chamas": 2,
    # FUELS
    "explosivo": 5,
    "fertilizante": 4,
    "químico": 5,
    "gas": 5,
    "gasolina": 5,
    "petroleo": 5,
    # HOUSES & PEOPLE
    "urbana": 3,
    "urbanização": 4,
    "casa": 3,
    "populacao": 3,
    "hospital": 5,
    # FIRE RELATED
    "ignicao": 3,
    "combustivel": 4,
    "propagacao": 2,
    "queimada": 3,
    "rescaldo": 2,
    "incendio": 1
}

PRIORITY_HAZARDS = []

for key, value in KEYWORD_WEIGHTS.items():
    if value > 4:
        PRIORITY_HAZARDS.append(key)

"""
#### Read input txt file. 
expected input format: sub_id;date/time;user_id;user_rating;fire_verified;smoke_verified;lat;lon;text_district;text_parish;text_keywords
expected output format: (0, datetime.datetime(2023, 3, 28, 10, 15), 1, 17, 1, 0, 40.151, -8.855, 'Coimbra', 'Figueira da Foz', 'Gasolina-Urbanizacao-Apartamentos'
calls encoder functions to create the binary array for keywords. keywords should be separated by - in the input
date ignores seconds
"""


def read_data_file(file_path):
    with open(file_path, 'r') as rf:
        lines = rf.readlines()[1:]  # skip the first line

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
    with open(file_path, 'a') as wf:
        wf.write(data_string + "\n")


def fetch_next_batch(file_data, start_line, end_line):
    data_batch = []
    # Get the lines within the specified range
    batch = file_data[start_line:end_line]

    # Process the batch of lines
    for line in batch:
        # Partition the line's data as needed
        line = line.strip().split(';')
        submission_id = int(line[0])

        submission_date = datetime.datetime.strptime(line[1], '%d/%m/%Y %H:%M')

        # THIS SHOULD UPDATE CURRENT CUSTOM DATE TO THE NEWEST INPUTTED DATE
        global CUSTOM_DATE
        CUSTOM_DATE = submission_date

        user_id = int(line[2])
        user_rating = int(line[3])
        fire_verified = int(line[4])
        smoke_verified = int(line[5])
        lat = float(line[6])
        lon = float(line[7])
        text_district = line[8]
        text_parish = line[9]
        text_keywords = line[10]

        # Add the line's data to the data_batch list
        data_batch.append((submission_id, submission_date, user_id, user_rating, fire_verified, smoke_verified,
                           lat, lon, text_district, text_parish, text_keywords))

    return data_batch


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
    c_instance = hdbscan.HDBSCAN(min_samples=1, min_cluster_size=2,
                                 metric='haversine', prediction_data=True, allow_single_cluster=True)
    clustering_results = c_instance.fit(coordinates)

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

    return clusters, c_instance


"""
update_hdbscan_approximate_predict will check if any new points belong to existing clusters. 
this is an extremely light operation. 
any points that dont belong to existing clusters will be labeled as noise (-1)
these are added to the data_queue and then wait for the main re-clustering operation
"""


def update_hdbscan_approximate_predict(data_point, c_instance, clusters):
    label = -1
    # get the x and y coordinates of the new data point
    coordinates = np.array([(data_point[6], data_point[7])])

    # get the predicted labels for the new data point using approximate_predict method
    new_labels, strengths = hdbscan.approximate_predict(c_instance, coordinates)

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
#### Fuse all known data clusters
# apply_data_fusion() iterates over all clusters and sends each individual cluster to the fusion function. 
# these are processed one at a time
# returns all fused clusters - a dictionary of dictionaries
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
    if clusters.get(-1):
        handle_noise_submissions(clusters.get(-1), fused_clusters)

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
    haversine_weights = []
    fused_latitudes = []
    fused_longitudes = []
    fused_districts = defaultdict(lambda: {"counter": 0, "percentage": 0})
    fused_parishes = defaultdict(lambda: {"counter": 0, "percentage": 0})
    fused_keywords = defaultdict(lambda: {"counter": 0, "weight": 0, "weight_percentage": 0})
    fused_fire_verified = 0
    fused_smoke_verified = 0

    # Loop over each submission in the cluster
    for submission in cluster_members:
        # Extract data from submission
        sub_id, date, user_id, user_rating, fire_verified, smoke_verified, latitude, longitude, district, parish, keywords = submission

        # Calculate the decay weight for the current submission
        time_decay = handle_time_decay(date)
        rating_decay = handle_user_rating_weight(user_rating)
        weight = time_decay * rating_decay

        # Weights to be used for the coordinate fusion
        haversine_weights.append(weight)

        # Add numeric data to lists
        fused_dates.append(date)  # for timeseries and time progression
        fused_user_ids.append(user_id)  # ID info
        fused_sub_ids.append(sub_id)  # ID info
        fused_ratings.append(user_rating)  # This info is handed by fireloc
        fused_latitudes.append(latitude)  # This info is handed by fireloc
        fused_longitudes.append(longitude)  # This info is handed by fireloc

        # Update fire/smoke verified flags if one of the fusion members has positive flags. This info is handed by fireloc
        fused_fire_verified |= fire_verified
        fused_smoke_verified |= smoke_verified

        # fill in dictionaries with the number of occurrences of districts and parishes
        fused_districts, fused_parishes = handle_locations(district, parish, fused_districts, fused_parishes, weight)

        # fill in dictionaries with the number of occurrences of keywords vs their set weights
        fused_keywords = handle_keywords(keywords, fused_keywords, weight)

    # Calculate centroid of the coordinates in the numpy arrays
    centroid_latitude, centroid_longitude = weighted_haversine_centroid(fused_latitudes, fused_longitudes, haversine_weights)

    # Calculate last update
    if USING_CUSTOM_DATE:
        inactivity = (CUSTOM_DATE - max(fused_dates)).total_seconds() / 60

    else:
        inactivity = (datetime.datetime.now() - max(fused_dates)).total_seconds() / 60

    # Calculate event hazard level based on the keywords it has - only affects map colour
    event_hazard_level = calculate_hazard_level(fused_keywords, fused_fire_verified, fused_smoke_verified, inactivity)

    if USING_CUSTOM_DATE:
        date_age = CUSTOM_DATE - min(fused_dates)
    else:
        date_age = datetime.datetime.now() - min(fused_dates)

    # Create fused event
    fused_event = {
        'event_id': cluster_id,
        'event_hazard_level': event_hazard_level,
        'date_latest': max(fused_dates),
        'date_last_update': inactivity,
        'date_age': date_age,
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
This process is similar to fuse_cluster_submissions(cluster_id, cluster_members), but only handles the noise (-1) cluster
in this cluster, each submission is meant to be seen as a 1-member "cluster", and so, creates a 1-submission event.
Most of the process is the same, but anything that involved multiple values is simplified:
    averages are not calculated and their values become the ones of the single submission
    the same happens with booleans and time variables
    keywords and other text variables remain the same
    >IMP> IDs of these events are generated starting at -1 and are always negative. this is to simplify the process of finding which events are isolated. 
"""


def handle_noise_submissions(cluster_members, fused_clusters):
    # Loop over each submission in the cluster
    for i, submission in enumerate(cluster_members):
        sub_id, date, user_id, user_rating, fire_verified, smoke_verified, latitude, longitude, district, parish, keywords = submission

        # Calculate the decay weight for the current submission
        time_decay = handle_time_decay(date)
        rating_decay = handle_user_rating_weight(user_rating)
        weight = time_decay * rating_decay

        fused_districts = defaultdict(lambda: {"counter": 0, "percentage": 0})
        fused_parishes = defaultdict(lambda: {"counter": 0, "percentage": 0})
        fused_keywords = defaultdict(lambda: {"counter": 0, "weight": 0, "weight_percentage": 0})

        # fill in dictionaries with the number of occurrences of districts and parishes
        fused_districts, fused_parishes = handle_locations(district, parish, fused_districts, fused_parishes, weight)

        # fill in dictionaries with the number of occurrences of keywords vs their set weights
        fused_keywords = handle_keywords(keywords, fused_keywords, weight)

        # Calculate last update
        if USING_CUSTOM_DATE:
            inactivity = (CUSTOM_DATE - date).total_seconds() / 60

        else:
            inactivity = (datetime.datetime.now() - date).total_seconds() / 60

        # Calculate event hazard level based on the keywords it has - only affects map colour
        event_hazard_level = calculate_hazard_level(fused_keywords, fire_verified, smoke_verified, inactivity)

        if USING_CUSTOM_DATE:
            date_age = CUSTOM_DATE - date
        else:
            date_age = datetime.datetime.now() - date

        # Create 1-submission event
        fused_event = {
            'event_id': -(i + 1),  # Assign negative ID for each noise event
            'event_hazard_level': event_hazard_level,
            'date_latest': date,
            'date_last_update': inactivity,
            'date_age': date_age,
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


########## AUX functions for Data Fusion ##########


# TODO simple functions that converts format [1, 20] to [0,1] 
def handle_user_rating_weight(user_rating):
    # Formula A (Low decay) --> 15/20 = 0.875  10/20 = 0.75 5/20 = 0.625 ...
    # result = ((user_rating - 10) / (20 - 10)) * (1 - 0.75) + 0.75
    # Formula B (Standard decay) --> 15/20 = 0.75  10/20 = 0.5 5/20 = 0.25 ...
    result = user_rating / 20
    # Formula C (High decay) --> 15/20 = 0.56  10/20 = 0.26 5/20 = 0.09 ...
    # result = ((user_rating - 1) / (20 - 1))**2 * (1 - 0.05) + 0.05
    
    return result


"""
#### Handle Time Weight Decay
# the newer the submission the higher the weight should be in the event fusion process. 
# calculates a weight to then use in the data fusion process
# expected usage: multiply a value 0-1 with a submissions values
# expected result: a lifespan of 24hours should result in approximately a 50% decay (1440 mins -> 0.51 decay) with a DECAY_FACTOR of 0.5
# maximum decay is capped at up to 80% of initial value being removed
"""


def handle_time_decay(date_value):
    # Calculate total lifespan in minutes of the current submission
    if USING_CUSTOM_DATE:
        lifespan = (CUSTOM_DATE - date_value).total_seconds() / 60
    else:
        lifespan = (datetime.datetime.now() - date_value).total_seconds() / 60

    # Calculate the decay value
    nr_minutes_per_day = 24 * 60
    decay_value = (1 - DECAY_FACTOR) ** (lifespan / nr_minutes_per_day)

    # Cap maximum decay at MAXIMUM_DECAY
    decay_value = max(MAXIMUM_DECAY, decay_value)

    return decay_value


"""
# handle_locations() and handle_keywords() go over all strings.
# for each non-empty string, a dictionary entry is created with the number of occurrences of said string
# these functions receive a dictionary entry that may or may not be empty, 
# and return an updated version of this same entry.
#
# locations and keywords use the weight variable in different ways. 
    locations uses it AS the counter variable to then calculate a likelihood of that location being the correct one
    keywords use it (along with a counter and dictionary of keywords) to calculate the weight of a certain keyword towards an hazard threshold
"""


def handle_locations(district, parish, fused_districts, fused_parishes, weighted_counter):
    # Ignore empty entries and spaces
    district = district.strip()
    parish = parish.strip()

    # Count the districts and parishes that appear within submissions if they exist within the given submission
    # .... for the district-related calculations:
    if district:
        # calculate counter
        if district in fused_districts:
            # increment counter and set weight calculation to counter value
            fused_districts[district]["counter"] += weighted_counter
            fused_districts[district]["percentage"] += 0
        else:
            # Create a new entry for the keyword
            fused_districts[district] = {
                "counter": weighted_counter,
                "percentage": 0
            }

    # Calculate the total counter value for districts
    total_districts = sum(entry["counter"] for entry in fused_districts.values())

    # Calculate the percentage of likelihood
    for district, data in fused_districts.items():
        if total_districts != 0:  # Check for zero division
            data["percentage"] = round((data["counter"] / total_districts) * 100, 2)
        else:
            data["percentage"] = 0

    # .... for the parish-related calculations:
    if parish:
        # calculate counter
        if parish in fused_parishes:
            # increment counter and set weight calculation to counter value
            fused_parishes[parish]["counter"] += weighted_counter
            fused_parishes[parish]["percentage"] += 0
        else:
            # Create a new entry for the keyword
            fused_parishes[parish] = {
                "counter": weighted_counter,
                "percentage": 0
            }

    # Calculate the total counter value for parishes
    total_parishes = sum(entry["counter"] for entry in fused_parishes.values())

    # Calculate the percentage of likelihood
    for parish, data in fused_parishes.items():
        if total_parishes != 0:  # Check for zero division
            data["percentage"] = round((data["counter"] / total_parishes) * 100, 2)
        else:
            data["percentage"] = 0

    # return both the district and parish data
    return fused_districts, fused_parishes


def handle_keywords(keywords, fused_keywords, decay_weight):
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
                calculated_weight = hazard_weight * decay_weight

                # update weight calculation
                fused_keywords[keyword]["weight"] += calculated_weight

            # else if the key is a new/unknown/weightless word, increment the counter
            else:
                if keyword in fused_keywords:
                    # increment counter and set weight calculation to counter value
                    fused_keywords[keyword]["counter"] += 1
                    fused_keywords[keyword]["weight"] += decay_weight
                else:
                    # Create a new entry for the keyword
                    fused_keywords[keyword] = {
                        "counter": 1,
                        "weight": decay_weight
                    }

    # Calculate weight as a percentage of the distribution between known keywords
    total_weight = sum(entry["weight"] for entry in fused_keywords.values())

    for keyword in fused_keywords:
        if total_weight != 0:  # Check for zero division
            fused_keywords[keyword]["weight_percentage"] = (fused_keywords[keyword]["weight"] / total_weight) * 100
        else:
            fused_keywords[keyword]["weight_percentage"] = 0

    return fused_keywords


"""
Updates map colours depending on the keywords of the event
"""


# TODO a better version of this hazard function
def calculate_hazard_level(keywords, fire_bool, smoke_bool, inactivity):
    # if set to label events as inactive
    if ASSIGN_GREEN:
        if inactivity > INACTIVITY_THRESHOLD:
            return 0

    # otherwise continue as normal
    hazard_level = 1  # Default hazard level (LOW PRIORITY)

    if keywords:
        if fire_bool or smoke_bool:
            hazard_level = 2  # AVERAGE PRIORITY

        for keyword in keywords:
            if keyword.lower() in PRIORITY_HAZARDS:
                hazard_level = 3  # HIGH PRIORITY
                break

    return hazard_level


"""
# haversine_centroid() calcs the centroid of a cluster using the geodesic algorithm (from an existing implementation)
# weighted_haversine_centroid() is the same as haversine_centroid(), but with a weighted average. The weights are time decay multiplied by user rating 
# a submission with an age of 30min and an user rating of 19 will result in a weight of 0.95. Weights vary between [0,1]
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


def weighted_haversine_centroid(_latitudes, _longitudes, _weights):
    radius = 6371  # Radius of the earth in km

    lat_radians = np.radians(_latitudes)
    long_radians = np.radians(_longitudes)

    x = radius * np.cos(lat_radians) * np.cos(long_radians)
    y = radius * np.cos(lat_radians) * np.sin(long_radians)
    z = radius * np.sin(lat_radians)

    # Calculate the weighted sum of coordinates
    weighted_sum_x = np.sum(_weights * x)
    weighted_sum_y = np.sum(_weights * y)
    weighted_sum_z = np.sum(_weights * z)

    # Calculate the total weight
    total_weight = np.sum(_weights)

    centroid_x = weighted_sum_x / total_weight
    centroid_y = weighted_sum_y / total_weight
    centroid_z = weighted_sum_z / total_weight

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
# plot_folium1() -> raw plot of all data points without any processing, for debug only
# plot_folium2() -> plot all fused data points
# plot_folium3() -> plot all clustered data points, for debug only
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

    return


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
        # Extract days
        days = incident_span.days
        # Calculate remainder hours and minutes
        hours, remainder = divmod(incident_span.seconds, 3600)
        minutes, _ = divmod(remainder, 60)

        # Create a formatted string for the incident span
        age_string = f"{days} days, {hours} hours, {minutes} minutes"

        inactivity = event.get('date_last_update')
        # Convert inactivity to timedelta object
        inactivity_timedelta = datetime.timedelta(minutes=inactivity)

        # Extract days, hours, and minutes from inactivity_timedelta
        days = inactivity_timedelta.days
        hours, remainder = divmod(inactivity_timedelta.seconds, 3600)
        minutes, _ = divmod(remainder, 60)

        # Create the formatted string for inactivity
        inactivity_string = f"{days} days, {hours} hours, {minutes} minutes"

        # if too many IDs are added on top of the rest of the text, the popup message will likely overflow and close itself.
        # comment the if's if you want to print all IDs regardless
        user_ids = event.get('user_ids')
        sub_ids = event.get('sub_ids')
        if isinstance(user_ids, list) and len(user_ids) > 15:
            user_ids = user_ids[:15]
            user_ids[-1] = "..."
        if isinstance(sub_ids, list) and len(sub_ids) > 15:
            sub_ids = sub_ids[:15]
            sub_ids[-1] = "..."

        # Create the popup message with the updated event data
        popup_text = f"Incident ID: {event.get('event_id')}<br><br>Latest Update: {event.get('date_latest')}<br>Time Since Update: {inactivity_string}<br>Incident Span: " \
                     f"{age_string}<br><br>Contributor IDs: {user_ids}<br>Contribution IDs: {sub_ids}"

        # Display "Has Fire" and "Has Smoke"
        has_fire = "Confirmed" if event.get('fire_verified') else "Unknown"
        has_smoke = "Confirmed" if event.get('smoke_verified') else "Unknown"

        popup_text += f"<br><br>Fire: {has_fire}<br>Smoke: {has_smoke}"

        ############# Add district to the popup message
        districts = event.get('districts')
        if districts:
            # Sort by percentage in descending order
            districts = sorted(districts, key=lambda x: x[1]["percentage"], reverse=True)

            popup_text += '<br><br>Distribution of Submitted Districts:'
            for district, data in districts:
                percentage = data["percentage"]
                popup_text += f'<br> - {district}: {percentage:.1f}%'
        else:
            popup_text += '<br><br>Distribution of Incident Districts: <br>Unknown'

        ############# Add parish to the popup message
        parishes = event.get('parishes')
        if parishes:
            # Sort by percentage in descending order
            parishes = sorted(parishes, key=lambda x: x[1]["percentage"], reverse=True)

            popup_text += '<br><br>Distribution of Submitted Parishes:'
            for parish, data in parishes:
                percentage = data["percentage"]
                popup_text += f'<br> - {parish}: {percentage:.1f}%'
        else:
            popup_text += '<br><br>Distribution of Incident Parishes: <br>Unknown'

        ############# Add keywords to the popup message
        keywords = event.get('keywords')
        if keywords:
            # Sort by percentage in descending order
            keywords = sorted(keywords, key=lambda x: x[1]["weight_percentage"], reverse=True)

            popup_text += '<br><br>Distribution of Submitted Hazards:'
            for keyword, data in keywords:
                counter = data.get('counter')
                percentage = data.get('weight_percentage')
                popup_text += f'<br> - {keyword}:<br>&emsp;Submissions: {counter}<br>&emsp; Weight: {percentage:.1f}%'
        else:
            popup_text += '<br><br>Distribution of Incident Hazards: <br>Unknown'

        ############# set event icon colours
        if event.get('event_id') < 0:
            # if isolated event
            popup_text += '<br><br>WARNING: <br>Isolated Incident - May be inaccurate'
            color = 'black'
        else:
            # if extremely hazardous event
            if event.get('event_hazard_level') == 3:
                color = 'red'

            # if average hazard event
            elif event.get('event_hazard_level') == 2:
                color = 'orange'

            # if lightly hazardous/low information event
            elif event.get('event_hazard_level') == 1:
                color = 'beige'

            # if event is old and has become redundant
            elif event.get('event_hazard_level') == 0 and ASSIGN_GREEN:
                color = 'green'

            # there shouldnt be a case where this else is reached
            else:
                color = 'beige'

        if REMOVE_GREEN and color == 'green':
            continue

        # Add the marker to the marker cluster layer
        popup = folium.Popup(popup_text, max_width=400)  # Adjust popup width
        folium.Marker([lat, lon], icon=folium.Icon(color=color), popup=popup).add_to(marker_cluster)

    # save map HTML
    map_plot.save(map_name)

    return


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

            popup_text = f"Submission ID:: {event[0]}<br><br>Cluster ID: {cluster_id}"
            # Determine the color for this marker based on the cluster ID
            if cluster_id == -1:
                color = 'black'
            else:
                color = colors[i % len(colors)]

            # Add the marker to the marker cluster layer, using the color for this cluster
            folium.Marker([lat, lon], popup=popup_text, icon=folium.Icon(color=color)).add_to(marker_cluster)

    map_plot.save(map_name)

    return


# debug functions
def print_cluster_members(data):
    print("DEBUG")
    for pt in data:
        values = data[pt]
        first_values = [v[0] for v in values]
        print(pt, first_values)


def print_fused_events(data):
    print("DEBUG")
    for eve in data:
        values = data[eve]
        print(eve, values)


# TODO this function needs work. the events wont refresh on the map unless you click in the layer control box after clicking next, and idk why
# TODO and the button custom component also appears in the control box, and i cant remove or hide it...
# TODO everything else seems to work properly
def plot_folium_iterative(datasets, map_name):
    # Create a map centered at the first event in the first dataset
    first_dataset = datasets[0]
    first_event = next(iter(first_dataset.values()))
    map_center = [first_event.get('latitude'), first_event.get('longitude')]
    map_plot = folium.Map(location=map_center, zoom_start=8, max_width='100%')

    # Create a marker cluster layer to add things to the map.
    # disableClusteringAtZoom -> at what zoom lvl events disperse
    marker_cluster = MarkerCluster(disableClusteringAtZoom=5)

    # Create a list to store the feature groups
    feature_groups = []

    # Iterate over the datasets
    for dataset_index, dataset in enumerate(datasets):
        # Create a unique ID for the feature group
        feature_group_id = f"feature_group_{dataset_index}"

        # Create a FeatureGroup for the current dataset
        feature_group = folium.FeatureGroup(name=f"Dataset {dataset_index + 1}", show=(dataset_index == 0), id=feature_group_id)
        feature_groups.append(feature_group)

        # Iterate over the events in the dataset
        for event_index, event in enumerate(dataset.values()):
            # Extract the latitude, longitude, and text location from the event
            lat, lon, text_loc = event.get('latitude'), event.get('longitude'), event.get('location')

            # Create a popup message with the event data
            # Format Age String
            incident_span = event.get('date_age')
            # Extract days
            days = incident_span.days
            # Calculate remainder hours and minutes
            hours, remainder = divmod(incident_span.seconds, 3600)
            minutes, _ = divmod(remainder, 60)

            # Create a formatted string for the incident span
            age_string = f"{days} days, {hours} hours, {minutes} minutes"

            inactivity = event.get('date_last_update')
            # Convert inactivity to timedelta object
            inactivity_timedelta = datetime.timedelta(minutes=inactivity)

            # Extract days, hours, and minutes from inactivity_timedelta
            days = inactivity_timedelta.days
            hours, remainder = divmod(inactivity_timedelta.seconds, 3600)
            minutes, _ = divmod(remainder, 60)

            # Create the formatted string for inactivity
            inactivity_string = f"{days} days, {hours} hours, {minutes} minutes"

            # if too many IDs are added on top of the rest of the text, the popup message will likely overflow and close itself.
            # comment the if's if you want to print all IDs regardless
            user_ids = event.get('user_ids')
            sub_ids = event.get('sub_ids')
            if isinstance(user_ids, list) and len(user_ids) > 15:
                user_ids = user_ids[:15]
                user_ids[-1] = "..."
            if isinstance(sub_ids, list) and len(sub_ids) > 15:
                sub_ids = sub_ids[:15]
                sub_ids[-1] = "..."

            # Create the popup message with the updated event data
            popup_text = f"Incident ID: {event.get('event_id')}<br><br>Latest Update: {event.get('date_latest')}<br>Time Since Update: {inactivity_string}<br>Incident Span: " \
                         f"{age_string}<br><br>Contributor IDs: {user_ids}<br>Contribution IDs: {sub_ids}"

            # Display "Has Fire" and "Has Smoke"
            has_fire = "Confirmed" if event.get('fire_verified') else "Unknown"
            has_smoke = "Confirmed" if event.get('smoke_verified') else "Unknown"

            popup_text += f"<br><br>Fire: {has_fire}<br>Smoke: {has_smoke}"

            ############# Add district to the popup message
            districts = event.get('districts')
            if districts:
                # Sort by percentage in descending order
                districts = sorted(districts, key=lambda x: x[1]["percentage"], reverse=True)

                popup_text += '<br><br>Distribution of Submitted Districts:'
                for district, data in districts:
                    percentage = data["percentage"]
                    popup_text += f'<br> - {district}: {percentage:.1f}%'
            else:
                popup_text += '<br><br>Distribution of Incident Districts: <br>Unknown'

            ############# Add parish to the popup message
            parishes = event.get('parishes')
            if parishes:
                # Sort by percentage in descending order
                parishes = sorted(parishes, key=lambda x: x[1]["percentage"], reverse=True)

                popup_text += '<br><br>Distribution of Submitted Parishes:'
                for parish, data in parishes:
                    percentage = data["percentage"]
                    popup_text += f'<br> - {parish}: {percentage:.1f}%'
            else:
                popup_text += '<br><br>Distribution of Incident Parishes: <br>Unknown'

            ############# Add keywords to the popup message
            keywords = event.get('keywords')
            if keywords:
                # Sort by percentage in descending order
                keywords = sorted(keywords, key=lambda x: x[1]["weight_percentage"], reverse=True)

                popup_text += '<br><br>Distribution of Submitted Hazards:'
                for keyword, data in keywords:
                    counter = data.get('counter')
                    percentage = data.get('weight_percentage')
                    popup_text += f'<br> - {keyword}:<br>&emsp;Submissions: {counter}<br>&emsp; Weight: {percentage:.1f}%'
            else:
                popup_text += '<br><br>Distribution of Incident Hazards: <br>Unknown'

            ############# set event icon colours
            if event.get('event_id') < 0:
                # if isolated event
                popup_text += '<br><br>WARNING: <br>Isolated Incident - May be inaccurate'
                color = 'black'
            else:
                # if extremely hazardous event
                if event.get('event_hazard_level') == 3:
                    color = 'red'
                # if average hazard event
                elif event.get('event_hazard_level') == 2:
                    color = 'orange'
                # if lightly hazardous/low information event
                elif event.get('event_hazard_level') == 1:
                    color = 'beige'
                # if event is old and has become redundant
                elif event.get('event_hazard_level') == 0 and ASSIGN_GREEN:
                    color = 'green'
                else:
                    color = 'beige'

            if REMOVE_GREEN and color == 'green':
                continue

            # Add markers to the feature group
            popup = folium.Popup(popup_text, max_width=400)  # Adjust popup width
            folium.Marker([lat, lon], icon=folium.Icon(color=color), popup=popup).add_to(feature_group)

        # Add feature group to the map
        feature_group.add_to(map_plot)

    # Add marker cluster layer to the map
    marker_cluster.add_to(map_plot)

    # Add layer control
    control = folium.LayerControl(position='topleft')
    control.add_to(map_plot)

    # JavaScript code for iterating datasets
    callback = """<script>
            var currentIndex = 0;
            var numDatasets = {0};
            var marker_cluster = null;
            function toggleVisibility() {{
                currentIndex = (currentIndex + 1) % numDatasets;
                var control = document.querySelector(".leaflet-control-layers");
                var inputs = control.querySelectorAll("input[type='checkbox']");
                for (var i = 0; i < inputs.length; i++) {{
                    inputs[i].checked = (i === currentIndex);
                    inputs[i].dispatchEvent(new Event('change'));
                }}

                // Hide the last element in the layer control
                if (currentIndex === numDatasets - 1) {{
                    var lastCheckbox = document.getElementById("checkbox_" + (numDatasets - 1));
                    var lastLabel = lastCheckbox.parentElement;
                    lastLabel.style.display = 'none';
                }}

                // Clear the marker cluster layer
                if (marker_cluster !== null) {{
                    marker_cluster.clearLayers();
                }}

                // Add markers from the current dataset to the marker cluster layer
                var currentFeatureGroup = map_plot.getPane("feature_group_" + currentIndex);
                marker_cluster = new L.MarkerClusterGroup({{ disableClusteringAtZoom: 10 }});
                marker_cluster.addLayer(currentFeatureGroup);
                map_plot.addLayer(marker_cluster);
            }}
    </script>""".format(len(datasets))

    # JavaScript code for generic button
    button_html = """
        <button onclick="toggleVisibility();">Next</button>
    """

    # Add HTML button
    map_plot.get_root().html.add_child(folium.Element(button_html))

    # Add the callback
    map_plot.get_root().html.add_child(folium.Element(callback))

    # Save map
    map_plot.save(map_name)


"""
#### MAIN ################

Stage 1: initial processing - call data_input -> apply_clustering_algorithm -> apply_data_fusion -> plot_folium2
    this should result in the mapping of all existing submissions within the "database"
Stage 2: real time processing - within the "while loop":
  manual user input. should parse_manual_input catch something:
    A - update_hdbscan_approximate_predict succeeds - fuse_cluster_submissions re-clusters target cluster and data is remapped
    B - update_hdbscan_approximate_predict fails - add to queue and wait for more inputs. Once enough inputs are queued,
        update_hdbscan_fit_predict repeats step 1 data is remapped 
  iterative input. read X txt files every Y minutes to simulate batches of input. Process is always the same as Stage 1. This option is only intended for slideshows. Subs in bulk are expected

        
Some example inputs:
    # example input that creates a fire in Serra da Estrela:
    # 26;10/07/2023 09:30;23;17;1;0;40.334;-7.618;;;
    # 27;10/07/2023 10:00;24;18;0;1;40.335;-7.617;;;
    # 28;10/07/2023 10:30;25;20;1;0;40.336;-7.616;;;
    # 29;10/07/2023 11:00;26;15;0;1;40.318;-7.600;;;
    # 30;10/07/2023 11:30;27;19;1;0;40.350;-7.622;;;

    # example input that creates a fire cluster next to ID 25 and flips it from "isolated" to standard event type
    # 26;09/06/2023 10:00;22;17;0;1;40.185;-8.512;;;
    # 27;09/06/2023 10:00;22;17;0;1;40.185;-8.514;;;
    # 28;09/06/2023 10:00;22;17;0;1;40.187;-8.513;;;
    # 29;09/06/2023 10:00;22;17;0;1;40.186;-8.509;;;
    # 30;09/06/2023 10:00;22;17;0;1;40.186;-8.507;;;

    # example input that adds a fire submission next to cluster [13, 15] and fuses it to an existing event
    # 31;09/05/2023 16:30;69;1;1;1;40.239;-8.445;;;
    
    # example input that adds 5 very spread-out noise points
    # 26;09/06/2023 10:00;22;17;0;1;41.185;-8.712;;;
    # 27;09/06/2023 10:00;22;17;0;1;40.185;-8.814;;;
    # 28;09/06/2023 10:00;22;17;0;1;39.187;-8.913;;;
    # 29;09/06/2023 10:00;22;17;0;1;40.186;-8.409;;;
    # 30;09/06/2023 10:00;22;17;0;1;40.186;-8.307;;;

"""


if __name__ == '__main__':
    # make this value TRUE if you want to add new submissions through the console
    # make this value FALSE if you want to add new submissions iteratively through ITER_FILE_PATH
    manual_input = False  # False True
    # make this value TRUE if you want to create a new map every time submissions are re-clustered (ex. for a slideshow)
    # make this value FALSE if you want to simply override the default fusion map
    create_new_maps_on_manual_input = True  # False True

    # counters & time variables
    time_counter = time.time()
    map_counter = 0

    start_l = 100
    end_l = start_l + BATCH_SIZE

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
    # print_fused_events(fused_events) # DEBUG

    print(">> Finished processing static dataset successfully.")
    DATA_ITERATIONS.append(fused_events)

    # if using iterative input, read entire file in advance and get nr of lines
    nr_iter_lines = 0
    if not manual_input:
        with open(ITER_FILE_PATH, 'r') as f:
            iter_data = f.readlines()[1:]  # skip the first line
        nr_iter_lines = len(iter_data)

    data_queue = []
    while True:
        # Simulate real-time and/or manual submissions

        """
        MANUAL INPUT
        """
        if manual_input:
            user_input = input("(Only one line at a time) >>  ")
            if user_input.lower() in {"quit", "q"}:
                # Quit program
                break

            else:  # Insert inputs manually into current data
                # parse and save new input
                parsed_input = parse_manual_input(user_input)
                # save current submissions (both original and manual inputs)
                # write_data_file(FILE_PATH, user_input)

                # quickly check if new point belongs to any existing cluster
                clustered_data, affected_label, found_existing_cluster = update_hdbscan_approximate_predict(
                    parsed_input, clusterer_instance, clustered_data)

                if found_existing_cluster:
                    # update fused event
                    fused_events[affected_label] = fuse_cluster_submissions(affected_label,
                                                                            clustered_data.get(affected_label))
                    print(" > Added to Cluster")
                    plot_folium3(clustered_data, "fireloc_map_clustered.html")
                    # update fused event map
                    plot_folium2(fused_events, "fireloc_map_fused.html")

                else:
                    # add point to clustering queue
                    data_queue.append(parsed_input)
                    print(" > Added to Queue")

                # once queue is full, cluster the new points
                if len(data_queue) >= BATCH_SIZE:
                    # merge data queue with existing data
                    data_input = data_input + data_queue

                    # re-apply clustering to all data, replace old data
                    clustered_data, clusterer_instance = apply_clustering_algorithm(data_input)

                    # re-apply data fusion to all clusters, replace old data
                    fused_events = apply_data_fusion(clustered_data)

                    # update fused event map
                    # if manual input is meant to create a slideshow of maps, make create_new_maps_on_manual_input true
                    if create_new_maps_on_manual_input:
                        new_map = "fireloc_map_fused_" + str(map_counter) + ".html"
                        plot_folium2(fused_events, new_map)
                        map_counter += 1

                    # otherwise, if only an updated map is needed, then make create_new_maps_on_manual_input false
                    else:
                        plot_folium2(fused_events, "fireloc_map_fused.html")

                    # reset queue
                    data_queue = []
                    print(" > Flushed Queue")

        else:
            """
            ITERATIVE INPUT
            """

            """
            Beware that the fetch_next_batch function will update CUSTOM_DATE to the date of the last line of input of a batch
            to avoid errors, make sure that the batch dataset is ordered by date. 
            """

            # Check if X time has passed
            seconds = 2
            if time.time() - time_counter >= seconds:
                print("> Fetching next batch of inputs")

                # Fetch the next batch of inputs
                next_data_input = fetch_next_batch(iter_data, start_l, end_l)
                data_input += next_data_input

                # Update the start and end lines for the next batch
                start_l = end_l + 1
                end_l = start_l + BATCH_SIZE

                # Check if end_l exceeds the last line index
                if end_l >= nr_iter_lines or nr_iter_lines <= 0:
                    break

                # cluster data, plot all clusters in fireloc_map_clustered
                clustered_data, clusterer_instance = apply_clustering_algorithm(data_input)

                # fuse data, plot all fused events in fireloc_map_fused
                fused_events = apply_data_fusion(clustered_data)
                DATA_ITERATIONS.append(fused_events)

                # update fused event map
                # if manual input is meant to create a slideshow of maps, make create_new_maps_on_manual_input true
                if create_new_maps_on_manual_input:
                    new_map = "iterative_maps\\iter_" + str(map_counter) + ".html"
                    plot_folium2(fused_events, new_map)
                    map_counter += 1

                # otherwise, if only an updated map is needed, then make create_new_maps_on_manual_input false
                else:
                    plot_folium2(fused_events, "fireloc_map_fused.html")

                time_counter = time.time()

            #if map_counter == 5:
             #   break

    plot_folium_iterative(DATA_ITERATIONS, "october_fires_iterative.html")
