import random
import tensorflow as tf

keras = tf.keras
from keras.layers import Input, Dense, concatenate
from keras.models import Model
from keras.utils import pad_sequences

import numpy as np

from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

NUM_COORDS = 2
MAX_TEXT_LENGTH = 15
MIN_NUMBER = 1
MAX_NUMBER = 5
DBSCAN_DISTANCE = 0.1
DBSCAN_MIN_SIZE = 3


def create_cnn():
    coord_input = Input(shape=(NUM_COORDS,))
    text_input = Input(shape=(MAX_TEXT_LENGTH,))
    num_input = Input(shape=(1,))

    coord_layer = Dense(16, activation='relu')(coord_input)
    text_layer = Dense(16, activation='relu')(text_input)
    num_layer = Dense(16, activation='relu')(num_input)

    merged_layer = concatenate([coord_layer, text_layer, num_layer])

    cnn_layer = Dense(32, activation='relu')(merged_layer)
    output_layer = Dense(NUM_COORDS, activation='linear')(cnn_layer)

    model = Model(inputs=[coord_input, text_input, num_input], outputs=output_layer)

    return model


def DF_coords(coords, distance_threshold):
    db = DBSCAN(eps=distance_threshold, min_samples=DBSCAN_MIN_SIZE).fit(coords)

    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    fused_coords = np.zeros((n_clusters, 2))

    for i in range(n_clusters):
        cluster_coords = coords[labels == i]
        centroid = np.mean(cluster_coords, axis=0)
        fused_coords[i] = centroid

    return fused_coords


def apply_cnn(coords, text, num, cnn_model):
    return cnn_model.predict([coords, text, num])


def apply_cnn_coords(coords, cnn_model):
    return cnn_model.predict(coords)


# apply dbscan to the fused coords dataset
def cluster_fused_coords(dataset):
    clustering = DBSCAN(eps=DBSCAN_DISTANCE, min_samples=DBSCAN_MIN_SIZE).fit(dataset)

    return clustering.labels_


# plot results with legend & colours
def plot_results(dataset, labels):
    fig, ax = plt.subplots()

    # define custom color map
    cmap = ListedColormap(['#e41a1c', '#4daf4a', '#377eb8', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf',
                           '#999999', '#a6cee3', '#1f78b4'])

    # plot clustered points
    scatter = ax.scatter(dataset[:, 0], dataset[:, 1], c=labels, cmap=cmap)

    # create legend
    unique_labels = set(labels)
    num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    legend_labels = ['Cluster ' + str(i + 1) for i in range(num_clusters)]

    legend_elements = [Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=cmap(scatter.norm(i)),
                              markersize=10) for i, label in enumerate(legend_labels)]

    if -1 in unique_labels:
        noise_label = Line2D([0], [0], marker='o', color='w', label='Noise', markerfacecolor=cmap(0), markersize=10)
        legend_elements.insert(0, noise_label)

    legend1 = ax.legend(handles=legend_elements, loc='upper right')
    ax.add_artist(legend1)

    # set plot title and axis labels
    ax.set_title('Fused Coordinates Clustering')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # show plot
    plt.show()


# simple plot of all coordinates. uses black colour only and there's no legend
def plot_simple(dataset):
    fig, ax = plt.subplots()
    ax.scatter(dataset[:, 0], dataset[:, 1], c='k')
    ax.set_title('Original Coordinates')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()


# randomly generate "num_samples" amount of coordinates
# the dataset entries are of the type --> {[coord_x, coord_y], "text", int}
# where the coords are random, the text is also random, and the integer is a number between 1 and 5
def generate_input_data(number):
    coords = np.random.randint(50, 1001, size=(number, NUM_COORDS))

    text = np.array([choose_random_word() for _ in range(number)], dtype=object)
    print(text)

    padded_text = pad_sequences(sequences=text, maxlen=MAX_TEXT_LENGTH, dtype=object, padding="post", truncating="post", value=" ")
    print(padded_text)

    num = np.random.randint(MIN_NUMBER, MAX_NUMBER + 1, size=(number, 1))

    return coords, padded_text, num


def choose_random_word():
    words = np.array(["fire", "a fire", "smoke", "fires a a", "smokey", "fire smoke", " smoke fires", " "])

    return list(np.random.choice(words))


if __name__ == '__main__':
    num_samples = 50

    # generate random input data
    coords, text, num = generate_input_data(num_samples)

    # create CNN network
    cnn_model = create_cnn()

    # apply CNN to input data
    fused_coords = apply_cnn(coords, text, num, cnn_model)

    # cluster fused coordinates based on density
    cluster_labels = cluster_fused_coords(fused_coords)

    # plot results with legend & colours
    plot_results(fused_coords, cluster_labels)

    # simple plot black no legend
    # plot_simple(fused_coords)
    plot_simple(coords)

    # test_fused_coords = DF_coords(coords, 0.12)

    # plot_simple(test_fused_coords)
    # plot_simple(coords)
