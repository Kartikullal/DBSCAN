import numpy as np
import pandas as pd
from dbscan import DBSCAN
from metrics import external_evaluation, retrieve_number_labels, print_clusters

if __name__ == '__main__':
    print('Reading Cirlces Data')

    df = pd.read_csv('data/circle.csv')

    clustering = DBSCAN(eps = 0.2, min_samples= 4)
    clustering.fit(df)

    print_clusters(df, clustering.labels_, 'circles')

    print('Reading Blobs Data')

    df = pd.read_csv('data/blobs.csv')

    clustering = DBSCAN(eps = 0.2, min_samples= 4)
    clustering.fit(df)

    print_clusters(df, clustering.labels_, 'blobs')

    print('Reading Moons Data')

    df = pd.read_csv('data/moons.csv')

    clustering = DBSCAN(eps = 0.2, min_samples= 4)
    clustering.fit(df)

    print_clusters(df, clustering.labels_, 'moons')