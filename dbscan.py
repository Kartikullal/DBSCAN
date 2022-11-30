import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class DBSCAN():
    def __init__(self, eps = 0.5, min_samples = 4, metric = 'euclidean', dist_matrix = None):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.c = 0
        self.dist_matrix = dist_matrix
    def distFunc(self, X):
        A = X
        B = X
        if(self.metric == 'euclidean'):
            from sklearn.metrics.pairwise import euclidean_distances
            d_matrix = euclidean_distances(A,B)
        if(self.metric == 'cosine'):
            from sklearn.metrics.pairwise import cosine_distances
            d_matrix = cosine_distances(A, B)
        return d_matrix
    
    def range_query(self, q):
        neighbors = (self.dist_matrix.iloc[q] <= self.eps).index[self.dist_matrix.iloc[q] <= self.eps].tolist()

        return neighbors
    
    def fit(self, X):
        if isinstance(X, pd.DataFrame):
            self.X = X
        else:
            self.X = pd.DataFrame(X)
        self.labels_ = np.zeros((len(X)))
        self.labels_[:] = np.nan
        self.labels_[:] = np.nan
        if(isinstance(self.dist_matrix, pd.DataFrame)):
            pass
        else:
            self.dist_matrix = pd.DataFrame(self.distFunc(self.X))
            print(" Distance matrix calculated")
        for i, row in self.X.iterrows():
            if(not pd.isna(self.labels_[i])):
                continue
            neighbors = self.range_query(i)
            if(len(neighbors)< self.min_samples):
                self.labels_[i] = -1
                continue
            self.c = self.c + 1
            self.labels_[i] = self.c
            S = neighbors
            
            for q in S:
                if(self.labels_[q] == -1):
                    self.labels_[q] = self.c
                if(not pd.isna(self.labels_[q])):
                    continue
                self.labels_[q] = self.c
                neighbors = self.range_query(q)
                if(len(neighbors)>= self.min_samples):
                    S += neighbors
        