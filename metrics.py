import numpy as np
import warnings 
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


def external_evaluation(true_labels, number_labels, metric):
    from sklearn.metrics import confusion_matrix
    
    m = confusion_matrix(true_labels, number_labels)
    
    N = np.sum(m, axis = 1)
    
    M = np.sum(m, axis = 0)
    
    if(metric == 'purity'):
        P = np.max(m , axis = 1)
        purity = np.sum(P) / np.sum(M)
        return purity
    
    elif(metric == 'gini' ):
        G_j = 1 - np.sum((np.nan_to_num(np.square(m/M))),axis = 0)
        G_avg = np.sum(G_j * M)/np.sum(M)
        return G_avg
    
    else:
        return "Invalid Metric"

def retrieve_number_labels(cluster_labels,y_train):
    # Initializing
    reference_labels = {}
    # For loop to run through each label of cluster label
    for i in np.unique(cluster_labels):
 
        index = np.where(cluster_labels == i,1,0)
        num = np.bincount(y_train[index==1]).argmax()
        reference_labels[i] = num

    number_labels = np.random.rand(len(cluster_labels))

    for i in range(len(cluster_labels)):
        number_labels[i] = reference_labels[cluster_labels[i]]
    return number_labels


def print_clusters(df, labels, fig_name):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = ax.scatter(df[df.columns[0]],df[df.columns[1]],c=labels,s=50)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(scatter)

    plt.show()
    path = 'figures/'+fig_name+'.pdf'
    print(path)
    fig.savefig(path)