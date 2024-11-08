import pandas as pd
from dataset_handler import DatasetHandler
import numpy as np
from sklearn.decomposition import PCA

def do_pca(num_components: int, X_train, X_test):
    """
    Given an integer number of principal components to compute,
    return the sklearn PCA object and the transformed X_train
    and X_test values.
    """
    pca = PCA(n_components=num_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    return pca, X_train_pca, X_test_pca

if __name__ == "__main__":

    targets = ['Access time (ns)', 'Cycle time (ns)',
            'Total dynamic read energy per access (nJ)', 'Total dynamic write energy per access (nJ)', 'Total leakage power of a bank (mW)']

    dataset = DatasetHandler()
    X_train, X_test, y_train, y_test = dataset.make_training_data(targets, 42, False)

    # Actually apply the PCA

    pca, X_train_pca, X_test_pca = do_pca(5, X_train, X_test)

    # Let's check what the explained variance is...
    # Access explained variance
    explained_variance = pca.explained_variance_ratio_
    print(explained_variance)

    import matplotlib.pyplot as plt

    # Plot cumulative explained variance
    plt.plot(np.cumsum(explained_variance))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.show()

