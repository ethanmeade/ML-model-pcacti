import pandas as pd
from dataset_handler import DatasetHandler
import numpy as np

targets = ['Access time (ns)', 'Cycle time (ns)',
         'Total dynamic read energy per access (nJ)', 'Total dynamic write energy per access (nJ)', 'Total leakage power of a bank (mW)']

dataset = DatasetHandler()
X_train, X_test, y_train, y_test = dataset.make_training_data(targets, 42, False)

# Actually apply the PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

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

