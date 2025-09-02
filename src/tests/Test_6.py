import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Creating a sample dataset with distinct clusters
np.random.seed(42)
data1 = np.random.rand(50, 5) + 2  # Cluster 1
data2 = np.random.rand(50, 5) - 2  # Cluster 2
data = np.vstack((data1, data2))

# Perform PCA
pca = PCA(n_components=3)
principal_components = pca.fit_transform(data)

# Extract the principal components
pc1 = principal_components[:, 0]
pc2 = principal_components[:, 1]
pc3 = principal_components[:, 2]

# Create a 3D plot with increased figure size
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of the first three principal components
ax.scatter(pc1, pc2, pc3, c='blue', marker='o')

# Set labels with padding
ax.set_xlabel('Principal Component 1', labelpad=20)
ax.set_ylabel('Principal Component 2', labelpad=20)
ax.set_zlabel('Principal Component 3', labelpad=20)

# Set title
ax.set_title('3D plot of Principal Components')

# Manually adjust subplot parameters
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

# Rotate the view
ax.view_init(elev=20., azim=-35)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

