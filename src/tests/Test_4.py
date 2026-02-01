import matplotlib.pyplot as plt
import numpy as np

matrix = np.array([[2, 1], [1, 2]])
fig, ax = plt.subplots(facecolor='lightgrey')
ax.imshow(matrix, cmap='Blues', vmin=0, vmax=2)
labels = [["TN", "FP"],
          ["FN", "TP"]]
for i in range(2):
    for j in range(2):
        ax.text(j, i, labels[i][j],
                ha="center", va="center",
                fontsize=14, fontweight="bold")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_facecolor('lightgrey')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Negative", "Positive"])
ax.set_yticklabels(["Negative", "Positive"])
ax.set_xticks(np.arange(-.5, 2, 1), minor=True)
ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
ax.grid(which="minor")
ax.tick_params(which="minor", bottom=False, left=False)
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.title("Confusion Matrix (Example)")
# fig.savefig('Confusion_Matrix_Example.png')
plt.show()
