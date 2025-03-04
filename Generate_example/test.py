import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Create a grid with 2 rows and 4 columns
fig = plt.figure(constrained_layout=True, figsize=(12, 8))
gs = gridspec.GridSpec(4, 4, figure=fig)

# 11: Individual graph
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot([1, 2, 3], [4, 5, 6])
ax1.set_title("Graph 11")

# 12: Individual graph
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot([1, 2, 3], [6, 5, 4])
ax2.set_title("Graph 12")

# 21 and 22: Single graph spanning 2 columns
ax3 = fig.add_subplot(gs[1, 0:2])
ax3.plot([1, 2, 3], [4, 7, 2])
ax3.set_title("Graph 21 and 22")

# 31 and 32: Single graph spanning 2 columns
ax4 = fig.add_subplot(gs[2, 0:2])
ax4.plot([1, 2, 3], [3, 8, 1])
ax4.set_title("Graph 31 and 32")

# 41 and 42: Single graph spanning 2 columns
ax5 = fig.add_subplot(gs[3, 0:2])
ax5.plot([1, 2, 3], [7, 2, 5])
ax5.set_title("Graph 41 and 42")

# Display the plot
plt.show()
