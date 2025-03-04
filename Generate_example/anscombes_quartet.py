import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np

# Load Anscombe's quartet dataset
anscombe = sns.load_dataset("anscombe")

# Create a grid of plots
plt.figure(figsize=(12, 8))

# Iterate through the unique dataset identifiers in Anscombe's quartet
for i, dataset in enumerate(anscombe['dataset'].unique(), 1):
    # Select data for each dataset
    subset = anscombe[anscombe['dataset'] == dataset]

    # Create a subplot for each dataset
    plt.subplot(2, 2, i)
    plt.xlim(2.5, 20)
    plt.ylim(3, 13)
    sns.scatterplot(data=subset, x='x', y='y')
    plt.title(f"Dataset {dataset}")
    plt.xlabel("X")
    plt.ylabel("Y")

    # Fit a linear regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        subset['x'], subset['y'])

    plt.plot(np.arange(0, 21), slope *
             np.arange(0, 21) + intercept, color='red')

# Adjust layout
plt.tight_layout()

plt.savefig('anscombes_quartet.svg', format='svg')
plt.show()
