import matplotlib.pyplot as plt
from zadu import zadu
from sklearn.datasets import load_iris, load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.manifold import TSNE

# Load datasets
digits, digits_label = load_digits(return_X_y=True)

# Initialize dimensionality reduction techniques
embedding_methods = {
    "PCA": PCA(n_components=2),
    "MDS": MDS(n_components=2),
    # Fixing random state for reproducibility
    "t-SNE": TSNE(n_components=2, random_state=42)
}

# Prepare to store LCMC scores
scores_dict = {method: [] for method in embedding_methods.keys()}

# Initialize parameters
max_k = len(digits)  # Maximum value of k
k_values = range(1, max_k, 50)  # Values for k from 1 to max_k

# Iterate over each embedding method
for method, embedder in embedding_methods.items():
    print(f"Applying {method}...")

    # Apply the dimensionality reduction technique
    if method == "t-SNE":
        # t-SNE is computationally intensive; make sure to fit_transform only once for efficiency
        embedded_digits = embedder.fit_transform(digits)
    else:
        embedded_digits = embedder.fit_transform(digits)

    # Iterate over different values of k
    for k in k_values:
        print(f"Computing LCMC for {method} with k={k}")

        # Define the specification for the zadu object
        spec_list = [
            {
                "id": "lcmc",
                "params": {
                    "k": k
                }
            }
        ]

        # Create a ZADU object with the current specification for digits
        zadu_digits = zadu.ZADU(spec_list, digits, return_local=True)
        scores_digits, _ = zadu_digits.measure(embedded_digits, digits_label)

        # Store the LCMC score for digits
        local_lcmc_digits = scores_digits[0]["lcmc"]
        scores_dict[method].append(local_lcmc_digits)

# Plot the evolution of LCMC scores for each method
plt.figure(figsize=(12, 6))
for method, lcmc_scores in scores_dict.items():
    plt.plot(k_values, lcmc_scores, label=f"{method} (Digits)")

plt.title("Local Continuity Meta-Criterion (LCMC) Scores")
plt.xlabel("k value")
plt.ylabel("LCMC Score")
plt.legend()
plt.grid()
plt.savefig("lcmc_scores.png")
plt.show()
