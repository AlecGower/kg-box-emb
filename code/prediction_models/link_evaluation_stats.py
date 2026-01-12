import os
import pickle
import pandas as pd
import seaborn as sns
from itertools import groupby
from scipy.stats import mannwhitneyu
from matplotlib import pyplot as plt

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "link_eval_dir", type=str, help="Directory containing link evaluation data"
)
args = parser.parse_args()
link_eval_dir = args.link_eval_dir

with open(os.path.join(link_eval_dir, "link_eval_data.pkl"), "rb") as fi:
    data = pickle.load(fi)

distances = {
    k: [t[1]["distance"] for t in g]
    for k, g in groupby(data["distances"].items(), key=lambda p: p[0][0][1])
}
constrained_random_distances = {
    k: [t[1]["distance"] for t in g]
    for k, g in groupby(
        data["constrained_random_distances"].items(), key=lambda p: p[0][0][1]
    )
}
random_distances = {
    k: [t[1]["distance"] for t in g]
    for k, g in groupby(data["random_distances"].items(), key=lambda p: p[0][0][1])
}

for edge in distances:
    u = mannwhitneyu(distances[edge], random_distances[edge])
    n1 = len(distances[edge])
    n2 = len(random_distances[edge])
    print(
        f"{edge:<10} - real vs. random      - {u} - Sample sizes: real {n1}, random      {n2}"
    )

print()

for edge in distances:
    u = mannwhitneyu(distances[edge], constrained_random_distances[edge])
    n1 = len(distances[edge])
    n2 = len(constrained_random_distances[edge])
    print(
        f"{edge:<10} - real vs. constrained - {u} - Sample sizes: real {n1}, constrained {n2}"
    )


dfdata = [('true', t, d) for t, l in distances.items() for d in l]
dfdata.extend([('constrained', t, d)
              for t, l in constrained_random_distances.items() for d in l])
dfdata.extend([('random', t, d)
              for t, l in random_distances.items() for d in l])

df = pd.DataFrame(dfdata, columns=["source", "edgeType", "distance"])

# Plot box plots for distributions for each edge type
plt.figure(figsize=(24, 24))
sns.set(style="whitegrid")
# Smaller circles for outliers
sns.boxplot(x="edgeType", y="distance", hue="source", data=df, fliersize=2)
plt.title("Link Distance Distributions by Edge Type")
plt.xlabel("Edge Type")
# Rotated x-axis labels to vertical for better readability
plt.xticks(rotation=90)
plt.ylabel("Distance")
plt.legend(title="Source")
plt.tight_layout()
plt.savefig(os.path.join(link_eval_dir, "link_distance_distributions.png"))
plt.close()

# Different figure which is essentialy the same but each box
# gets its own subplot for better visibility

# Create figure with subplots
num_edge_types = len(df['edgeType'].unique())
fig, axes = plt.subplots(num_edge_types, 1, figsize=(8, num_edge_types * 4), sharex=True)
sns.set(style="whitegrid")
for ax, (edge_type, group_data) in zip(axes, df.groupby('edgeType')):
    sns.boxplot(x="source", y="distance", hue="source", data=group_data, fliersize=2, ax=ax)
    ax.set_title(f"Link Distance Distribution for Edge Type: {edge_type}")
    ax.set_xlabel("Source")
    ax.set_ylabel("Distance")
    # Create a legend only for the first subplot
    if ax == axes[0]:
        plt.legend(title="Source")
    else:
        ax.get_legend().remove()
plt.tight_layout()
plt.savefig(os.path.join(link_eval_dir, "link_distance_distributions_separate.png"))
plt.close()