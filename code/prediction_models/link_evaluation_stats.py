import os, sys
import pickle
import pandas as pd
import seaborn as sns
from itertools import groupby
from scipy.stats import mannwhitneyu, bws_test, brunnermunzel
from matplotlib import pyplot as plt
import numpy as np

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "link_eval_dir", type=str, help="Directory containing link evaluation data"
)
args = parser.parse_args()
link_eval_dir = args.link_eval_dir
print(f"Loading link evaluation data from {link_eval_dir}", file=sys.stderr, flush=True)
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

table_data = []

for edge in distances:
    print(f"{edge:<10} - real vs. random", file=sys.stderr, flush=True)
    n1 = len(distances[edge])
    n2 = len(random_distances[edge])
    # Report mean and median distances
    print(f"\tMean (real): {np.mean(distances[edge]):.2e}, Median (real): {np.median(distances[edge]):.2e}, % > 0.0: {np.mean(np.array(distances[edge])>0.0)*100:.2e}%", file=sys.stderr, flush=True)
    print(f"\tMean (random): {np.mean(random_distances[edge]):.2e}, Median (random): {np.median(random_distances[edge]):.2e}, % > 0.0: {np.mean(np.array(random_distances[edge])>0.0)*100:.2e}%", file=sys.stderr, flush=True)
    # Mann-Whitney U test
    u_rand = mannwhitneyu(distances[edge], random_distances[edge])
    print(
        f"\t{u_rand} - Sample sizes: real {n1}, random {n2}", file=sys.stderr, flush=True
    )
    # # Brunner-Munzel test
    # bm_rand = brunnermunzel(distances[edge], random_distances[edge])
    # print(
    #     f"\tBM stat: {bm_rand.statistic}, p-value: {bm_rand.pvalue} - Sample sizes: real {n1}, random {n2}", file=sys.stderr, flush=True
    # )
    # # Baumgartner-Weiss-Schindler test
    # bws_rand = bws_test(distances[edge], random_distances[edge])
    # print(
    #     f"\tBWS stat: {bws_rand.statistic}, p-value: {bws_rand.pvalue} - Sample sizes: real {n1}, random {n2}", file=sys.stderr, flush=True
    # )

# print()

# for edge in distances:
    print(f"{edge:<10} - real vs. constrained", file=sys.stderr, flush=True)
    n1 = len(distances[edge])
    n2 = len(constrained_random_distances[edge])
    # Report mean and median distances
    print(f"\tMean (real): {np.mean(distances[edge]):.2e}, Median (real): {np.median(distances[edge]):.2e}, % > 0.0: {np.mean(np.array(distances[edge])>0.0)*100:.2e}%", file=sys.stderr, flush=True)
    print(f"\tMean (constrained): {np.mean(constrained_random_distances[edge]):.2e}, Median (constrained): {np.median(constrained_random_distances[edge]):.2e}, % > 0.0: {np.mean(np.array(constrained_random_distances[edge])>0.0)*100:.2e}%", file=sys.stderr, flush=True)
    # Mann-Whitney U test
    u_constr = mannwhitneyu(distances[edge], constrained_random_distances[edge])
    print(
        f"\t{u_constr} - Sample sizes: real {n1}, constrained {n2}", file=sys.stderr, flush=True 
    )
    # # Brunner-Munzel test
    # bm_constr = brunnermunzel(distances[edge], constrained_random_distances[edge])
    # print(
    #     f"\tBM stat: {bm_constr.statistic}, p-value: {bm_constr.pvalue} - Sample sizes: real {n1}, constrained {n2}", file=sys.stderr, flush=True
    # )
    # # Baumgartner-Weiss-Schindler test
    # bws_constr = bws_test(distances[edge], constrained_random_distances[edge])
    # print(
    #     f"\tBWS stat: {bws_constr.statistic}, p-value: {bws_constr.pvalue} - Sample sizes: real {n1}, constrained {n2}", file=sys.stderr, flush=True
    # )
    # # Brunner-Munzel test
    # bm_constr = brunnermunzel(distances[edge], constrained_random_distances[edge])
    # print(
    #     f"\tBM stat: {bm_constr.statistic}, p-value: {bm_constr.pvalue} - Sample sizes: real {n1}, constrained {n2}", file=sys.stderr, flush=True
    # )

    table_data.append([
        edge,
        n1,
        # n2,
        np.mean(distances[edge]),
        # np.median(distances[edge]),
        np.mean(constrained_random_distances[edge]),
        # np.median(constrained_random_distances[edge]),
        np.mean(random_distances[edge]),
        # np.median(random_distances[edge]),
        u_rand.statistic,
        u_rand.pvalue,
        # bm_rand.statistic,
        # bm_rand.pvalue,
        # bws_rand.statistic,
        # bws_rand.pvalue,
        u_constr.statistic,
        u_constr.pvalue,
        # bm_constr.statistic,
        # bm_constr.pvalue,
        # bws_constr.statistic,
        # bws_constr.pvalue,
    ])

table_headings = [
    "Edge Type",
    "Number of Test Edges",
    # "n_constrained",
    "Mean Distance (real)",
    # "median_real",
    "Mean Distance (constrained)",
    # "median_constrained",
    "Mean Distance (random)",
    # "median_random",
    "Mann-Whitney U statistic (Real vs. Random)",
    "Mann-Whitney U p-value (Real vs. Random)",
    # "bm_rand_stat",
    # "bm_rand_pval",
    # "bws_rand_stat",
    # "bws_rand_pval",
    "Mann-Whitney U statistic (Real vs. Constrained)",
    "Mann-Whitney U p-value (Real vs. Constrained)",
    # "bm_constr_stat",
    # "bm_constr_pval",
    # "bws_constr_stat",
    # "bws_constr_pval",
]

formatters = {
    "Number of Test Edges" : "{}".format,
    "Mean Distance (real)" : "{:.3e}".format,
    "Mean Distance (constrained)" : "{:.3e}".format,
    "Mean Distance (random)" : "{:.3e}".format,
    "Mann-Whitney U statistic (Real vs. Random)" : "{:.1f}".format,
    "Mann-Whitney U p-value (Real vs. Random)" : "{:.3e}".format,
    "Mann-Whitney U statistic (Real vs. Constrained)" : "{:.1f}".format,
    "Mann-Whitney U p-value (Real vs. Constrained)" : "{:.3e}".format
}

table_df = pd.DataFrame(table_data, columns=table_headings)
# pd.options.display.float_format = '{:.3e}'.format
print(table_df.to_latex(index=False, formatters=formatters), file=sys.stderr, flush=True)

# Also calculate overall statistics
all_true = [d for l in distances.values() for d in l]
all_constrained = [d for l in constrained_random_distances.values() for d in l]
all_random = [d for l in random_distances.values() for d in l]  

# Mann-Whitney U test
u = mannwhitneyu(all_true, all_random)
n1 = len(all_true)
n2 = len(all_random)
print()
print(
    f"{'overall':<10} - real vs. random      - {u} - Sample sizes: real {n1}, random      {n2}", file=sys.stderr, flush=True
)
u = mannwhitneyu(all_true, all_constrained)
n1 = len(all_true)
n2 = len(all_constrained)
print(
    f"{'overall':<10} - real vs. constrained - {u} - Sample sizes: real {n1}, constrained {n2}", file=sys.stderr, flush=True
)

# # Brunner-Munzel test
# bm = brunnermunzel(all_true, all_random)
# print()
# print(
#     f"{'overall':<10} - real vs. random      - BM stat: {bm.statistic}, p-value: {bm.pvalue}", file=sys.stderr, flush=True
# )
# bm = brunnermunzel(all_true, all_constrained)
# print(
#     f"{'overall':<10} - real vs. constrained - BM stat: {bm.statistic}, p-value: {bm.pvalue}", file=sys.stderr, flush=True
# )

# # Baumgartner-Weiss-Schindler test
# bws = bws_test(all_true, all_random)
# print()
# print(
#     f"{'overall':<10} - real vs. random      - BWS stat: {bws.statistic}, p-value: {bws.pvalue}", file=sys.stderr, flush=True
# )
# bws = bws_test(all_true, all_constrained)
# print(
#     f"{'overall':<10} - real vs. constrained - BWS stat: {bws.statistic}, p-value: {bws.pvalue}", file=sys.stderr, flush=True
# )

dfdata = [('true', t, d) for t, l in distances.items() for d in l]
dfdata.extend([('constrained', t, d)
              for t, l in constrained_random_distances.items() for d in l])
dfdata.extend([('random', t, d)
              for t, l in random_distances.items() for d in l])

df = pd.DataFrame(dfdata, columns=["source", "edgeType", "distance"])

# Plot box plots for distributions for each edge type
plt.figure(figsize=(12, 12))
sns.set(style="whitegrid")
# Smaller circles for outliers
print("Plotting boxplots", file=sys.stderr, flush=True)

edge_filter = """hasChemCellMorph
hasChemNutrientUtilization
hasChemNutrientUtilization_Decreased
hasChemStressResistance
hasChemStressResistance_Decreased
hasChemStressResistance_Increased""".splitlines()

ylabs = [
    "hasChemCell\nMorph",
    "hasChemNutrient\nUtilization",
    "hasChemNutrient\nUtilization_Decreased",
    "hasChemStress\nResistance",
    "hasChemStress\nResistance_Decreased",
    "hasChemStress\nResistance_Increased",
]

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "legend.title_fontsize": 15
})
sns.boxplot(x="distance", y="edgeType", hue="source", data=df[df['edgeType'].isin(edge_filter)], fliersize=1.5)
print("Finished boxplots.", file=sys.stderr, flush=True)
plt.title("Link Distance Distributions by Edge Type")
plt.xlabel("Distance")
plt.ylabel("Edge Type")
print("Adding legend and saving figure.", file=sys.stderr, flush=True)
plt.legend(title="Source", loc='upper right')
plt.gca().set_yticklabels(ylabs, ha="right")
# plt.gca().set_yticklabels(ylabs, rotation=-45, ha="right")
plt.gca().set_xlim(left=-0.001,right=0.010)

# ax = plt.gca()

# # --- locate the category numerically ---
# yticklabels = [t.get_text() for t in ax.get_yticklabels()]
# target_label = "hasChemStress\nResistance_Increased"

# y_idx = yticklabels.index(target_label)
# print(f"Y index of target label '{target_label}': {y_idx}", file=sys.stderr, flush=True)

# # category occupies roughly [i-0.5, i+0.5]
# ymin, ymax = y_idx - 0.45, y_idx + 0.45

# # --- create inset ---
# axins = inset_axes(
#     ax,
#     width="23%",
#     height="11%",
#     loc="lower right",
#     bbox_to_anchor=(0.08, 0.08, 1, 1),
#     bbox_transform=ax.transAxes,
#     borderpad=1
# )

# # --- THIS is the zoom window ---
# axins.set_xlim(-1.2e-6, 1.2e-6)
# axins.set_ylim(ymin, ymax)

# # reuse artists (true window)
# axins.sharex(ax)
# axins.sharey(ax)

# # formatting
# axins.set_yticks([])
# axins.set_xlabel("Distance (zoomed)", fontsize=12)
# axins.tick_params(axis="x", labelsize=11)
# axins.grid(False)

# axins.set_title(
#     "hasChemStressResistance_Increased",
#     fontsize=12,
#     pad=6
# )





plt.tight_layout()
print("Saving figure.", file=sys.stderr, flush=True)
plt.savefig(
    os.path.join(link_eval_dir, "link_distance_distributions.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.savefig(
    os.path.join(link_eval_dir, "link_distance_distributions.pdf"),
    bbox_inches="tight"
)

plt.close()

# Different figure which is essentialy the same but each box
# gets its own subplot for better visibility

# # Create figure with subplots
# df = df[(df['edgeType'] == 'RO_0000087') | (df['edgeType'] == 'RO_0002200')]
# num_edge_types = len(df['edgeType'].unique())
# fig, axes = plt.subplots(num_edge_types, 1, figsize=(8, num_edge_types * 4), sharex=True)
# sns.set(style="whitegrid")
# for ax, (edge_type, group_data) in zip(axes, df.groupby('edgeType')):
#     sns.boxplot(x="source", y="distance", hue="source", data=group_data, fliersize=2, ax=ax)
#     ax.set_title(f"Link Distance Distribution for edgeType: {edge_type}")
#     ax.set_xlabel("Source")
#     ax.set_ylabel("Distance")
#     # Create a legend only for the first subplot
#     if ax == axes[0]:
#         plt.legend(title="Source", loc='upper right')
#     else:
#         try:
#             ax.get_legend().remove()
#         except AttributeError:
#             pass
# plt.tight_layout()
# plt.savefig(os.path.join(link_eval_dir, "link_distance_distributions_separate.png"))
# plt.close()