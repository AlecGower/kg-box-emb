import sys, os
import pickle, json
from itertools import islice

from box_embeddings.parameterizations import MinDeltaBoxTensor

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE, "code", "presentation"))
sys.path.append(os.path.join(BASE, "code", "prediction_models"))

from train_boxes import plot_boxes_mpl
from boxplot2d import plot_min_delta_boxes_2d_matplotlib

with open(os.path.join(BASE, "data", "2d-with-digenic-interaction-xdicts.pkl"), "rb") as f:
    xdicts = pickle.load(f)

for d in xdicts:
    print(d.keys())

with open(os.path.join(BASE, "data", "idx_dict.json"), "r") as f:
    idx_dict = json.load(f)

for k in islice(idx_dict, 5):
    print(f"{k}, {type(k)}")

rev_class_dict = {f"{kd}-{i}" : c for kd in idx_dict.keys() for i, c in idx_dict[kd].items()}

boxes = [{f"{kd}-{i}" : MinDeltaBoxTensor.from_vector(t.detach()[i,:]).data.numpy() for kd, t in d.items() for i in range(t.shape[0])} for d in xdicts]


print(*islice(rev_class_dict.items(), 5))
print(*islice(boxes[0].items(), 5))



fig, ax = plot_boxes_mpl(
    data=None,
    rev_class_dict=rev_class_dict,
    plot_boxes=boxes[0],
    base_fp=BASE,
    # fig=None,
    # ax=None,
    loss_type="inclusion",
    # plot_labels=True,
    # box_filter=None,
)

fig.savefig("2d_boxes_digenic-layer-0.pdf", bbox_inches="tight")

fig, ax = plot_boxes_mpl(
    data=None,
    rev_class_dict=rev_class_dict,
    plot_boxes=boxes[1],
    base_fp=BASE,
    # fig=None,
    # ax=None,
    loss_type="inclusion",
    # plot_labels=True,
    # box_filter=None,
)

fig.savefig("2d_boxes_digenic-layer-1.pdf", bbox_inches="tight")

fig, ax = plot_boxes_mpl(
    data=None,
    rev_class_dict=rev_class_dict,
    plot_boxes=boxes[2],
    base_fp=BASE,
    # fig=None,
    # ax=None,
    loss_type="inclusion",
    # plot_labels=True,
    # box_filter=None,
)

fig.savefig("2d_boxes_digenic-layer-2.pdf", bbox_inches="tight")