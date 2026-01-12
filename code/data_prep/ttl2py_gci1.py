"""Prepare box graph from ontology TTL file."""
# %%
from utils.dataset_utils import get_normalized_el_dataset, get_bots
import os
import pickle

# import rdflib
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

from argparse import ArgumentParser

# Argument for the dataset path
parser = ArgumentParser()
parser.add_argument(
    "--ontology_path", type=str, default="", help="Path to the ontology file"
)
args = parser.parse_args()
ontology_path = args.ontology_path
# %%
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
full_fp = os.path.join(BASE, ontology_path)

with open(os.path.join(BASE, "datasets/box_graph_gci0only.pkl"), "rb") as f:
    box_graph_data = pickle.load(f)
data = box_graph_data["graph"]
gci = box_graph_data["gci"]
rev_rel_dict = box_graph_data["rev_rel_dict"]
rev_class_dict = box_graph_data["rev_class_dict"]
index = {
    "class_index": {v: k for k, v in rev_class_dict.items()},
    "property_index": {v: k for k, v in rev_rel_dict.items()},
}

i2c = {v: k for k, v in index["class_index"].items()}
c2i = index["class_index"]

print("Running get_bots for gci1_bot...")
gci["gci1_bot"] = get_bots(gci1_bot=gci["gci1_bot"]['classes'], i2c=i2c, c2i=c2i, full_fp=full_fp)


with open(os.path.join(BASE, "datasets/box_graph.pkl"), "wb") as fo:
    pickle.dump(
        {
            "source_ontology": ontology_path,
            "graph": data,
            "gci": {
                "gci0": {"classes": gci["gci0"]},
                "gci1_bot": {"classes": gci["gci1_bot"]},
            },
            "rev_class_dict": rev_class_dict,
            "rev_rel_dict": rev_rel_dict,
        },
        fo,
    )
# %%