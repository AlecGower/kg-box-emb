# %%
from memory_profiler import profile
import numpy as np
from box_embeddings.modules.volume import BesselApproxVolume, HardVolume
from box_embeddings.modules.intersection import GumbelIntersection, HardIntersection
from box_embeddings.parameterizations import MinDeltaBoxTensor, SigmoidBoxTensor
from box_embeddings.modules.regularization import L2SideBoxRegularizer
from model import HeteroGNNGAT, HeteroGNNSAGE
import pickle
import os
import sys
print("Python version:", sys.version, file=sys.stderr, flush=True)
from pprint import pprint
print("Importing models...", file=sys.stderr, flush=True)
from model import HeteroGNNGAT, HeteroGNNSAGE, HeteroGNNTransformer, OntologyGNN
import torch
from torch_geometric import seed_everything
from matplotlib import pyplot as plt

import rdflib
from rdflib.namespace import RDF

import logging
import traceback
from time import time

from itertools import product
from tqdm.auto import tqdm

from box_forward import get_boxes_from_model_and_graph, get_initial_boxes_from_model

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE, "code", "presentation"))
from boxplot2d import (
    plot_box_2d,
    plot_min_delta_boxes_2d_matplotlib,
    animate_boxes,
    animate_boxes_with_blitting,
)
from boxplot3d import plot_min_delta_boxes_3d_matplotlib


device = "cuda" if torch.cuda.is_available() else "cpu"
seed_everything(42)
torch.manual_seed(42)
# Enable detect anomaly mode
torch.autograd.set_detect_anomaly(True)
# %%

# %%
GNN_CHANNELS = [2 * 2]
# GNN_CHANNELS = [8, 4]
# GNN_CHANNELS = [24, 4]
# GNN_CHANNELS = [4, 4, 4, 4]
BOX_REGULARIZATION = 0
# BOX_REGULARIZATION = 1e-7
# BOX_REGULARIZATION = 1e-5
# BOX_REGULARIZATION = 0.0001
# BOX_REGULARIZATION = 1000
EPOCHS = 500
SCALE_LOSSES = False
LOSS_TYPE = "distance"
LR = 0.1
LR_DECAY = 0.001
# LR_DECAY = 0.0
REGULARIZATION = 0.001
# REGULARIZATION = 0.00
# NEG_WEIGHT = 2.0
NEG_WEIGHT = 0.5
NEG_RANDOM_WEIGHT = 1.0
# NEG_RANDOM_WEIGHT = 0.1

# PHENO_LOSS_SCALE = 10000.0
PHENO_LOSS_SCALE = 1.0

# Outputs
PLOT_LAST_PRE_GNN = True
PLOT_LAST = True
ANIMATE = False


class TrainingLogger:
    def __init__(self, log_interval=10):
        self.log_interval = log_interval
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        self.training_logs = []

    def on_epoch_begin(self, epoch):
        self.epoch_start_time = time()
        logging.info(f"Epoch {epoch + 1} starting.")

    def on_epoch_end(self, epoch, logs=None):
        elapsed_time = time() - self.epoch_start_time
        logging.info(f"Epoch {epoch + 1} finished in {elapsed_time:.2f} seconds.")
        logs["epoch_time"] = elapsed_time  # Add epoch time to logs
        self.training_logs.append(logs)  # Collect training logs


def box_loss(
    embeddings,
    gci0,
    loss_type="distance",
    box_transform="mindelta",
    inter="gumbel",
    inter_temp=0.1,
    vol="bessel",
    vol_temp=0.1,
    gamma=0.0,
    neg_data=None,
    neg=False,
    neg_random_weight=0.0,
    neg_classes_to_skip=0,
    pheno_loss_scale=1.0,
    **kwargs,
):
    match box_transform:
        case "mindelta":
            box = MinDeltaBoxTensor
        case "sigmoid":
            box = SigmoidBoxTensor
        case _:
            raise NotImplementedError()
    if loss_type == "inclusion":
        return box_loss_inclusion(
            embeddings,
            gci0,
            box=box,
            inter=inter,
            inter_temp=inter_temp,
            vol=vol,
            vol_temp=vol_temp,
            neg_data=neg_data,
            neg=neg,
            neg_random_weight=neg_random_weight,
            neg_classes_to_skip=neg_classes_to_skip,
            pheno_loss_scale=pheno_loss_scale,
        )
    if loss_type == "distance":
        return box_loss_distance(
            embeddings,
            gci0,
            box=box,
            gamma=gamma,
            neg_data=neg_data,
            neg=neg,
            neg_random_weight=neg_random_weight,
            neg_classes_to_skip=neg_classes_to_skip,
            pheno_loss_scale=pheno_loss_scale,
        )
    pass


def box_loss_inclusion(
    embeddings,
    gci0,
    box=MinDeltaBoxTensor,
    inter="gumbel",
    inter_temp=0.1,
    vol="bessel",
    vol_temp=0.1,
    neg_data=None,
    neg=False,
    neg_random_weight=0.0,
    neg_classes_to_skip=0,
    pheno_loss_scale=1.0,
    **kwargs,
):
    def neg_loss_func(A, B, volume, intersect, verbose=False):
        if verbose:
            print(
                f"\tLog arg: {(1 - (volume(intersect(A, B)) / torch.minimum(volume(A), volume(B)))).sort()}"
            )
            print(
                f"\tLog: {(1 - (volume(intersect(A, B)) / torch.minimum(volume(A), volume(B)))).log().sort()}"
            )
            print(
                f"\tClamped: {(1 - (volume(intersect(A, B)) / torch.minimum(volume(A), volume(B)))).clamp(min=1e-9, max=1).sort()}"
            )
            print(
                f"\tClamped log: {(1 - (volume(intersect(A, B)) / torch.minimum(volume(A), volume(B)))).clamp(min=1e-9, max=1).log().sort()}"
            )

            print("A:", volume(A).sort())
            print("B:", volume(B).sort())

        # Have division by zero here, need to fix this...
        return (
            (1 - (volume(intersect(A, B)) / torch.minimum(volume(A), volume(B))))
            .clamp(min=1e-9, max=1)
            .log()
            .sum()
        )

    # if neg or neg_data:
    #     raise NotImplementedError("Negative loss not yet implemented "
    #                               "for inclusion loss")
    match inter:
        case "gumbel":
            intersect = GumbelIntersection(intersection_temperature=inter_temp)
        case "hard":
            intersect = HardIntersection()
        case _:
            raise NotImplementedError()

    match vol:
        case "bessel":
            volume = BesselApproxVolume(
                intersection_temperature=inter_temp,
                volume_temperature=vol_temp,
                log_scale=False,
            )
        case "hard":
            volume = HardVolume(log_scale=False)
        case _:
            raise NotImplementedError()

    loss = 0
    neg_loss = 0
    for x_dict in embeddings:
        for k, emb in x_dict.items():

            if k == "genes":
                continue
            if k == "quality":
                scale_factor = pheno_loss_scale
            else:
                scale_factor = 1.0
            box_emb = box.from_vector(emb)

            subclasses = box_emb[gci0[k][:, 0], ...]
            supclasses = box_emb[gci0[k][:, 1], ...]

            loss -= scale_factor * (
                (volume(intersect(subclasses, supclasses)) / volume(subclasses))
                .clamp(min=1e-9, max=1)
                .log()
                .sum()
            )
            # print(volume(intersect(subclasses, supclasses)))
            # print(volume(subclasses))
            # print((volume(intersect(subclasses, supclasses)) /
            #          volume(subclasses)))
            # print(volume(subclasses))
            # print(volume(supclasses))
            # print(torch.minimum(volume(subclasses), volume(supclasses)))

            # print(((volume(subclasses), volume(subclasses)).min()))
            # print()

            if neg:
                max_i = len(emb)
                rand_classes = torch.randint(
                    low=neg_classes_to_skip,
                    high=max_i,
                    size=(len(gci0[k]),),
                    device=gci0[k].device,
                )
                A = box_emb[rand_classes, ...]
                neg_loss -= scale_factor * neg_loss_func(A, supclasses, volume, intersect)

                rand_classes = torch.randint(
                    low=neg_classes_to_skip,
                    high=max_i,
                    size=(len(gci0[k]),),
                    device=gci0[k].device,
                )
                A = box_emb[rand_classes, ...]
                neg_loss -= scale_factor * neg_loss_func(A, subclasses, volume, intersect)

                rand_classes = torch.randint(
                    low=neg_classes_to_skip,
                    high=max_i,
                    size=(len(gci0[k]), 2),
                    device=gci0[k].device,
                )
                A = box_emb[rand_classes[:, 0], ...]
                B = box_emb[rand_classes[:, 1], ...]
                neg_loss -= scale_factor * neg_loss_func(A, B, volume, intersect)

            if neg_data:
                A = box_emb[neg_data[k][:, 0], ...]
                B = box_emb[neg_data[k][:, 1], ...]

                neg_loss -= scale_factor * neg_loss_func(A, B, volume, intersect, verbose=False)
                # print(f"Neg loss -= {neg_loss_func(A, B, volume, intersect)}")

    return loss, neg_loss


def box_loss_distance(
    embeddings,
    gci0,
    box=MinDeltaBoxTensor,
    gamma=0.0,
    neg_data=None,
    neg=False,
    neg_random_weight=0.0,
    neg_classes_to_skip=0,
    pheno_loss_scale=1.0,
):

    def dist_inclusion(sub_c, sub_o, sup_c, sup_o, neg=False):
        n = -1 if neg else 1
        if neg:
            # return (
            #     torch.relu(-torch.abs(sub_c - sup_c) + sub_o + sup_o + gamma)
            #     .norm(dim=-1)
            #     .sum()
            # )
            v = torch.relu(-torch.abs(sub_c - sup_c) + sub_o + sup_o + gamma)
            delta = (v > 0).all(dim=-1).float()
            return (delta * v.norm(dim=-1)).sum()
        else:
            return (
                torch.relu(torch.abs(sub_c - sup_c) + sub_o - sup_o - gamma)
                .norm(dim=-1)
                .sum()
            )

    loss = 0
    neg_loss = 0
    for x_dict in embeddings:
        for k, emb in x_dict.items():
            if k == "genes":
                continue
            if k == "quality":
                scale_factor = pheno_loss_scale
            else:
                scale_factor = 1.0
            box_emb = box.from_vector(emb)

            subclasses = box_emb[gci0[k][:, 0], ...]
            sub_c, sub_o = subclasses.centre, subclasses.centre - subclasses.z
            supclasses = box_emb[gci0[k][:, 1], ...]
            sup_c, sup_o = supclasses.centre, supclasses.centre - supclasses.z

            loss += scale_factor * dist_inclusion(sub_c, sub_o, sup_c, sup_o, neg=False)

            if neg:
                max_i = len(emb)

                rand_classes = torch.randint(
                    low=neg_classes_to_skip,
                    high=max_i,
                    size=(len(gci0[k]),),
                    device=gci0[k].device,
                )
                nsub = box_emb[rand_classes, ...]
                nsub_c, nsub_o = nsub.centre, nsub.centre - nsub.z
                neg_loss += scale_factor * neg_random_weight * dist_inclusion(
                    nsub_c, nsub_o, sup_c, sup_o, neg=True
                )

                rand_classes = torch.randint(
                    low=neg_classes_to_skip,
                    high=max_i,
                    size=(len(gci0[k]),),
                    device=gci0[k].device,
                )
                nsup = box_emb[rand_classes, ...]
                nsup_c, nsup_o = nsup.centre, nsup.centre - nsup.z
                neg_loss += scale_factor * neg_random_weight * dist_inclusion(
                    sub_c, sub_o, nsup_c, nsup_o, neg=True
                )

                rand_classes = torch.randint(
                    low=neg_classes_to_skip,
                    high=max_i,
                    size=(len(gci0[k]), 2),
                    device=gci0[k].device,
                )
                nsub = box_emb[rand_classes[:, 0], ...]
                nsub_c, nsub_o = nsub.centre, nsub.centre - nsub.z
                nsup = box_emb[rand_classes[:, 1], ...]
                nsup_c, nsup_o = nsup.centre, nsup.centre - nsup.z
                neg_loss += scale_factor * neg_random_weight * dist_inclusion(
                    nsub_c, nsub_o, nsup_c, nsup_o, neg=True
                )

            if neg_data:
                subclasses = box_emb[neg_data[k][:, 0], ...]
                sub_c = subclasses.centre
                sub_o = subclasses.centre - subclasses.z
                supclasses = box_emb[neg_data[k][:, 1], ...]
                sup_c = supclasses.centre
                sup_o = supclasses.centre - supclasses.z

                neg_loss += scale_factor * dist_inclusion(sub_c, sub_o, sup_c, sup_o, neg=True)

    return loss, neg_loss


box_regularizer = L2SideBoxRegularizer(weight=1.0, log_scale=False)
box = MinDeltaBoxTensor


def regularize_box(embeddings):
    reg_loss = 0
    for x_dict in embeddings:
        for k, emb in x_dict.items():
            box_emb = box.from_vector(emb)
            reg_loss -= box_regularizer(box_emb)
    return reg_loss


def small_box_penalty(embeddings):
    loss = 0
    for x_dict in embeddings:
        for emb in x_dict.values():
            box_emb = box.from_vector(emb)
            box_sizes = torch.norm(box_emb.Z - box_emb.z, dim=-1)
            # print(box_sizes)
            loss += torch.relu(1 / box_sizes - 1).sum()
    return loss


# @profile
def train_boxes_OntologyGNN(
    graph,
    gci,
    gnn_channels=GNN_CHANNELS,
    lr=LR,
    lr_decay=LR_DECAY,
    epochs=EPOCHS,
    loss_type=LOSS_TYPE,
    regularization=REGULARIZATION,
    box_regularization=BOX_REGULARIZATION,
    neg_weight=NEG_WEIGHT,
    neg_random_weight=NEG_RANDOM_WEIGHT,
    scale_losses=SCALE_LOSSES,
    save_weights=False,
    neg_classes_to_skip=0,
    pheno_loss_scale=1.0,
):
    print(
        f"""

GNN_CHANNELS: {gnn_channels}
LR: {lr}
LR_DECAY: {lr_decay}
EPOCHS: {epochs}
LOSS_TYPE: {loss_type}
REGULARIZATION: {regularization}
BOX_REGULARIZATION: {box_regularization}
NEG_WEIGHT: {neg_weight}
SCALE_LOSSES: {scale_losses}""", file=sys.stderr, flush=True
    )
    # model = HeteroGNNGAT(GNN_CHANNELS, graph.edge_types, graph.x_dict)
    # model = HeteroGNNSAGE(GNN_CHANNELS, graph.edge_types, graph.x_dict)
    # model = HeteroGNNTransformer(GNN_CHANNELS, graph.edge_types, graph.x_dict)
    model = OntologyGNN(gnn_channels, graph.edge_types, graph.x_dict)
    print(model, file=sys.stderr, flush=True)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=regularization)
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    # %%
    model.requires_grad_(True)
    model.node_embeddings.requires_grad_(True)

    boxes = []
    weights = [] if save_weights else None
    last_epoch = 0
    try:
        for epoch in range(epochs):
            optimizer.zero_grad()

            x_dicts = model(graph, return_embs=True)
            # ^^^ List of dictionaries, one for each layer (inc. initial embeddings)
            # x_dicts = [model(graph, return_embs=False)]

            pos_loss, neg_loss = box_loss(
                x_dicts,
                gci["gci0"],
                loss_type=loss_type,
                inter="gumbel",
                vol="bessel",
                neg_data=gci["gci1_bot"],
                neg=True,
                neg_random_weight=neg_random_weight,
                neg_classes_to_skip=neg_classes_to_skip,
                pheno_loss_scale=pheno_loss_scale,
            )
            if box_regularization > 0.0:
                reg_loss = small_box_penalty(x_dicts)
            else:
                reg_loss = torch.tensor(0.0)
            pos_loss_scaled = pos_loss / len(gci["gci0"]["classes"])
            neg_loss_scaled = neg_loss / (
                3 * len(gci["gci0"]["classes"]) + len(gci["gci1_bot"]["classes"])
            )
            pos_ratio = torch.exp(-pos_loss_scaled)
            neg_ratio = 1 - torch.exp(-neg_loss_scaled)
            if scale_losses:
                loss = (
                    pos_loss_scaled
                    + neg_weight * neg_loss_scaled
                    + box_regularization * reg_loss
                )
            else:
                loss = pos_loss + neg_weight * neg_loss + box_regularization * reg_loss
            # loss = neg_loss
            # loss = 1 - pos_ratio + neg_weight * neg_ratio + box_regularization * reg_loss
            total_loss = loss.detach().item()

            if loss_type == "distance":
                print(
                    f"Epoch: {epoch}, total loss: {total_loss:.4g}, pos loss: {pos_loss:.6g}, neg loss: {neg_loss:.6g}, reg: {reg_loss:.3g}", 
                    file=sys.stderr, flush=True   
                )
            else:
                print(
                    f"Epoch: {epoch}, total loss: {total_loss:.4g}, pos ratio: {pos_ratio:.6g}, neg ratio: {neg_ratio:.6g}, reg: {reg_loss:.8g}",
                    file=sys.stderr, flush=True   
                )

            # Backpropagate loss gradients
            loss.backward()
            optimizer.step()
            #
            # if epoch % 1 == 0:
            # Only save first and last epoch to save space
            if epoch == 0 or epoch == epochs - 1:

                if loss_type == "distance":
                    boxes.append(
                        (
                            get_boxes_from_model_and_graph(model, graph)
                            .data.detach().cpu()
                            .numpy(),
                            total_loss,
                            pos_loss.detach().item(),
                            neg_loss.detach().item(),
                            reg_loss.detach().item(),
                        )
                    )
                else:
                    boxes.append(
                        (
                            get_boxes_from_model_and_graph(model, graph)
                            .data.detach().cpu()
                            .numpy(),
                            total_loss,
                            pos_ratio.detach().item(),
                            neg_ratio.detach().item(),
                            reg_loss.detach().item(),
                        )
                    )
                if save_weights:
                    weights.append(model.state_dict())
            last_epoch = epoch

            # decay LR
            lr = lr * (1 - lr_decay)

    except KeyboardInterrupt:
        print(f"\nTraining stopped by user during Epoch {epoch}", file=sys.stderr, flush=True)
        boxes = boxes[:last_epoch]
    # print(MinDeltaBoxTensor.from_vector(x_dicts[-1]['classes']).Z)
    # %%
    return model, boxes, last_epoch, weights


def plot_boxes_mpl(
    data,
    rev_class_dict,
    plot_boxes,
    base_fp,
    fig=None,
    ax=None,
    loss_type=LOSS_TYPE,
    plot_labels=True,
    box_filter=None,
):


    print("Loading ontology for filtering...", file=sys.stderr, flush=True)
    g = rdflib.Graph()
    # g.parse(os.path.join(base_fp, data["source_ontology"]))
    # g.parse(os.path.join(base_fp, "graphs/split_graphs/quality-disjoint.ttl"))
    # g.parse(os.path.join(base_fp, "graphs/split_graphs/cell_comp-disjoint.ttl"))
    g.parse(os.path.join(base_fp, "graphs/split_graphs/mol_func-disjoint.ttl"))
    print("Ontology loaded.", file=sys.stderr, flush=True)
    ROOT = "obo:GO_0005575"
    # ROOT = "obo:APO_0000017"
    # ROOT = "obo:GO_0003674"

    class_dict = {v: k for k, v in rev_class_dict.items()}

    
    filter_query = f"""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX obo: <http://purl.obolibrary.org/obo/>
SELECT * WHERE {{
?concept rdfs:subClassOf* ?top .
?top rdfs:subClassOf {ROOT} .
OPTIONAL {{ ?concept rdfs:label ?label }}
}}
""".strip()
    top_class_query = f"""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX obo: <http://purl.obolibrary.org/obo/>
SELECT * WHERE {{
?concept rdfs:subClassOf {ROOT} .
OPTIONAL {{ ?concept rdfs:label ?label }}
}}
""".strip()
    ancestor_query = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX obo: <http://purl.obolibrary.org/obo/>
ASK {{
  <{}> rdfs:subClassOf+ <{}> .
}}
""".strip()

    # plot_boxes = {k: v for k, v in plot_boxes.items() if (box_filter is None or box_filter(k))}
    
    # Filter boxes to only include the ?concepts from the SPARQL query
    # noting that the concept will be the uri in the graph, so will need
    # to use class_dict to get the class index from the uri
    top_classes = {str(row.concept) : str(row.label) for row in g.query(top_class_query)}
    # assign each top class a colour from a matplotlib colormap
    import matplotlib
    cmap = matplotlib.cm.get_cmap('tab10')
    top_class_colors = {}
    for i, uri in enumerate(top_classes.keys()):
        top_class_colors[uri] = matplotlib.colors.rgb2hex(cmap(i % 10))
    plot_cls = g.query(filter_query)
    plot_cls_set = set()
    color_dict = {}
    label_dict = {}
    for row in plot_cls:
        plot_cls_set.add(class_dict.get(str(row.concept)))
        if str(row.concept) in top_classes:
            # print(f"Class {str(row.label)} is a direct subclass of {ROOT}", file=sys.stderr, flush=True)
            color_dict[class_dict.get(str(row.concept))] = "black"
            label_dict[class_dict.get(str(row.concept))] = str(row.label)
        else:
            # print(f"Class {str(row.label)} is subclass of {str(row.top)}, a deeper subclass of {ROOT}", file=sys.stderr, flush=True)
            color_dict[class_dict.get(str(row.concept))] = top_class_colors.get(str(row.top))
            label_dict[class_dict.get(str(row.concept))] = None

    

    # From file instead
    # with open(os.path.join(base_fp, "top_classes.csv"), "r") as fi:
    #     top_classes = {line.split(";")[1]: line.split(";")[2].strip() for line in fi.readlines()}
    # with open(os.path.join(base_fp, "phenotypes.csv"), "r") as fi:
    #     phenotypes = [line.strip().split(";") for line in fi.readlines()]

    # for p in phenotypes:
    #     p[0] = int(p[0])

    # phenos_set = set()
    # color_dict = {}
    # label_dict = {}
    # for pheno_class_id, pheno_uri, pheno_label in phenotypes:
    #     phenos_set.add(pheno_class_id)
    #     if pheno_uri in top_classes:
    #         print(f"Phenotype {pheno_label} is a direct subclass of APO_0000017", file=sys.stderr, flush=True)
    #         color_dict[pheno_class_id] = "green"
    #         label_dict[pheno_class_id] = pheno_label
    #     else:
    #         print(f"Phenotype {pheno_label} is a deeper subclass of APO_0000017", file=sys.stderr, flush=True)
    #         color_dict[pheno_class_id] = "black"
    #         label_dict[pheno_class_id] = None

    # # Print out the phehnotypes being plotted and the top classes
    # print(f"Phenotypes to be plotted ({len(plot_cls_set)}):", file=sys.stderr, flush=True)
    # for c in plot_cls_set:
    #     print(f"{c};{rev_class_dict.get(c)};{label_dict.get(c)}", file=sys.stderr, flush=True)
    # print(f"Top-level classes ({len(top_classes)}):", file=sys.stderr, flush=True)
    # for uri, label in top_classes.items():
    #     print(f"{class_dict.get(uri)};{uri};{label}", file=sys.stderr, flush=True)

    plot_boxes = {k: v for k, v in plot_boxes.items() if k in plot_cls_set}
    print(f"Plotting {len(plot_boxes)} boxes after filtering for classes under root {ROOT}.", file=sys.stderr, flush=True)

    w_list = [t[0, :] for t in plot_boxes.values()]
    d_list = [t[1, :] for t in plot_boxes.values()]
    colors = [color_dict.get(k) for k in plot_boxes.keys()]
    labels = [label_dict.get(k) for k in plot_boxes.keys()]
    alphas = [0.2 if c != "black" else 0.8 for c in colors]
    linewidths = [0.4 if c != "black" else 1.6 for c in colors]

    # rev_superclass_dict = {}
    # for key, sub in rev_class_dict.items():
    #     q = [
    #         t
    #         for t in g.triples((rdflib.URIRef(sub), RDF.type, None))
    #         if t[2] != rdflib.URIRef("http://www.w3.org/2002/07/owl#NamedIndividual")
    #     ]
    #     if len(q) == 0:
    #         rev_superclass_dict[key] = None
    #     else:
    #         rev_superclass_dict[key] = q[0][2]

    # color_dict = dict(
    #     zip(
    #         sorted(list(set(rev_superclass_dict.values()))),
    #         [None, "green", "blue", "purple", "red"],
    #     )
    # )
    # colors = [color_dict.get(v) for v in rev_superclass_dict.values()]
    # colors = ["black" if c == "red" else c for c in colors]
    # colors = ["black" for v in rev_superclass_dict.values()]
    # colors = ["black" for k in plot_boxes.keys()]
    # labels = [
    #     rev_class_dict.get(k).split("/")[-1] if plot_labels else None
    #     for k in plot_boxes.keys()
    # ]
    print({f"Box shape: {w_list[0].shape}"}, file=sys.stderr, flush=True)
    if w_list[0].shape == (3,):
        print("Plotting 3D boxes...", file=sys.stderr, flush=True)
        fig, ax = plot_min_delta_boxes_3d_matplotlib(
            w_list,
            d_list,
            colors,
            alphas=[
                1.0 if i < 4 else 0.0 if i < 6 else 0.3 for i in range(len(colors))
            ],
            draw_labels=True,
            labels=[l if i < 4 else None for i, l in enumerate(labels)],
            linewidths=[
                2.5 if i < 4 else 0.0 if i < 6 else 0.4 for i in range(len(colors))
            ],
            color_legend={"purple": "Women", "blue": "Men", "green": "Countries"},
            title=f"Box Embeddings - {'Overlap' if loss_type == 'inclusion' else 'Distance'}",
            fig=fig,
            ax=ax,
        )
    elif w_list[0].shape == (2,):
        print("Plotting 2D boxes...", file=sys.stderr, flush=True)
        fig, ax = plot_min_delta_boxes_2d_matplotlib(
            w_list,
            d_list,
            colors,
            # alphas=[
            #     1.0 if i < 4 else 0.0 if i < 6 else 0.3 for i in range(len(colors))
            # ],
            # alphas=[0.3 for i in range(len(colors))],
            alphas=alphas,
            draw_labels=True,
            # labels=[l if i < 4 else None for i, l in enumerate(labels)],
            # labels=[None for i, l in enumerate(labels)],
            labels=labels,
            label_fontsize=6,
            # linewidths=[
            #     2.5 if i < 4 else 0.0 if i < 6 else 0.4 for i in range(len(colors))
            # ],
            # linewidths=[0.4 for i in range(len(colors))],
            linewidths=linewidths,
            # color_legend={"purple": "Women", "blue": "Men", "green": "Countries"},
            # Add a legend for the top-level classes and make it small and just outside the plot
            color_legend={v: top_classes.get(k) for k, v in top_class_colors.items()},
            title=f"Box Embeddings - {'Overlap' if loss_type == 'inclusion' else 'Distance'}",
            fig=fig,
            ax=ax,
        )
        # set equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
    else:
        raise NotImplementedError(
            "Plots for dimensions other than 2 or 3 not implemented."
        )

    return fig, ax


if __name__ == "__main__":

    print("Starting training...", file=sys.stderr, flush=True)

    lrs = [1e-2, 1e-1, 1e0]
    # lrs = [1e-1]
    lr_decays = [0, 0.001]
    box_regs = [0, 1e-4, 1]
    # box_regs = [0]
    # neg_weights = [1e-2, 1e-1, 1]
    neg_weights = [0.1, 1, 10]
    neg_rand_weights = [0.01, 1]
    scales = [True, False]
    # scales = [False]
    # gnns = [[2 * 2], [2 * 2, 2 * 2]]
    gnns = [[2 * 2]]

    # for i, (lr, dec, br, neg, neg_rand, scale, g) in tqdm(
    #     enumerate(
    #         product(
    #             lrs, lr_decays, box_regs, neg_weights, neg_rand_weights, scales, gnns
    #         )
    #     )
    # ):
    #     if i < 67:
    #         continue
    #     LR = lr
    #     LR_DECAY = dec
    #     BOX_REGULARIZATION = br
    #     NEG_WEIGHT = neg
    #     NEG_RANDOM_WEIGHT = neg_rand
    #     SCALE_LOSSES = scale
    #     GNN_CHANNELS = g

    print(
        f"""

GNN_CHANNELS: {GNN_CHANNELS}
LR: {LR}
LR_DECAY: {LR_DECAY}
EPOCHS: {EPOCHS}
LOSS_TYPE: {LOSS_TYPE}
REGULARIZATION: {REGULARIZATION}
BOX_REGULARIZATION: {BOX_REGULARIZATION}
NEG_WEIGHT: {NEG_WEIGHT}
SCALE_LOSSES: {SCALE_LOSSES}""", file=sys.stderr, flush=True   
    )

    # %%
    BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import datetime

    now = datetime.datetime.now()
    output_dir = os.path.join(
        BASE,
        "trained_models",
        # "hyperparam_search",
        f"{LOSS_TYPE}_loss__lr_{LR}_lr_dec_{LR_DECAY}_reg_{REGULARIZATION}_boxreg_{BOX_REGULARIZATION}_neg_{NEG_WEIGHT}_negrand_{NEG_RANDOM_WEIGHT}_scale_{SCALE_LOSSES}_{now.strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(output_dir, exist_ok=True)
    orig_stdout = sys.stdout
    f = open(os.path.join(output_dir, "log.txt"), "w")
    # Output stdout to log file but also to console
    class Tee(object):
        def __init__(self, *files):
            self.files = files

        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()  # If you want the output to be visible immediately

        def flush(self):
            for f in self.files:
                f.flush()

    # sys.stdout = f
    sys.stdout = Tee(sys.stdout, f)

    #
    # with open(os.path.join(BASE, "datasets/box_graph_all.pkl"), "rb") as fi:
    # with open(os.path.join(BASE, "datasets/box_graph_all_disjoint_quality.pkl"), "rb") as fi:
    # with open(os.path.join(BASE, "datasets/box_graph_all_disjoint_cell_comp.pkl"), "rb") as fi:
    with open(os.path.join(BASE, "datasets/box_graph_all_disjoint_mol_func.pkl"), "rb") as fi:
        data = pickle.load(fi)

    pheno = [
        k for k,v in data['rev_class_dict'].items() 
        if v.startswith("http://purl.obolibrary.org/obo/APO_") 
        or v.startswith("http://sgd-kg.project-genesis.io#APO_")
    ]

    graph = data["graph"].to(device)
    rev_class_dict = data["rev_class_dict"]
    rev_rel_dict = data["rev_rel_dict"]
    # pprint(graph.edge_types)
    gci = data["gci"]
    gci = {k: {kk: vv.to(device) for kk, vv in v.items()} for k, v in gci.items()}
    # graph['classes'].node_id = torch.arange(len(graph['classes'].x))
    # %%
    true_classes = set(gci["gci0"]["classes"][:, 1].detach().cpu().numpy())
    # pprint({k: v for k, v in rev_class_dict.items() if k in true_classes})

    # Create an output directory if it doesn't exist
    # with current date and time in the directory name

    # Save the hyperparameters and training information to a text file
    with open(os.path.join(output_dir, "training_info.txt"), "w") as fo:
        fo.write(f"SLURM JOB ID: {os.getenv('SLURM_JOB_ID')}\n")
        fo.write(f"Ontology source: {data['source_ontology']}\n")
        fo.write(f"Loss type: {LOSS_TYPE}\n")
        fo.write(f"Epochs: {EPOCHS}\n")
        fo.write(f"Learning rate: {LR}\n")
        fo.write(f"Learning rate decay: {LR_DECAY}\n")
        fo.write(f"Regularization: {REGULARIZATION}\n")
        fo.write(f"Box regularization: {BOX_REGULARIZATION}\n")
        fo.write(f"Negative weight: {NEG_WEIGHT}\n")
        fo.write(f"Negative (Random) weight: {NEG_RANDOM_WEIGHT}\n")
        fo.write(f"Losses scaled (pos. and neg.): {SCALE_LOSSES}\n")
        fo.write(f"Phenotype losses scaled by factor: {PHENO_LOSS_SCALE}\n")
        fo.write(f"Model channels: {GNN_CHANNELS}\n")
        fo.write(f"Training started at: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")

        # try:
    model, boxes, stop_epoch, weights = train_boxes_OntologyGNN(
        graph,
        gci,
        save_weights=True,
        # neg_classes_to_skip=len(true_classes) + 2,
        # lr=lr,
        # lr_decay=dec,
        # gnn_channels=g,
        # box_regularization=br,
        # neg_weight=neg,
        # neg_random_weight=neg_rand,
        # scale_losses=scale,
        pheno_loss_scale=PHENO_LOSS_SCALE
    )

    with open(os.path.join(output_dir, "training_info.txt"), "a") as fo:
        fo.write(f"Training ended at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    model.to("cpu")
    graph.to("cpu")

    with open(os.path.join(output_dir, "training_info.txt"), "a") as fo:
        fo.write(f"Model: {model.__class__.__name__}\n")
        fo.write(f"Model GNN type: {model.gnn.__class__.__name__}\n")
        fo.write(
            f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n"
        )
        fo.write(f"Epochs completed: {stop_epoch + 1} of planned {EPOCHS} epochs")

    # Save the model to a file
    print(f"Saving model to {output_dir}")

    # Save the model and graph to a pickle file
    with open(os.path.join(output_dir, "box_model.pkl"), "wb") as fo:
        pickle.dump(model, fo)

    # Save first and last boxes
    with open(os.path.join(output_dir, "boxes_first_last.pkl"), "wb") as fo:
        pickle.dump(
            {
                "first": boxes[0],
                "last": boxes[-1],
                "rev_class_dict": rev_class_dict,
                "rev_rel_dict": rev_rel_dict,
            },
            fo,
        )
    # %%

    # Get boxes and losses
    boxes_epochs = np.stack([b[0] for b in boxes])
    be_dict = {i: boxes_epochs[:, i, :, :] for i in range(boxes_epochs.shape[1])}
    # print([[t for t in b[1:]] for b in boxes])
    losses = np.array([b[1:] for b in boxes])

    # plot_classes = set(
    #     c
    #     for c in gci["gci0"]["classes"][:, 1].detach().cpu().numpy()
    #     if (
    #         lambda k: any(
    #             [
    #                 k.startswith("http://purl.obolibrary.org/obo/APO"),
    #                 k.startswith("http://purl.obolibrary.org/obo/CHEBI"),
    #                 k == "http://hypo.project-genesis.io#organismState",
    #             ]
    #         )
    #     )(rev_class_dict[c])
    # )


    sys.stdout = sys.__stdout__

    # Plot last embeddings
    if PLOT_LAST_PRE_GNN:
        first_boxes = get_initial_boxes_from_model(model, graph).data.detach().cpu().numpy()
        print(f"Plotting {len(first_boxes)} pre-GNN boxes.", file=sys.stderr, flush=True)
        plot_boxes_pre = {i: first_boxes[i, :, :] for i in range(boxes_epochs.shape[1])}
        fig_pre, ax_pre = plot_boxes_mpl(data, rev_class_dict, plot_boxes_pre, BASE, plot_labels=True)
        print("Saving final boxes plot (pre-GNN) (PNG)...", file=sys.stderr, flush=True)
        fig_pre.savefig(os.path.join(output_dir, "final_boxes_pre_gnn.png"), bbox_inches='tight', dpi=300)
        print("Saving final boxes plot (pre-GNN) (PDF)...", file=sys.stderr, flush=True)
        fig_pre.savefig(os.path.join(output_dir, "final_boxes_pre_gnn.pdf"), bbox_inches='tight')
        print("Saving final boxes plot (pre-GNN) as matplotlib figure for potential future use...", file=sys.stderr, flush=True)
        with open(os.path.join(output_dir, "final_boxes_fig_ax_pre.pkl"), "wb") as fo:
            pickle.dump((fig_pre, ax_pre), fo)  
        # plt.close("all")

    if PLOT_LAST:
        plot_boxes = {k: v[-1] for k, v in be_dict.items()}
        print(f"Plotting {len(plot_boxes)} post-GNN boxes.", file=sys.stderr, flush=True)
        fig, ax = plot_boxes_mpl(data, rev_class_dict, plot_boxes, BASE, plot_labels=True)
        print("Saving final boxes plot (post-GNN) (PNG)...", file=sys.stderr, flush=True)
        fig.savefig(os.path.join(output_dir, "final_boxes.png"), bbox_inches='tight', dpi=300)
        print("Saving final boxes plot (post-GNN) (PDF)...", file=sys.stderr, flush=True)
        fig.savefig(os.path.join(output_dir, "final_boxes.pdf"), bbox_inches='tight')
        print("Saving final boxes plot (post-GNN) as matplotlib figure for potential future use...", file=sys.stderr, flush=True)
        with open(os.path.join(output_dir, "final_boxes_fig_ax_post.pkl"), "wb") as fo:
            pickle.dump((fig, ax), fo)  
        # plt.close("all")

    if ANIMATE:
        animate_boxes_with_blitting(
            be_dict,
            losses,
            save=True,
            fp=os.path.join(output_dir, "training.mp4"),
            box_filter=lambda k: k in true_classes,
            # box_filter=lambda k: k in plot_classes,
            # box_filter=lambda k: True,
            box_filter_type="bold",
            box_labels=rev_class_dict,
            box_label_filter=lambda k: k in true_classes,
            # box_label_filter=lambda k: k in plot_classes
        )
    # except Exception as e:
    #     print(traceback.format_exc(), file=sys.stderr)
    # finally:
    f.close()
    sys.stdout = orig_stdout

    # plt.close("all")
