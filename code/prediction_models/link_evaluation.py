# Import
import os
import sys
import pickle
import torch
import datetime
import numpy as np
from tqdm.auto import tqdm
from copy import deepcopy
from random import sample, choice
from torch_geometric import seed_everything
from sklearn.model_selection import train_test_split
from box_embeddings.parameterizations import MinDeltaBoxTensor
from train_boxes import train_boxes_OntologyGNN, box_loss
from itertools import islice
import argparse
import json
import threading
import time
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import time

# Set maximum number of test edges to evaluate
MAX_TEST_EDGES = -1  # Set to -1 for all edges
MAX_TEST_EDGES_PER_TYPE = 500  # Set to -1 for all edges
MAX_EDGE_TYPES = -1  # Set to -1 for all edge types

# Constants
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# GNN_CHANNELS = [16, 16]
GNN_CHANNELS = [64]
# GNN_CHANNELS = [24]
# GNN_CHANNELS = [2 * 2]
LR = 0.05
LR_DECAY = 0.000
REGULARIZATION = 0.001
BOX_REGULARIZATION = 0.000
EPOCHS = 500
NEG_WEIGHT = 0.5
NEG_RANDOM_WEIGHT = 1.0
LOSS_TYPE = "inclusion"
SCALE_LOSSES = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
seed_everything(42)
torch.manual_seed(42)
# Enable detect anomaly mode
torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
parser.add_argument("--shard-id", type=int, default=0)
parser.add_argument("--num-shards", type=int, default=1)
parser.add_argument("--output-dir", type=str, default=None)
parser.add_argument("--workers", type=int, default=1, help="Number of worker threads for inference; keep 1 for safety on GPU")
parser.add_argument("--results-log", type=str, default=None, help="Path to append JSONL results; defaults to output_dir/link_eval_results.jsonl")
parser.add_argument("--flush-every", type=int, default=1, help="Flush results to disk every N edges")
parser.add_argument("--no-deepcopy", action="store_true", help="Avoid deepcopy of G_train; modify edge_index in-place with restore (faster)")
args = parser.parse_args()
SHARD_ID = args.shard_id
NUM_SHARDS = args.num_shards
WORKERS = args.workers
# Enforce a single worker for now (safe and avoids GPU contention); clamp and warn if user requested >1
if WORKERS is None:
    WORKERS = 1
elif WORKERS > 1:
    print(f"Warning: forcing --workers to 1 (requested {args.workers}) to avoid GPU contention.", file=sys.stderr)
    WORKERS = 1
RESULTS_LOG = args.results_log
FLUSH_EVERY = args.flush_every
NO_DEEPCOPY = args.no_deepcopy
# use args.output_dir / args.checkpoint_dir if provided


# Define functions
def load_graph():
    with open(os.path.join(BASE, "datasets/box_graph_all.pkl"), "rb") as fi:
        data = pickle.load(fi)
    graph = data["graph"].to(DEVICE)
    rev_class_dict = data["rev_class_dict"]
    rev_rel_dict = data["rev_rel_dict"]
    gci = data["gci"]
    gci = {k: {kk: vv.to(DEVICE) for kk, vv in v.items()} for k, v in gci.items()}

    return graph, gci, data, rev_class_dict, rev_rel_dict


def graph_train_test_split(G, ratio=0.8):
    G_train, G_test = deepcopy(G), deepcopy(G)
    for edge_i, (edge_type, edge_details) in enumerate(G.edge_items()):
        if len(edge_details["edge_index"].T) < MAX_TEST_EDGES_PER_TYPE:
            print(
                f"Skipping edge type {edge_type} with only {len(edge_details['edge_index'].T)} edges", file=sys.stderr
            )
            del G_train[edge_type]
            del G_test[edge_type]
            continue
        if edge_i > MAX_EDGE_TYPES and MAX_EDGE_TYPES > 0:
            print(
                f"Skipping edge type {edge_type} beyond max edge types {MAX_EDGE_TYPES}",
                file=sys.stderr,
            )
            del G_train[edge_type]
            del G_test[edge_type]
            continue
        train, test = train_test_split(
            edge_details["edge_index"].T, random_state=42, test_size=1 - ratio
        )
        # Only include MAX_TEST_EDGES_PER_TYPE in test set
        print(test.shape, file=sys.stderr)
        if MAX_TEST_EDGES_PER_TYPE > 0:
            test = test[:MAX_TEST_EDGES_PER_TYPE, :]
        print(test.shape, file=sys.stderr)
        G_train[edge_type]["edge_index"] = train.T
        G_test[edge_type]["edge_index"] = test.T
        assert (
            G_train[edge_type]["edge_index"].shape[0]
            == edge_details["edge_index"].shape[0]
        )
    return G_train, G_test


def train_and_save_model(
    G, gci, true_classes, output_dir
):  # -> tuple[Any, Any, Any, Any]:
    # Return model to device
    model, boxes, stop_epoch, weights = train_boxes_OntologyGNN(
        G,
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
        neg_classes_to_skip=len(true_classes) + 2,
        # gnn_channels=g,
        # box_regularization=br,
        # neg_weight=neg,
        # scale_losses=scale,
    )
    model.to("cpu")
    with open(os.path.join(output_dir, "training_info.txt"), "a") as fo:
        fo.write(f"Model: {model.__class__.__name__}\n")
        fo.write(
            f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n"
        )
        fo.write(f"Epochs completed: {stop_epoch + 1} of planned {EPOCHS} epochs")

    # Save the model to a file
    print(f"Saving model to {output_dir}", file=sys.stderr)

    # Save the model and graph to a pickle file
    with open(os.path.join(output_dir, "box_model.pkl"), "wb") as fo:
        pickle.dump(model, fo)

    return model, boxes, stop_epoch, weights


def box_embeddings_from_model(model, G, box=MinDeltaBoxTensor, final_only=True):
    if final_only:
        x_dicts = [model(G, return_embs=False)]
        return box.from_vector(x_dicts[0]["classes"]), x_dicts
    else:
        # x_dicts = model(G, return_embs=True)
        # ^^^ List of dictionaries, one for each layer (inc. initial embeddings)
        raise NotImplementedError


def dist_inclusion(sub_c, sub_o, sup_c, sup_o, gamma=0.0):
    return (
        torch.relu(torch.abs(sub_c - sup_c) + sub_o - sup_o - gamma).norm(dim=-1).sum()
    )


def embedding_distance(
    emb1,
    emb2,
    gamma=0.0,
):
    emb1_c, emb1_o = emb1.centre, emb1.Z - emb1.centre
    emb2_c, emb2_o = emb2.centre, emb2.Z - emb2.centre
    return 0.5 * (
        dist_inclusion(emb1_c, emb1_o, emb2_c, emb2_o, gamma)
        + dist_inclusion(emb2_c, emb2_o, emb1_c, emb1_o, gamma)
    )


def get_superclass(subclass, gci, person=False):
    if person:
        return 2
    else:
        cands = (
            gci["gci0"]["classes"][gci["gci0"]["classes"][:, 0] == subclass][:, 1]
            .detach()
            .tolist()
        )
        return next(filter(lambda c: c != 2, cands))


def get_random_subclass(superclass, gci):
    return choice(
        gci["gci0"]["classes"][gci["gci0"]["classes"][:, 1] == superclass][:, 0]
        .detach().to('cpu')
        .numpy()
    )


if __name__ == "__main__":
    # Load graph and find "true classes"
    G, gci, data, rev_class_dict, rev_rel_dict = load_graph()
    true_classes = set(gci["gci0"]["classes"][:, 1].detach().to('cpu').numpy())

    # Create an output directory if it doesn't exist
    # with current date and time in the directory name
    now = datetime.datetime.now()
    output_dir = os.path.join(
        BASE,
        "trained_models",
        "link_evaluation",
        f"{LOSS_TYPE}_loss_{now.strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(output_dir, exist_ok=True)

    # Save the hyperparameters and training information to a text file
    with open(os.path.join(output_dir, "training_info.txt"), "w") as fo:
        fo.write(f"Ontology source: {data['source_ontology']}\n")
        fo.write(f"Loss type: {LOSS_TYPE}\n")
        fo.write(f"Epochs: {EPOCHS}\n")
        fo.write(f"Learning rate: {LR}\n")
        fo.write(f"Learning rate decay: {LR_DECAY}\n")
        fo.write(f"Regularization: {REGULARIZATION}\n")
        fo.write(f"Box regularization: {BOX_REGULARIZATION}\n")
        fo.write(f"Negative weight: {NEG_WEIGHT}\n")
        fo.write(f"Losses scaled (pos. and neg.): {SCALE_LOSSES}\n")
        fo.write(f"Model channels: {GNN_CHANNELS}\n")
        fo.write(f"Training started at: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")

    #################################################################
    # Pseudocode
    # 1. Split graph into train, G:=(V,E) and test, G':=(V,E')
    # 2. Train embedding parameters using G
    # 3. Calculate embeddings B from G using trained parameters
    # 4. distances = {}
    # 5. For each edge, e, in E':
    # 6.   Create graph G*:=(V,E u {e})
    # 7.   calculate embeddings B* from G* using trained parameters
    # 8.   Set distance dist:=0
    # 9.   For v in V:
    # 10.    b:=B(v), b*=B*(v)
    # 11.    dist:= dist + 〈b,b*〉
    # 12.  distances[e]:= dist
    #################################################################

    # Pseudocode  (filled in)
    # 1. Split graph into train, G:=(V,E) and test, G':=(V,E')
    G_train, G_test = graph_train_test_split(G, ratio=0.8)

    # Save these for use with other models
    with open(os.path.join(output_dir, "G_train.pkl"), "wb") as fo:
        pickle.dump(G_train.cpu(), fo)
    with open(os.path.join(output_dir, "G_test.pkl"), "wb") as fo:
        pickle.dump(G_test.cpu(), fo)
    G_train = G_train.to(DEVICE)
    G_test = G_test.to(DEVICE)

    # 2. Train embedding parameters using G
    print("Training base model using G_train...", file=sys.stderr)
    model, boxes, stop_epoch, weights = train_and_save_model(
        G_train, gci, true_classes, output_dir
    )

    # Ensure model is on the same device as the graph to avoid device mismatch errors
    try:
        model.to(DEVICE)
        model.eval()
    except Exception as e:
        print(f"Warning: failed to move model to {DEVICE}: {e}", file=sys.stderr)

    # 3. Calculate embeddings B from G using trained parameters
    B, x_dicts_orig = box_embeddings_from_model(model, G_train, box=MinDeltaBoxTensor)
    B_full, x_dicts_full = box_embeddings_from_model(model, G, box=MinDeltaBoxTensor)
    print(f"Total distance from G_train to G: {embedding_distance(B, B_full)}", file=sys.stderr)
    # Stream out results to a JSONL file (append) to avoid keeping everything in memory
    if args.output_dir:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    results_log = RESULTS_LOG if RESULTS_LOG else os.path.join(output_dir, "link_eval_results.jsonl")

    write_lock = threading.Lock()
    write_counter = {"n": 0}
    results_fp = open(results_log, "a", buffering=1)

    def _write_result(d):
        s = json.dumps(d)
        with write_lock:
            results_fp.write(s + "\n")
            write_counter["n"] += 1
            if write_counter["n"] % FLUSH_EVERY == 0:
                try:
                    results_fp.flush()
                    os.fsync(results_fp.fileno())
                except Exception:
                    pass

    # single-edge processing (no grad)
    def _process_edge(idx, edge_type, source, target):
        with torch.no_grad():
            dev = G_train[edge_type]["edge_index"].device
            new_idx = torch.stack([source, target]).unsqueeze(1).to(dev)

            # Test edge
            if NO_DEEPCOPY:
                old_idx = G_train[edge_type]["edge_index"]
                G_train[edge_type]["edge_index"] = torch.cat([old_idx, new_idx], dim=1)
                try:
                    B_star, x_dicts_star = box_embeddings_from_model(model, G_train, box=MinDeltaBoxTensor)
                finally:
                    G_train[edge_type]["edge_index"] = old_idx
            else:
                G_star = deepcopy(G_train)
                G_star[edge_type]["edge_index"] = torch.cat([G_star[edge_type]["edge_index"], new_idx], dim=1)
                B_star, x_dicts_star = box_embeddings_from_model(model, G_star, box=MinDeltaBoxTensor)

            # Random edge
            rs, rt = sample(range(len(rev_class_dict)), k=2)
            rand_idx = torch.tensor([[rs, rt]], device=dev).T
            if NO_DEEPCOPY:
                old_idx = G_train[edge_type]["edge_index"]
                G_train[edge_type]["edge_index"] = torch.cat([old_idx, rand_idx], dim=1)
                try:
                    B_rand, x_dicts_rand = box_embeddings_from_model(model, G_train, box=MinDeltaBoxTensor)
                finally:
                    G_train[edge_type]["edge_index"] = old_idx
            else:
                G_rand = deepcopy(G_train)
                G_rand[edge_type]["edge_index"] = torch.cat([G_rand[edge_type]["edge_index"], rand_idx], dim=1)
                B_rand, x_dicts_rand = box_embeddings_from_model(model, G_rand, box=MinDeltaBoxTensor)

            # Constrained random
            crs = get_random_subclass(get_superclass(source, gci), gci)
            crt = get_random_subclass(
                get_superclass(
                    target,
                    gci,
                    person=edge_type[1] in ["parentOf", "fatherOf", "motherOf"],
                ),
                gci,
            )
            crand_idx = torch.tensor([[crs, crt]], device=dev).T
            if NO_DEEPCOPY:
                old_idx = G_train[edge_type]["edge_index"]
                G_train[edge_type]["edge_index"] = torch.cat([old_idx, crand_idx], dim=1)
                try:
                    B_crand, x_dicts_crand = box_embeddings_from_model(model, G_train, box=MinDeltaBoxTensor)
                finally:
                    G_train[edge_type]["edge_index"] = old_idx
            else:
                G_crand = deepcopy(G_train)
                G_crand[edge_type]["edge_index"] = torch.cat([G_crand[edge_type]["edge_index"], crand_idx], dim=1)
                B_crand, x_dicts_crand = box_embeddings_from_model(model, G_crand, box=MinDeltaBoxTensor)

            # make sure GPU work is finished before moving to CPU
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass

            # compute metrics
            dist_val = embedding_distance(B, B_star).detach().item()
            pos_neg = [l.detach().item() for l in box_loss(x_dicts_star, gci["gci0"], loss_type=LOSS_TYPE, neg=True, neg_data=gci["gci1_bot"], neg_random_weight=0)]
            rand_val = embedding_distance(B, B_rand).detach().item()
            rand_pos_neg = [l.detach().item() for l in box_loss(x_dicts_rand, gci["gci0"], loss_type=LOSS_TYPE, neg=True, neg_data=gci["gci1_bot"], neg_random_weight=0)]
            crand_val = embedding_distance(B, B_crand).detach().item()
            crand_pos_neg = [l.detach().item() for l in box_loss(x_dicts_crand, gci["gci0"], loss_type=LOSS_TYPE, neg=True, neg_data=gci["gci1_bot"], neg_random_weight=0)]

            out = {
                "idx": idx,
                "edge": [str(edge_type[0]), str(edge_type[1]), int(source.detach().item()), int(target.detach().item())],
                "distance": [dist_val] + pos_neg,
                "random_edge": [int(rs), int(rt)],
                "random": [rand_val] + rand_pos_neg,
                "constrained_edge": [int(crs), int(crt)],
                "constrained_random": [crand_val] + crand_pos_neg,
                "timestamp": time.time(),
                "shard": SHARD_ID,
            }
            _write_result(out)

    # collect edges
    edges = []
    for edge_type, edge_details in tqdm(G_test.edge_items(), desc="Collecting test edges", file=sys.stderr):
        if edge_type[1].startswith("rev_"): continue
        for s, t in edge_details["edge_index"].T:
            edges.append((edge_type, s, t))

    # process sequentially (workers clamped to 1)
    processed = 0
    for idx, (edge_type, source, target) in tqdm(
        enumerate(edges),
        desc="Processing test edges",
        total=len(edges) if MAX_TEST_EDGES < 0 else min(MAX_TEST_EDGES, len(edges)),
        file=sys.stderr
    ):
        if idx % NUM_SHARDS != SHARD_ID:
            continue
        if processed >= MAX_TEST_EDGES and MAX_TEST_EDGES > 0:
            break
        try:
            _process_edge(idx, edge_type, source, target)
        except Exception as e:
            print(f"Error processing edge {idx}: {e}", file=sys.stderr)
        processed += 1

    # close results file
    try:
        results_fp.flush()
        os.fsync(results_fp.fileno())
    except Exception:
        pass
    results_fp.close()

    # Aggregate JSONL results into final pickle (small memory usage at the end)
    distances = {}
    random_distances = {}
    constrained_random_distances = {}
    with open(results_log, "r") as fi:
        for line in fi:
            try:
                d = json.loads(line)
                et = (d["edge"][0], d["edge"][1])
                s = d["edge"][2]
                t = d["edge"][3]
                distances[(et, s, t)] = dict(zip(["distance", "pos_loss", "neg_loss"], d["distance"]))
                re = d["random_edge"]
                random_distances[(et, re[0], re[1])] = dict(zip(["distance", "pos_loss", "neg_loss"], d["random"]))
                cre = d["constrained_edge"]
                constrained_random_distances[(et, cre[0], cre[1])] = dict(zip(["distance", "pos_loss", "neg_loss"], d["constrained_random"]))
            except Exception as e:
                print(f"Failed to parse result line: {e}", file=sys.stderr)

    with open(os.path.join(output_dir, "link_eval_data.pkl"), "wb") as fo:
        pickle.dump(
            {
                "distances": distances,
                "random_distances": random_distances,
                "constrained_random_distances": constrained_random_distances,
            },
            fo,
        )

    # Save the 
