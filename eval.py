import argparse
import json
import os

import numpy as np
from utils import load_data
import traceback
from collections import Counter, defaultdict


def store_edges(quads):
    """
    Store all edges for each relation.

    Parameters:
        quads (np.ndarray): indices of quadruples

    Returns:
        edges (dict): edges for each relation
    """

    edges = dict()
    relations = list(set(quads[:, 1]))
    for rel in relations:
        edges[rel] = quads[quads[:, 1] == rel]

    return edges

def baseline_candidates(test_query_rel, edges, obj_dist, rel_obj_dist):
    """
    Define the answer candidates based on the object distribution as a simple baseline.

    Parameters:
        test_query_rel (int): test query relation
        edges (dict): edges from the data on which the rules should be learned
        obj_dist (dict): overall object distribution
        rel_obj_dist (dict): object distribution for each relation

    Returns:
        candidates (dict): candidates along with their distribution values
    """

    if test_query_rel in edges:
        candidates = rel_obj_dist[test_query_rel]
    else:
        candidates = obj_dist

    return candidates.copy()

def calculate_obj_distribution(learn_data, edges):
    """
    Calculate the overall object distribution and the object distribution for each relation in the data.

    Parameters:
        learn_data (np.ndarray): data on which the rules should be learned
        edges (dict): edges from the data on which the rules should be learned

    Returns:
        obj_dist (dict): overall object distribution
        rel_obj_dist (dict): object distribution for each relation
    """

    objects = learn_data[:, 2]
    dist = Counter(objects)
    for obj in dist:
        dist[obj] /= len(learn_data)
    obj_dist = {k: round(v, 6) for k, v in dist.items()}
    obj_dist = dict(sorted(obj_dist.items(), key=lambda item: item[1], reverse=True))

    rel_obj_dist = dict()
    for rel in edges:
        objects = edges[rel][:, 2]
        dist = Counter(objects)
        for obj in dist:
            dist[obj] /= len(objects)
        rel_obj_dist[rel] = {k: round(v, 6) for k, v in dist.items()}
        rel_obj_dist[rel] = dict(
            sorted(rel_obj_dist[rel].items(), key=lambda item: item[1], reverse=True)
        )

    
    head_obj_dist = defaultdict(list)
    for query in learn_data:
        head = query[0]
        tail = query[2]
        head_obj_dist[head].append(tail)
    for k, v in head_obj_dist.items():
        total = len(v)
        head_obj_dist[k] = Counter(v)
        for obj in head_obj_dist[k]:
            head_obj_dist[k][obj] /= total
        head_obj_dist[k] = dict(
            sorted(head_obj_dist[k].items(), key=lambda item: item[1], reverse=True)
        )

    return obj_dist, head_obj_dist, rel_obj_dist


def filter_candidates(test_query, candidates, test_data):
    """
    Filter out those candidates that are also answers to the test query
    but not the correct answer.

    Parameters:
        test_query (np.ndarray): test_query
        candidates (dict): answer candidates with corresponding confidence scores
        test_data (np.ndarray): test dataset

    Returns:
        candidates (dict): filtered candidates
    """

    other_answers = test_data[
        (test_data[:, 0] == test_query[0])
        * (test_data[:, 1] == test_query[1])
        * (test_data[:, 2] != test_query[2])
        * (test_data[:, 3] == test_query[3])
    ]

    if len(other_answers):
        objects = other_answers[:, 2]
        for obj in objects:
            candidates.pop(obj, None)

    return candidates

def calculate_rank(test_query_answer, candidates, num_entities, setting="average"):
    """
    Calculate the rank of the correct answer for a test query.
    Depending on the setting, the average/best/worst rank is taken if there
    are several candidates with the same confidence score.

    Parameters:
        test_query_answer (int): test query answer
        candidates (dict): answer candidates with corresponding confidence scores
        num_entities (int): number of entities in the dataset
        setting (str): "average", "best", or "worst"

    Returns:
        rank (int): rank of the correct answer
    """

    rank = None
    if test_query_answer in candidates:
        conf = candidates[test_query_answer]
        all_confs = sorted(list(candidates.values()), reverse=True)
        ranks = [idx for idx, x in enumerate(all_confs) if x == conf]
        if setting == "average":
            rank = (ranks[0] + ranks[-1]) // 2 + 1
        elif setting == "best":
            rank = ranks[0] + 1
        elif setting == "worst":
            rank = ranks[-1] + 1

    return rank

def eval(title, candidates_file, data, learn_edges, obj_dist, head_obj_dist, rel_obj_dist, num_entities):
    out = [title]
    print(title)
    
    all_candidates = {}
    if candidates_file:
        all_candidates = json.load(open(candidates_file))
        all_candidates = {int(k): v for k, v in all_candidates.items()}
        for k in all_candidates:
            all_candidates[k] = {int(cand): v for cand, v in all_candidates[k].items()}

    hits_1 = 0
    hits_3 = 0
    hits_10 = 0
    mrr = 0

    num_samples: int = len(data)

    for i, test_query in enumerate(data):
        if i not in all_candidates:
            print("No candidates for", i)
            num_samples = i
            break

        head = test_query[0]
        rel = test_query[1]

        # Embedding-based set up
        # candidates_lsts_names = ["candidates", "obj_dist"]
        # candidates_lsts = [all_candidates.get(i, {}).copy(), obj_dist.copy()]

        # TLogic set up
        # candidates_lsts_names = ["candidates", "rel_obj_dist", "obj_dist"]
        # candidates_lsts = [all_candidates.get(i, {}).copy(), rel_obj_dist.get(rel, {}).copy(), obj_dist.copy()]
 
        # Ours set up (2 variations)
        # HCO
        candidates_lsts_names = ["head_obj_dist", "candidates", "rel_obj_dist", "obj_dist"]
        candidates_lsts = [head_obj_dist.get(head, {}).copy(), all_candidates.get(i, {}).copy(), rel_obj_dist.get(rel, {}).copy(), obj_dist.copy()]
        # CHO
        # candidates_lsts_names = ["candidates", "head_obj_dist", "rel_obj_dist", "obj_dist"]
        # candidates_lsts = [all_candidates.get(i, {}).copy(), head_obj_dist.get(head, {}).copy(), rel_obj_dist.get(rel, {}).copy(), obj_dist.copy()]
        
        for a in range(len(candidates_lsts)):
            candidates_lst_a = candidates_lsts[a]
            for b in range(a+1, len(candidates_lsts)):
                candidates_lst_b = candidates_lsts[b]
                for rel in candidates_lst_a:
                    if rel in candidates_lst_b:
                        del candidates_lst_b[rel]

        add_to_rank = 0
        for i, candidates in enumerate(candidates_lsts):
            candidates = filter_candidates(test_query, candidates, all_data)
            rank = calculate_rank(test_query[2], candidates, num_entities)
            if rank:
                rank += add_to_rank
                break
            add_to_rank += len(candidates)

        if rank <= 10:
            hits_10 += 1
            if rank <= 3:
                hits_3 += 1
                if rank == 1:
                    hits_1 += 1
        mrr += 1 / rank
        
    print("N: ", num_samples)
    out.append(f"N: {num_samples}")

    print("Hits@1: ", round(hits_1, 6))
    out.append(f"Hits@1: {hits_1}")

    print("Hits@3: ", round(hits_3, 6))
    out.append(f"Hits@3: {hits_3}")

    print("Hits@10: ", round(hits_10, 6))
    out.append(f"Hits@10: {hits_10}")

    print("MRR: ", round(mrr, 6))
    out.append(f"MRR: {mrr}")

    if candidates_file:
        filename = candidates_file.replace(".json", "_eval.txt")
        with open(filename, "w", encoding="utf-8") as fout:
            fout.write("\n".join(out))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", default="zs_DuEE", type=str, choices=["zs_DuEE", "zs_Wiki", "zs_Gdelt"])
    parser.add_argument("--id_candidates_file", "-i", type=str)
    parser.add_argument("--ood_candidates_file", "-o", type=str)
    parser.add_argument('--baseline', action='store_true')
    args = parser.parse_args()
    
    data_path = os.path.join("data", args.dataset)

    with open(os.path.join(data_path, "entities2id.txt"), 'r') as freader:
        e2i = {row.split("\t")[0]:row.split("\t")[1][:-1] for row in freader.readlines()}
    with open(os.path.join(data_path, "relations2id.txt"), 'r') as freader:
        r2i = {row.split("\t")[0]:row.split("\t")[1][:-1] for row in freader.readlines()}
    i2e = {v:k for k, v in e2i.items()}
    i2r = {v:k for k, v in r2i.items()}

    if args.dataset in ["zs_DuEE", "zs_Gdelt"]:
        # DuEE file swaps the id and entities fields, so I swap them to be as expected here
        i2e, e2i = e2i, i2e
        i2r, r2i = r2i, i2r

    num_relations = len(i2r)
    ri2inv = {int(r): num_relations + int(r) for r in i2r.keys()}
    
    
    train = load_data(data_path, "train.txt", ri2inv, True)
    # Load test data
    id_test = load_data(data_path, "val.txt", ri2inv)
    ood_test = load_data(data_path, "test.txt", ri2inv)
    
    # Convert all timestamps
    ts = set()
    # Get all train entities
    seen_ent = set()
    for quad in train:
        ts.add(quad[-1])
        seen_ent.add(quad[0])
        seen_ent.add(quad[2])
    for quad in id_test:
        ts.add(quad[-1])
    for quad in ood_test:
        ts.add(quad[-1])
    ts_conversion = {t: i for i, t in enumerate(sorted(list(ts)))}

    train = [quad[:-1] + [ts_conversion[quad[-1]]] for quad in train]
    train = np.array(train)
    id_test = [quad[:-1] + [ts_conversion[quad[-1]]] for quad in id_test]
    id_test = np.array(id_test)
    ood_test = [quad[:-1] + [ts_conversion[quad[-1]]] for quad in ood_test]
    ood_test = np.array(ood_test)
    all_data = np.concatenate([train, id_test, ood_test])

    id_candidates_file = ""
    ood_candidates_file = ""
    if not args.baseline:
        id_candidates_file = args.id_candidates_file
        ood_candidates_file = args.ood_candidates_file

    learn_edges = store_edges(train)
    num_entities = len(i2e)
    obj_dist, head_obj_dist, rel_obj_dist = calculate_obj_distribution(train, learn_edges)

    try:
        if args.baseline or id_candidates_file: 
            print(id_candidates_file)
            eval("In-Domain Evaluation", id_candidates_file, id_test, learn_edges, obj_dist, head_obj_dist, rel_obj_dist, num_entities)
    except Exception as e:
        print(traceback.format_exc())
        print(e)
    try:
        if args.baseline or ood_candidates_file: 
            print(ood_candidates_file)
            eval("Out-Domain Evaluation", ood_candidates_file, ood_test, learn_edges, obj_dist, head_obj_dist, rel_obj_dist, num_entities)
    except Exception as e:
        print(traceback.format_exc())
        print(e)



