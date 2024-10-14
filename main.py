# -*- coding: utf-8 -*-

import argparse
from collections import defaultdict
from enum import Enum
import itertools
import json
import os
import numpy as np
import transformers
import torch
from score_functions import score_12
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd


class Language(Enum):
    ZH = "Chinese"
    EN = "English"

MAX_DOCS = 10

ours_versions = ["enrichEntityRel", "enrichRel"]
ablation_versions = ["", "noTranslate", "noGenDoc", "noEnrichRel"]

prompts = {
    Language.ZH: {
        "subject_query": lambda docs_str, s: f"{docs_str}\n---\n给定上述文件, 请简要向我介绍实体{s}。",
        "relation_query": lambda s, r: f"请简要概述关系{r} 在{s} 对其他一些实体存在{r} 关系这一情境中的意思。",
        "translate_query": lambda s, r:  f"在这句话中什么是UNKNOWN ENTITY: [{s}] [{r}] UNKNOWN_ENTITY?",
        "natural_language_query": lambda translate_query:  f"给定上述情境, 将下列问题用自然语言重写。只要求给出重写的版本，不要给出其他任何内容。不要加任何引号: \"{translate_query}\"",
        "hypothetical_query_no_context": lambda query:  f"基于你的事实性知识，生成一段话以简要回答下述问题: {query}",
        "hypothetical_query_context": lambda query:  f"给定上述情境，基于你的事实性知识，生成一段话以简要回答下述问题: {query}",
        "expl_context": lambda doc_context, nl_query:  f"{doc_context}\n---\n给定上述情境, 请回答这一问题: {nl_query} 请包含你的思考，并在解释中引用提供文件中的相关信息。",
        "eval_context": lambda nl_query, expl:  f"评估如下解释是否是对: {nl_query} 的良好回答。如果是好的回答，只要求写出\"YES\", 如果不是好的回答，只写出\"NO\"。不要阐述理由。\n題解: {expl}",
        "expl_query": lambda nl_query, entity, relation: f"哪个实体最能回答问题{nl_query}。通过如下形式\"[{entity}] [{relation}] [x]\" 来回答，其中x 是回答需要的目标实体。",
    },
    Language.EN: {
        "subject_query": lambda docs_str, s: f"{docs_str}\n---\nGiven the above documents, please concisely tell me about the entity {s}.",
        "relation_query": lambda s, r: f"Please concisely describe the meaning of the relation {r} in the context of {s} being {r} some other entity.",
        "translate_query": lambda s, r:  f"What is UNKNOWN ENTITY in this sentence: [{s}] [{r}] UNKNOWN_ENTITY?",
        "natural_language_query": lambda translate_query:  f"Given the context above, rewrite the following question in natural language. Give the rewritten version only and don't say anything extra. Do not add any quotes: \"{translate_query}\"",
        "hypothetical_query_no_context": lambda query:  f"Generate a paragraph to concisely answer the following question grounded in your factual knowledge: {query}",
        "hypothetical_query_context": lambda query:  f"Given the context above, generate a paragraph to concisely answer the following question grounded in your factual knowledge: {query}",
        "expl_context": lambda doc_context, nl_query:  f"{doc_context}\n---\nGiven the above context, please answer this question: {nl_query} Include your thinking and in your explanation, reference the information in the provided documents.",
        "eval_context": lambda nl_query, expl:  f"Evaluate whether the following explanation is a good response to: {nl_query} Write only \"YES\" if it is a good response, or \"NO\" if it is not a good response. Do not elaborate.\nExplanation: {expl}",
        "expl_query": lambda nl_query, entity, relation: f"Which entity best answers the question {nl_query} Respond in the format \"[{entity}] [{relation}] [x]\" where x is the answer target entity.",  
    }
}

# Simplified version of vector DB impl based on HyperDB
class DB:
    def __init__(
        self,
        embedding_function=None,
    ):
        self.documents = []
        self.vectors = None
        self.embedding_function = embedding_function
    
    def get_norm_vector(self, vector):
        if len(vector.shape) == 1:
            return vector / np.linalg.norm(vector)
        else:
            return vector / np.linalg.norm(vector, axis=1)[:, np.newaxis]

    def cosine_similarity(self, vectors, query_vector):
        norm_vectors = self.get_norm_vector(vectors)
        norm_query_vector = self.get_norm_vector(query_vector)
        similarities = np.dot(norm_vectors, norm_query_vector.T)
        return similarities

    def add_document(self, document: dict):
        vector = self.embedding_function([document])
        if self.vectors is None:
            self.vectors = np.empty((0, len(vector)), dtype=np.float32)
        elif len(vector) != self.vectors.shape[1]:
            raise ValueError("All vectors must have the same length.")
        self.vectors = np.vstack([self.vectors, vector]).astype(np.float32)
        self.documents.append(document)

    def add_documents(self, documents, vectors=None):
        if not documents:
            return
        vectors = vectors or np.array(self.embedding_function(documents)).astype(
            np.float32
        )
        for vector, document in zip(vectors, documents):
            self.add_document(document, vector)

    def save(self, storage_file):
        if self.vectors is None or self.documents is None:
            return

        data = {"vectors": self.vectors, "documents": self.documents}
        if storage_file.endswith(".gz"):
            with gzip.open(storage_file, "wb") as f:
                pickle.dump(data, f)
        else:
            with open(storage_file, "wb") as f:
                pickle.dump(data, f)

    def load(self, storage_file):
        if storage_file.endswith(".gz"):
            with gzip.open(storage_file, "rb") as f:
                data = pickle.load(f)
        else:
            with open(storage_file, "rb") as f:
                data = pickle.load(f)
        self.vectors = data["vectors"].astype(np.float32)
        self.documents = data["documents"]

    def query(self, query_text, top_k=5):
        if self.vectors is None:
            return []

        query_vector = self.embedding_function([query_text])[0]

        similarities = self.cosine_similarity(vectors, query_vector)
        top_indices = np.argsort(similarities, axis=0)[-top_k:][::-1]

        ranked_results = top_indices.flatten()
        similarities = similarities[top_indices].flatten()

        return list(zip([self.documents[index] for index in ranked_results], similarities))

def load_data(data_path, path, ri2inv, inv=False):
        fwd = [[int(v) for v in line.strip().split("\t")] for line in open(os.path.join(data_path, path)).readlines()]
        if inv:
            inv_lst = [[line[2], ri2inv[line[1]], line[0], line[3]] for line in  fwd]
            fwd.extend(inv_lst)
        return fwd

def convert_to_natural_language(version, ablation, query, e2doc, prompt_language, descr_cache):
    s = i2e[str(query[0])]
    r = i2r[str(query[1])]

    full_history = []

    if str(query[0]) in e2doc:
        docs = e2doc[str(query[0])][:MAX_DOCS]
    else:
        docs = []
    docs_str = "\n---\n".join([doc[:1000] for doc in docs])

    if version == "enrichEntityRel":
        # Ask LLM to generate description of subject and relation as context to rewrite the query.
        subject_query = prompts[prompt_language]["subject_query"] (docs_str, s)
        full_history .append(
            {"role": "user", "content": subject_query},
        )
        if s in descr_cache:
            subject_descr = descr_cache[s]["description"]
        else:
            subject_descr = resp(full_history)
            descr_cache[s] = {}
            descr_cache[s]["description"] = subject_descr
        full_history.append({"role": "assistant", "content": subject_descr})
    elif version == "enrichRel":
        if s not in descr_cache:
            descr_cache[s] = {}
        
    if ablation != "noEnrichRel":
        relation_query = prompts[prompt_language]["relation_query"] (s, r)
        full_history.append({"role": "user", "content": relation_query})
        if f"rel:{r}" in descr_cache[s]:
            relation_descr = descr_cache[s][f"rel:{r}"]
        else:
            relation_descr = resp(full_history)
            descr_cache[s][f"rel:{r}"] = relation_descr
        full_history.append({"role": "assistant", "content": relation_descr})

    translated_query = prompts[prompt_language]["translate_query"] (s, r)
    inp = prompts[prompt_language]["natural_language_query"] (translated_query)
    full_history.append({"role": "user", "content": inp})
    
    # Chinese LLM keeps adding "在这句话中，" in the response which confuses subsequent interactions.
    return resp(full_history).replace("在这句话中，", ""), full_history


def generate_hypothetical_document(base_version, query, context, prompt_language):
    if context and base_version == "enrichEntityRel":
        inp = prompts[prompt_language]["hypothetical_query_context"] (query)
    else:
        inp = prompts[prompt_language]["hypothetical_query_no_context"] (query)
    if not context:
        context = []
    messages = context
    messages.append({"role": "user", "content": inp})
    return resp(messages)


def linear_interpolate(cands_1, cands_2, alpha):
    final_cand = {}
    for k, v in cands_1.items():
        if k in cands_2:
            # Linearly interpolate
            final_cand[k] = v*alpha + cands_2[k]*(1-alpha)
        else:
            final_cand[k] = v*alpha
    for k, v in cands_2.items():
        if k not in cands_1:
            final_cand[k] = v*(1-alpha)
    return final_cand


def alg(version, query, results, rules, all_data, prompt_language, top_k, verbose):
    explainability = {}
    # Compute the relation relative frequency weighted by similarity score
    rel_ranking = defaultdict(list)
    total = 0
    for retrieved_docs, sim_score in results:
        # From the associated documents, find the associated KG edges
        edges = retrieved_docs["tuple"]
        for edge in edges:
            rel = edge[1]
            # Skip if relation has no logic rules
            if str(rel) in rules.keys():
                rel_ranking[rel].append(sim_score)
                total += sim_score

    if verbose:
        print("Relation ranking from docs:", rel_ranking, total)

    # Normalize
    rel_ranking = {k: sum(v) / total for k, v in rel_ranking.items()}

    if verbose:
        print("Normalized relation ranking from docs:", rel_ranking)
    explainability["normalized_relation_rank_from_doc"] = rel_ranking

    # Weighted sum of cand rankings using the relations' normalized frequency
    cand_ranking = defaultdict(list)
    rel_walks = {}
    for rel, weight in rel_ranking.items():
        rel_cand_ranking, walks = TLogic(query[0], rel, query[-1], all_data, top_k)
        rel_walks[rel] = walks
        if verbose:
            print("TLogic", query[0], rel, query[-1], "-->", rel_cand_ranking)
        for k, v in rel_cand_ranking.items():
            cand_ranking[k].append(v * weight)
    explainability["cand_rank_per_rel"] = cand_ranking
    explainability["rel_walks"] = rel_walks
    # Aggregrate candidate rankings
    mean_cand_ranking = {k: np.mean(v) for k, v in cand_ranking.items()}
    return mean_cand_ranking, explainability

def edges_by_relation(edges, relations=None):
    edges_dict = {}
    if not relations:
        relations = list(set(edges[:, 1]))
    for rel in relations:
        edges_dict[rel] = edges[edges[:, 1] == rel]
    return edges_dict


def filter_edges_by_ts(all_data, test_query_ts):
    mask = all_data[:, 3] < test_query_ts
    return edges_by_relation(all_data[mask])


def filter_edges_by_rule(rule, edges, test_query_head):
    rels = rule["body_rels"]
    # Match query subject and first body relation
    try:
        rel_edges = edges[rels[0]]
        mask = rel_edges[:, 0] == test_query_head
        new_edges = rel_edges[mask]
        walk_edges = [
            np.hstack((new_edges[:, 0:1], new_edges[:, 2:4]))
        ]  # [head, obj, ts]
        cur_targets = np.array(list(set(walk_edges[0][:, 1])))

        for body_rel in rels[1:]:
            # Match current targets and next body relation
            try:
                # Get all edges who has relation = next body relation
                rel_edges = edges[body_rel]
                # Keep all edges that has head entity = any of the tail entities from previous timestep
                mask = np.any(rel_edges[:, 0] == cur_targets[:, None], axis=0)
                new_edges = rel_edges[mask]
                walk_edges.append(
                    np.hstack((new_edges[:, 0:1], new_edges[:, 2:4]))
                )  # [sub, obj, ts]
                cur_targets = np.array(list(set(walk_edges[-1][:, 1])))
            except KeyError:
                walk_edges.append([])
                break
    except KeyError:
        walk_edges = [[]]
    return walk_edges


def get_walks(rule, walk_edges):
    df_edges = []
    df = pd.DataFrame(
        walk_edges[0],
        columns=["entity_" + str(0), "entity_" + str(1), "timestamp_" + str(0)],
        dtype=np.uint16,
    )  # Change type if necessary for better memory efficiency
    if not rule["var_constraints"]:
        del df["entity_" + str(0)]
    df_edges.append(df)

    for i in range(1, len(walk_edges)):
        df = pd.DataFrame(
            walk_edges[i],
            columns=["entity_" + str(i), "entity_" + str(i + 1), "timestamp_" + str(i)],
            dtype=np.uint16,
        )  # Change type if necessary
        df_edges.append(df)
    rule_walks = df_edges[0]
    for i in range(1, len(df_edges)):
        rule_walks = pd.merge(rule_walks, df_edges[i], on=["entity_" + str(i)])
        # timestamp constraint
        rule_walks = rule_walks[
            rule_walks["timestamp_" + str(i - 1)] <= rule_walks["timestamp_" + str(i)]
        ]
        if not rule["var_constraints"]:
            del rule_walks["entity_" + str(i)]

    for i in range(1, len(rule["body_rels"])):
        del rule_walks["timestamp_" + str(i)]

    if rule["var_constraints"]:
        for const in rule["var_constraints"]:
            for i in range(len(const) - 1):
                rule_walks = rule_walks[
                    rule_walks["entity_" + str(const[i])]
                    == rule_walks["entity_" + str(const[i + 1])]
                ]

    return rule_walks

def get_candidates(rule, rule_walks, test_query_ts, cands_dict, score_func, score_parameters):
    max_entity = "entity_" + str(len(rule["body_rels"]))
    cands = set(rule_walks[max_entity])

    for cand in cands:
        cands_walks = rule_walks[rule_walks[max_entity] == cand]
        score = score_func(rule, cands_walks, test_query_ts, *score_parameters).astype(
            np.float32
        )
        try:
            cands_dict[cand].append(score)
        except KeyError:
            cands_dict[cand] = [score]

    return cands_dict

def TLogic(query_head_entity, query_relation, query_ts, all_data, top_k, lmbda=0.1, a=0.5):
    score_parameters = [lmbda, a]
    # a controls lin interp between score1 & score2
    # score 1 = rule conf
    # score 2 = time diff between query and first body rel (oldest)
    score_func = score_12

    # key: candidate tail entity
    # value: List[rule i's score for candidate]
    cands_dict = {}

    cur_ts = query_ts
    edges = filter_edges_by_ts(all_data, cur_ts)

    explainability = []

    query_relation = str(query_relation)
    if query_relation in rules:
        for rule in rules[query_relation]:
            walk_edges = filter_edges_by_rule(rule, edges, query_head_entity)
            # If no part of the walk superset edges ends up being invalid (no edges)
            if 0 not in [len(x) for x in walk_edges]:
                rule_walks = get_walks(rule, walk_edges)

                explainability.append((rule, rule_walks.to_json(orient="records")))
                

                if not rule_walks.empty:
                    cands_dict = get_candidates(
                        rule,
                        rule_walks,
                        cur_ts,
                        cands_dict,
                        score_func,
                        score_parameters,
                    )
                    # For each candidate, list all scores with highest score first
                    cands_dict = {
                        x: sorted(cands_dict[x], reverse=True)
                        for x in cands_dict.keys()
                    }
                    # Sort candidates largest highest rule conf first
                    cands_dict = dict(
                        sorted(
                            cands_dict.items(),
                            key=lambda item: item[1],
                            reverse=True,
                        )
                    )
                    
                    top_k_scores = list(cands_dict.values())[:top_k]
                    # We stop the rule application when the number of different answer candidates |{c | ∃R : (c, f (R, c)) ∈ C}| is at least k.
                    # To potentially increase cands returned, can remove this check
                    unique_scores = list(
                        scores for scores, _ in itertools.groupby(top_k_scores)
                    )
                    if len(unique_scores) >= top_k:
                        break
        if len(cands_dict.keys()) > 0:
            # Calculate noisy-or scores
            # Aggregate all scores
            # x = rule score
            # 1-x = rule's "badness"
            # prod(1-x) = the lower the more bad scores the cand has
            # 1 - prod(1-x) = the higher the less bad scores the cand has
            scores = list(
                map(
                    lambda x: 1 - np.product(1 - np.array(x)),
                    cands_dict.values(),
                )
            )
            # Candidate to aggregated score dict
            cands_scores = dict(zip(cands_dict.keys(), scores))
            # Rank cand to have highest score first
            noisy_or_cands = dict(
                sorted(cands_scores.items(), key=lambda x: x[1], reverse=True)
            )
            return noisy_or_cands, explainability
    # No candidates found by applying rules
    return {}, explainability

def rank_cand(base_version, ablation, query, rules, all_data, db, e2doc, cache, cache_key, save_cache, query_artifacts, query_key, save_query, descr_cache, save_descr_cache, N, TLogic_top_k, prompt_language, verbose):
    print("QUERY", i2e[str(query[0])], i2r[str(query[1])], i2e[str(query[2])])
    if ablation != "noTranslate":
        # Convert query to a natural language question.
        nl_query = cache.get(cache_key, {}).get("natural_query", None)
        nl_query_history = cache.get(cache_key, {}).get("query_generation_history", None)
        if not nl_query:
            nl_query, nl_query_history = convert_to_natural_language(version, ablation, query, e2doc, prompt_language, descr_cache)
            save_descr_cache()
    else:
        nl_query_history = None
        nl_query = f"[{i2e[str(query[0])]}] [{i2r[str(query[1])]}] [UNKNOWN ENTITY]"

    if verbose:
        print("Generated query:", nl_query)

    # For each query, generate hypothetical document answering the query and retrieve the top N documents
    if ablation == "noGenDoc":
        hypothetical_doc = nl_query
    else:
        hypothetical_doc = cache.get(cache_key, {}).get("document", None)
        if not hypothetical_doc:
            doc_context = nl_query_history
            if doc_context:
                doc_context = nl_query_history[:-1]
            hypothetical_doc = generate_hypothetical_document(base_version, nl_query, doc_context, prompt_language)
            if verbose:
                print(nl_query, "\n---\n", hypothetical_doc, "\n---\n", nl_query_history)

    if verbose:
        print("Generated document:", hypothetical_doc)

    # Retrieve documents from DB using the hypothetical document
    results = db.query(hypothetical_doc, top_k=N)

    if verbose:
        print("Retrieved documents:", [(doc["document"], score) for doc, score in results])
    
    # Save to cache
    cache[cache_key] = {
        "document": hypothetical_doc,
        "natural_query": nl_query,
        "query_generation_history": nl_query_history,
    }
    save_cache()

    if query_key not in query_artifacts:
        query_artifacts[query_key] = {}

    cands = {}
    if "kg_cands" in query_artifacts[query_key]:
        cands = query_artifacts[query_key]["kg_cands"]
        if verbose:
            print("Using cached KG candidates", cands)
    else:
        cands, explainability = alg(version, query, results, rules, all_data, prompt_language=prompt_language, top_k=TLogic_top_k, verbose=verbose)
        if verbose:
            print("Generated KG candidates", cands)
        save_cache()
        query_artifacts[query_key]["kg_cands"] = cands
        save_query()
    return cands

def create_db(dataset, embedding_model, ts_conversion, verbose):
    if not os.path.exists("db/"):
        os.makedirs("db/")

    db_name = os.path.join("db", f"{dataset}.pickle.gz")

    db = DB(embedding_function=embedding_model.encode)
    data_path = os.path.join("data", dataset)

    if os.path.exists(db_name):
        # Load the DB instance from the save file
        db.load(db_name)
    else:
        if verbose:
            print("Generating DB")
        docs = defaultdict(list)
        for line in open(os.path.join(data_path, "training_data.json"), encoding="utf-8").readlines():
            f = json.loads(line)
            docs[f["document"]].append(f)
        # Aggregate documents together so they are unique
        documents = [{
            "document": k,
            "token_ids": v[0]["token_ids"],
            "tuple": [item["tuple"][:-1] + [ts_conversion[item["tuple"][-1]]] for item in v]
        } for k, v in docs.items()]
        
        db.add_documents(documents)

        # Save the DB to a file
        db.save(db_name)
    return db

def test(base_version, ablation, fp, test_data, test_mapping, save_test_query, N, top_k, descr_cache, save_descr_cache, db, lang, replace, verbose):
    cand_rankings = {}
    if os.path.exists(fp) and not replace:
        with open(fp, "r", encoding="utf-8") as fout:
            cand_rankings = json.load(fout)
    for i, query in enumerate(test_data):
        print(f"{i+1} / {len(test_data)} queries for {fp}")
        if str(i) in cand_rankings:
            continue
        qk = "|".join([str(v) for v in query[:2]])
        cand_ranking = rank_cand(base_version, ablation, query, rules, all_data, db, e2doc, generation_mapping, qk, save_cache, test_mapping, str(i), save_test_query, descr_cache, save_descr_cache=save_descr_cache, N=N, TLogic_top_k=top_k, prompt_language=lang, verbose=verbose)
        save_cache()
        cand_rankings[i] = cand_ranking
        with open(fp, "w", encoding="utf-8") as fout:
            json.dump(cand_rankings, fout, indent=4)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", "-v", required=True, type=str, choices=ours_versions)
    parser.add_argument("--ablation", default="", type=str, choices=ablation_versions)
    parser.add_argument("--dataset", "-d", default="zs_DuEE", type=str, choices=["zs_DuEE", "zs_Wiki", "zs_Gdelt"])
    parser.add_argument("--N_docs", "-N", default=5, type=int)
    parser.add_argument("--top_k", default=20, type=int)
    parser.add_argument("--verbose", default=True, type=bool)
    parser.add_argument("--small_llm", default=False, type=bool)
    parser.add_argument("--replace", default=False, type=bool)
    args = parser.parse_args()
    
    if args.dataset in ["zs_DuEE"]:
        lang = Language.ZH
    elif args.dataset in ["zs_Wiki", "zs_Gdelt"]:
        lang = Language.EN
    else:
        raise Exception("Dataset not supported.")

    # Configuration for language models
    # Initialize model_name, pipeline, and resp
    if lang == Language.ZH:
        # Embedding model
        model_name = "uer/sbert-base-chinese-nli"
        # LLM model
        model_id = "shenzhi-wang/Llama3-70B-Chinese-Chat"  # Finetuned for Chinese language
        if args.small_llm:
            model_id = "shenzhi-wang/Llama3-8B-Chinese-Chat"

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto"
        )

        def pipeline(messages, **kwargs):
            input_ids = tokenizer.apply_chat_template(
                [messages], add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)
            
            return model.generate(
                input_ids,
                **kwargs
            )
        
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        def resp(messages):
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)

            outputs = model.generate(
                input_ids,
                max_new_tokens=256,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
            response = outputs[0][input_ids.shape[-1]:]
            return tokenizer.decode(response, skip_special_tokens=True)
    elif lang == Language.EN:
        # Embedding model
        model_name = "sentence-transformers/all-mpnet-base-v2"   
        # LLM model
        model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
        if args.small_llm:
            model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        def resp(messages):
            outputs = pipeline(
                messages,
                max_new_tokens=256,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
            return outputs[0]["generated_text"][-1]["content"]
    else:
        raise Exception("Language not supported.")

    base_model_name = model_id.split("/")[-1]
    embedding_model = SentenceTransformer(model_name)
    
    data_path = os.path.join("data", args.dataset)
    
    # Set up entity and relation mapping
    with open(os.path.join(data_path, "entities2id.txt"), "r", encoding="utf-8") as freader:
        e2i = {row.split("\t")[0]:row.split("\t")[1][:-1] for row in freader.readlines()}
    with open(os.path.join(data_path, "relations2id.txt"), "r", encoding="utf-8") as freader:
        r2i = {row.split("\t")[0]:row.split("\t")[1][:-1] for row in freader.readlines()}
    i2e = {v:k for k, v in e2i.items()}
    i2r = {v:k for k, v in r2i.items()}

    if args.dataset in ["zs_DuEE", "zs_Gdelt"]:
        # These datasets swap the id and entities fields, so I swap them to be as expected here
        i2e, e2i = e2i, i2e
        i2r, r2i = r2i, i2r

    # Load TLogic rules
    with open(os.path.join("rules", f"{args.dataset}_TLogic_Rules.json"), "r", encoding="utf-8") as freader:
        rules = json.load(freader)

    # Load entity to document mapping
    with open(os.path.join(data_path, f"train_e2doc.json"), "r", encoding="utf-8") as freader:
        e2doc = json.load(freader)
    
    # Set up relation inverse mapping
    num_r = len(i2r)
    ri2inv = {int(r): num_r + int(r) for r in i2r.keys()}

    for r, ir in ri2inv.items():
        i2r[str(ir)] = "_"+i2r[str(r)]
        r2i["_"+i2r[str(r)]] = str(ir)

    # Save entity embeddings
    entity_embeddings_mapping = {}
    entity_embeddings = []
    for i, e in i2e.items():
        embedding = embedding_model.encode(([e]))[0]
        entity_embeddings_mapping[tuple(embedding.tolist())] = i
        entity_embeddings.append(embedding)
    entity_embeddings = np.array(entity_embeddings)
    
    train = load_data(data_path, "train.txt", ri2inv, inv=True)

    # Load test data
    id_test = load_data(data_path, "val.txt", ri2inv)
    ood_test = load_data(data_path, "test.txt", ri2inv)

    # Convert all timestamps
    ts = set()
    for quad in train:
        ts.add(quad[-1])
    for quad in id_test:
        ts.add(quad[-1])
    for quad in ood_test:
        ts.add(quad[-1])
    ts_conversion = {t: i for i, t in enumerate(sorted(list(ts)))}

    train = [quad[:-1] + [ts_conversion[quad[-1]]] for quad in train]
    all_data = np.array(train)
    id_test = [quad[:-1] + [ts_conversion[quad[-1]]] for quad in id_test]
    ood_test = [quad[:-1] + [ts_conversion[quad[-1]]] for quad in ood_test]

    db = create_db(args.dataset, embedding_model, ts_conversion, args.verbose)

    generation_mapping = {}
    genmap_fp = f"{base_model_name}_v{args.version}_{args.dataset}_genmap.json"
    if os.path.exists(genmap_fp) and not args.replace:
        with open(genmap_fp, "r", encoding="utf-8") as fout:
            generation_mapping = json.load(fout)
    id_mapping = {}
    idmap_fp = f"{base_model_name}_v{args.version}_{args.dataset}_idmap.json"
    if os.path.exists(idmap_fp) and not args.replace:
        with open(idmap_fp, "r", encoding="utf-8") as fout:
            id_mapping = json.load(fout)
    ood_mapping = {}
    oodmap_fp = f"{base_model_name}_v{args.version}_{args.dataset}_oodmap.json"
    if os.path.exists(oodmap_fp) and not args.replace:
        with open(oodmap_fp, "r", encoding="utf-8") as fout:
            ood_mapping = json.load(fout)
    descr_cache = {}
    descr_cache_fp = f"{base_model_name}_v{args.version}_{args.dataset}_descriptions.json"
    if os.path.exists(descr_cache_fp) and not args.replace:
        with open(descr_cache_fp, "r", encoding="utf-8") as fout:
            descr_cache = json.load(fout)

    def save_cache():
        with open(genmap_fp, "w", encoding="utf-8") as fout:
            json.dump(generation_mapping, fout, indent=4)

    def save_query(query_mapping, query_fp):
        with open(query_fp, "w", encoding="utf-8") as fout:
            json.dump(query_mapping, fout, indent=4)

    test(args.version, args.ablation, f"{base_model_name}_v{args.version}_{args.dataset}_test_cand_rankings.json", ood_test, ood_mapping, lambda: save_query(ood_mapping, oodmap_fp), args.N_docs, args.top_k, descr_cache, lambda: save_query(descr_cache, descr_cache_fp), db, lang, args.replace, args.verbose)
    test(args.version, args.ablation, f"{base_model_name}_v{args.version}_{args.dataset}_val_cand_rankings.json", id_test, id_mapping,  lambda: save_query(id_mapping, idmap_fp), rgs.N_docs, args.top_k, descr_cache, lambda: save_query(descr_cache, descr_cache_fp), db, lang, args.replace, args.verbose)