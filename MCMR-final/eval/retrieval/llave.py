import os
import io
import re
import json
import math
import copy
import urllib.parse
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path, PurePath
from contextlib import nullcontext

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F

                      
os.environ.setdefault("HF_HOME", "/code/.cache/huggingface")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.environ["HF_HOME"])
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["WANDB_DISABLED"] = "true"

from transformers.utils import logging as hf_logging
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
hf_logging.set_verbosity_error()

                                             
import pdb as _pdb
os.environ.setdefault("PYTHONBREAKPOINT", "0")
try:
    _pdb.set_trace = (lambda *a, **k: None)
except Exception:
    pass


                                                              
CONFIG: Dict[str, Any] = dict(
          
    MODEL_ROOT="/code/.cache/huggingface/LLaVE-7B",
    
                        
    CANDIDATES = [
        "/code/multimodal_retrieval/MCMR-final/candidates.jsonl",
    ],
    IMAGE_DIR = "/code/multimodal_retrieval/MCMR-final/images",

          
    QUERIES = [
        "/code/multimodal_retrieval/MCMR-final/query.jsonl",
    ],
    MAX_QUERIES=None,                            
    CONV_TEMPLATE="qwen_1_5",            
    
             
    TOPK_LIST=[1, 5, 10, 50, 100],
    EVAL_ONLY_COVERED=True,                               

          
    BATCH_SIZE=32,
    AMP=True,
    DEVICE="cuda" if torch.cuda.is_available() else "cpu",

          
    DUMP_TOPK_JSON="llave_direct_eval_top10.json",
    EXPORT_JSON=True,
    
                            
    EXPORT_TOPK_JSONL=False,
    EXPORT_TOPK_JSONL_PATH="/code/multimodal_retrieval/test/model-test/llave-7b/eval/llave_direct_eval_topk50.jsonl",
    EXPORT_TOPK_K=50,
    
    PREVIEW_FIRST_N=3,
)
                                                                  

DEVICE = CONFIG["DEVICE"]
amp_ctx = torch.autocast("cuda", dtype=torch.float16) if (DEVICE == "cuda" and CONFIG["AMP"]) else nullcontext()


                                                                                  
try:
    from transformers import GenerationConfig, PretrainedConfig
    import transformers.generation.configuration_utils as _gc_mod

    def _safe_from_model_config(config):
        try:
            return _gc_mod._orig_from_model_config(config)
        except Exception:
            gc = getattr(config, "generation_config", None)
            if isinstance(gc, dict):
                return GenerationConfig.from_dict(gc)
            return GenerationConfig()

    if not hasattr(_gc_mod, "_orig_from_model_config"):
        _gc_mod._orig_from_model_config = _gc_mod.GenerationConfig.from_model_config
        _gc_mod.GenerationConfig.from_model_config = _safe_from_model_config                              

    _orig_get_text_config = PretrainedConfig.get_text_config
    def _safe_get_text_config(self, decoder: bool = False):
        cfg = _orig_get_text_config(self, decoder=decoder)
        if isinstance(cfg, str):
            base = getattr(self, "name_or_path", "")
            cfg_path = Path(base) / cfg if not PurePath(cfg).is_absolute() else Path(cfg)
            return PretrainedConfig.from_json_file(str(cfg_path))
        if isinstance(cfg, dict):
            return PretrainedConfig.from_dict(cfg)
        return cfg

    PretrainedConfig.get_text_config = _safe_get_text_config                            
except Exception as _e:
    pass

try:
    from transformers.configuration_utils import PretrainedConfig as _HFPretrainedConfig
    def _safe_repr(self): return f"{self.__class__.__name__}(safe_repr)"
    _HFPretrainedConfig.__repr__ = _safe_repr; _HFPretrainedConfig.__str__ = _safe_repr
except Exception:
    pass


                              
try:
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import tokenizer_image_token, process_images
except Exception as e:
    raise RuntimeError("llava package not found. Please install it according to official LLaVA/LLaVE instructions.") from e


                                                    
def jsonl_iter(path):
    try:
        import orjson as fastjson
        with open(path, "rb") as f:
            for line in f:
                if not line.strip(): continue
                yield fastjson.loads(line)
    except Exception:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s: continue
                yield json.loads(s)

def iter_candidates(paths):
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        p = Path(path)
        if not p.exists(): continue
        yield from jsonl_iter(path)

def url_basename(u: str) -> str:
    return os.path.basename(urllib.parse.urlparse(u).path)

def find_image_strict(image_dir: str, url: str) -> Optional[str]:
    p = os.path.join(image_dir, url_basename(url))
    return p if os.path.exists(p) else None

def pil_load_rgb(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(io.BytesIO(f.read()))
        return img.convert("RGB")

def join_fields(it: Dict[str, Any]) -> str:
    """Concatenate candidate text fields using the LLaVE/LLaVA convention."""
    title = it.get("title") or ""
    desc  = it.get("description") or []
    feats = it.get("features") or []
    price = it.get("price", None)
    dfa   = it.get("Date First Available") or it.get("date first available")
    desc_txt = " ".join(str(x) for x in desc if x) if isinstance(desc, list) else str(desc)
    feats_txt = " ".join(str(x) for x in feats if x) if isinstance(feats, list) else str(feats)
    price_txt = f"price {price}" if price is not None else ""
    dfa_txt   = f"date_first_available {dfa}" if dfa else ""
    text = ". ".join([title, desc_txt, feats_txt, price_txt, dfa_txt])
    return re.sub(r"\s+", " ", text).strip()

def _first_image_url(obj: Dict[str, Any]) -> str:
    imgs = obj.get("images") or []
    for it in imgs:
        url = it.get("url") or it.get("large") or it.get("hi_res")
        if isinstance(url, str) and url.strip():
            return url.strip()
    return ""

def l2_normalize_np(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return (v / (n + eps)).astype("float32")

def build_faiss_ip_index(c_vecs: np.ndarray):
    try:
        import faiss
    except Exception as e:
        raise RuntimeError("faiss is not installed. Please install faiss-cpu or faiss-gpu.") from e
    c_vecs = np.ascontiguousarray(c_vecs.astype("float32"))
    dim = int(c_vecs.shape[1])
    cpu_index = faiss.IndexFlatIP(dim)
    if DEVICE == "cuda" and hasattr(faiss, "StandardGpuResources"):
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        except Exception as e:
            print(f"[faiss] GPU index init failed, fallback to CPU: {e}")
            index = cpu_index
    else:
        index = cpu_index
    index.add(c_vecs)
    return index

def faiss_topk_search(index, q_vecs: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    q_vecs = np.ascontiguousarray(q_vecs.astype("float32"))
    scores, indices = index.search(q_vecs, int(k))
    return scores, indices


                                                    
def load_llave(model_root: str):
    print(f"[model] Loading LLaVE from: {model_root}")
    tok, model, img_proc, _ = load_pretrained_model(
        model_root, None, "llava_qwen", device_map="auto" if DEVICE == "cuda" else None
    )
    if isinstance(model.generation_config, dict):
        model.generation_config = GenerationConfig.from_dict(model.generation_config)
    model.to(DEVICE).eval()
    return tok, model, img_proc


                                                                
def build_prompt(text: str, has_image: bool, conv_tmpl: str) -> str:
    text = (text or "").strip()
    conv = copy.deepcopy(conv_templates[conv_tmpl])
    if has_image:
        prompt = f"{DEFAULT_IMAGE_TOKEN} Represent the given image with the following description: {text}" if text\
                 else f"{DEFAULT_IMAGE_TOKEN} Represent the given image."
    else:
        prompt = text if text else "Represent the given text."
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], "\n")
    return conv.get_prompt()

def encode_fused_batch_llave(tokenizer, model, image_processor, conv_tmpl: str, texts: List[str], imgs: List[Image.Image]) -> np.ndarray:
    """Extract fused image-text embeddings for candidates."""
    outs = []
    with torch.no_grad(), amp_ctx:
        for text, img in zip(texts, imgs):
            prompt = build_prompt(text, True, conv_tmpl)
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(DEVICE)
            attention_mask = input_ids.ne(tokenizer.pad_token_id)
            img_tensors = process_images([img], image_processor, model.config)
            img_tensors = [_im.to(dtype=torch.float16 if DEVICE=="cuda" else torch.float32, device=DEVICE) for _im in img_tensors]
            image_sizes = [img.size]
            
            emb = model.encode_multimodal_embeddings(input_ids, attention_mask=attention_mask,
                                                     images=img_tensors, image_sizes=image_sizes)
            emb = emb / (emb.norm(p=2, dim=-1, keepdim=True) + 1e-8)
            outs.append(emb.detach().cpu().numpy().astype("float32"))
            
    return np.vstack(outs)

def encode_queries_llave(tokenizer, model, texts: List[str], batch_size: int, conv_tmpl: str) -> np.ndarray:
    """Extract text-only embeddings for queries."""
    vecs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding Queries"):
        batch = texts[i:i + batch_size]
        conv_prompts = []
        for t in batch:
            conv = copy.deepcopy(conv_templates[conv_tmpl])
            conv.append_message(conv.roles[0], (t or "").strip(' "“”'))
            conv.append_message(conv.roles[1], "\n")
            conv_prompts.append(conv.get_prompt())
            
        enc = tokenizer(conv_prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = enc.input_ids.to(DEVICE)
        attn = input_ids.ne(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        
        with torch.no_grad(), amp_ctx:
            emb = model.encode_multimodal_embeddings(input_ids, attention_mask=attn)
        
        if emb.dim() == 3:                   
            emb = emb.mean(1)
            
        emb = emb / (emb.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        vecs.append(emb.detach().cpu().numpy().astype("float32"))
        
    return np.vstack(vecs)


                                                  
def dcg_at_k(rels: List[int], k: int) -> float:
    s = 0.0
    for i, r in enumerate(rels[:k], start=1):
        if r: s += 1.0 / math.log2(i + 1)
    return s

def eval_metrics(all_rels: List[List[int]], ks: List[int]) -> Dict[int, Dict[str, float]]:
    M = len(all_rels)
    out = {}
    for k in ks:
        hits = ndcgs = 0.0
        for rels in all_rels:
            if any(rels[:k]): hits += 1
            dcg = dcg_at_k(rels, k)
            idcg = dcg_at_k([1] * min(sum(rels), k), k)
            ndcgs += (dcg / idcg) if idcg > 0 else 0.0
        out[k] = {"Hit@k": hits / (M or 1), "NDCG@k": float(ndcgs / (M or 1))}
    return out

def mrr_at_k(all_rels: List[List[int]], k: int) -> float:
    M = len(all_rels)
    s = 0.0
    for rels in all_rels:
        try:
            pos = rels[:k].index(1)
            s += 1.0 / (pos + 1)
        except ValueError:
            pass
    return s / (M or 1)


                                                  
def load_queries_any(qspec: Any, max_q: Optional[int]):
    queries, pos_sets, qids, origins = [], [], [], []
    if not qspec: return queries, pos_sets, qids, origins
    paths = qspec if isinstance(qspec, (list, tuple)) else [qspec]
    
    for p in paths:
        if not p or not Path(p).exists(): continue
        src_tag = Path(p).name
        for obj in jsonl_iter(p):
            q = obj.get("query") or obj.get("text") or ""
            pos = [str(x) for x in (obj.get("pos_ids") or obj.get("positives") or [])]
            qid = obj.get("qid") or obj.get("id")
            if q.strip():
                queries.append(q.strip())
                pos_sets.append(set(pos))
                qids.append(qid)
                origins.append(src_tag)
                if isinstance(max_q, int) and len(queries) >= max_q: break
        if isinstance(max_q, int) and len(queries) >= max_q: break
    print(f"[queries] loaded {len(queries)} valid queries")
    return queries, pos_sets, qids, origins

def load_and_encode_candidates(tokenizer, model, image_processor) -> Tuple[np.ndarray, List[Dict]]:
    """Load and batch-encode all valid candidates."""
    c_vecs = []
    c_metas = []
    
    batch_texts, batch_imgs, batch_meta_tmp = [], [], []
    skipped_no_image = 0
    total_seen = 0
    
    def _flush():
        nonlocal batch_texts, batch_imgs, batch_meta_tmp
        if not batch_texts: return
        vecs = encode_fused_batch_llave(tokenizer, model, image_processor, CONFIG["CONV_TEMPLATE"], batch_texts, batch_imgs)
        c_vecs.append(vecs)
        c_metas.extend(batch_meta_tmp)
        batch_texts.clear(); batch_imgs.clear(); batch_meta_tmp.clear()

    print("[candidates] Reading and Encoding candidates...")
    for obj in tqdm(iter_candidates(CONFIG["CANDIDATES"]), desc="Encoding Candidates"):
        total_seen += 1
        cid = str(obj.get("candidate_id") or obj.get("id") or "")
        if not cid: continue

        imgs = obj.get("images") or []
        url  = (imgs[0].get("url") if imgs else None) or (imgs[0].get("large") if imgs else None) or (imgs[0].get("image") if imgs else None)
        if not url:
            skipped_no_image += 1; continue
            
        img_path = find_image_strict(CONFIG["IMAGE_DIR"], url)
        if not img_path:
            skipped_no_image += 1; continue

        try:
            img = pil_load_rgb(img_path)
        except Exception: 
            skipped_no_image += 1; continue

        text = join_fields(obj)

        batch_texts.append(text)
        batch_imgs.append(img)
        batch_meta_tmp.append({
            "candidate_id": cid,
            "title": obj.get("title"),
            "price": obj.get("price"),
            "date": obj.get("Date First Available") or obj.get("date"),
            "image_url": _first_image_url(obj),
            "raw": obj                      
        })

        if len(batch_texts) >= CONFIG["BATCH_SIZE"]:
            _flush()

    _flush()
    print(f"[candidates] Seen: {total_seen}, Valid encoded: {len(c_metas)}, Skipped missing/bad images: {skipped_no_image}")
    
    if len(c_vecs) == 0:
        raise RuntimeError("No valid candidates were loaded!")
        
    return np.vstack(c_vecs), c_metas


                                                 
def main():
    print("===== LLaVE Direct Eval Started (FAISS) =====")
    
             
    tokenizer, model, image_proc = load_llave(CONFIG["MODEL_ROOT"])
    
                 
    C, c_metas = load_and_encode_candidates(tokenizer, model, image_proc)
    in_index_cids = set(m["candidate_id"] for m in c_metas)
    
                 
    queries, pos_sets, qids, origins = load_queries_any(CONFIG["QUERIES"], CONFIG["MAX_QUERIES"])
    if not queries:
        print("[warn] No valid queries loaded. Exit.")
        return

    cover_flags = [(len(ps & in_index_cids) > 0) for ps in pos_sets]
    cover_rate = sum(cover_flags) / (len(cover_flags) or 1)
    print(f"[coverage] Queries with >=1 positive in candidates: {sum(cover_flags)}/{len(cover_flags)} ({cover_rate:.1%})")

    eval_only = bool(CONFIG.get("EVAL_ONLY_COVERED", False))
    total_queries_all = len(cover_flags)
    
    if eval_only:
        kept = [(q, ps, qid, org) for (q, ps, qid, org, keep) in zip(queries, pos_sets, qids, origins, cover_flags) if keep]
        if not kept:
            print("[filter] WARN: no evaluable queries (all positives are outside candidates).")
            return
        queries, pos_sets, qids, origins = zip(*kept)
        queries, pos_sets, qids, origins = list(queries), list(pos_sets), list(qids), list(origins)
        print(f"[filter] Kept {len(queries)}/{total_queries_all} covered queries.")
        
    total_queries_eval = len(queries)

               
    Q = encode_queries_llave(tokenizer, model, queries, CONFIG["BATCH_SIZE"], CONFIG["CONV_TEMPLATE"])
    
                                    
    print("[search] Building Faiss index and retrieving top-K...")
    max_k = max(10, max(CONFIG["TOPK_LIST"]), int(CONFIG.get("EXPORT_TOPK_K", 0)))
    top_k_req = min(max_k, len(c_metas))
    faiss_index = build_faiss_ip_index(C)
    scores_np, indices_np = faiss_topk_search(faiss_index, Q, top_k_req)

                
    all_rels = []
    dump_records = []
    jsonl_out_records = []
    preview_n = int(CONFIG.get("PREVIEW_FIRST_N") or 0)

    for qi, pos_set in enumerate(pos_sets):
        row_indices = indices_np[qi]
        row_scores = scores_np[qi]
        
        cids = [c_metas[idx]["candidate_id"] for idx in row_indices]
        rels = [1 if cid in pos_set else 0 for cid in cids]
        all_rels.append(rels)
        
        items = []
        for rank, (idx, sc, cid, is_pos) in enumerate(zip(row_indices, row_scores, cids, rels), start=1):
            meta = c_metas[idx]
            items.append({
                "rank": rank,
                "candidate_id": cid,
                "score": float(sc),
                "is_pos": int(is_pos),
                "image_url": meta.get("image_url", ""),
                "title": meta.get("title"),
                "price": meta.get("price"),
                "date": meta.get("date"),
                "candidate": meta.get("raw", {})
            })

        pos_in_index = sorted(list(pos_set & in_index_cids))
        pos_missed = [p for p in pos_in_index if p not in [it["candidate_id"] for it in items[:10]]]

        record = {
            "qid": qids[qi] if qids[qi] is not None else qi,
            "query": queries[qi],
            "origin": origins[qi],
            "positives": sorted(list(pos_set)),
            "positives_in_index": pos_in_index,
            "positives_missed_in_top10": pos_missed,
            "top10": items[:10],
        }
        dump_records.append(record)

                        
        if bool(CONFIG.get("EXPORT_TOPK_JSONL", False)):
            Kexp = int(CONFIG.get("EXPORT_TOPK_K", 100))
            topk_items = []
            for it in items[:Kexp]:
                topk_items.append({
                    "rank": it["rank"],
                    "candidate_id": it["candidate_id"],
                    "score": it["score"],
                    "is_pos": it["is_pos"],
                    "candidate": it.get("candidate", {}),
                })
            
            topk_cids = [it["candidate_id"] for it in items[:Kexp]]
            pos_in_topk = sorted([p for p in pos_set if p in set(topk_cids)])
            if pos_in_topk:                     
                jsonl_out_records.append({
                    "qid": qids[qi] if qids[qi] is not None else qi,
                    "query": queries[qi],
                    "positives": sorted(list(pos_set)),
                    "positives_in_topk": pos_in_topk,
                    "topk": topk_items
                })

        if preview_n > 0 and qi < preview_n:
            print("\n[preview] qid={}  query={}  origin={}".format(record["qid"], record["query"], origins[qi]))
            print("  positives:", record["positives"])
            print("  top10:")
            for it in items[:10]:
                tag = "✓POS" if it["is_pos"] == 1 else "  -  "
                print("   {:>2d}. {}  {:.4f}  {}  {}".format(
                    it["rank"], it["candidate_id"], it["score"], tag, it.get("title") or ""
                ))

             
    ks_needed = sorted(set(CONFIG["TOPK_LIST"]) | {1, 5, 10})
    metrics = eval_metrics(all_rels, ks_needed)
    mrr10 = mrr_at_k(all_rels, 10)

    print("\n===== Retrieval Metrics (LLaVE Direct Eval) =====")
    print(f"[config] EVAL_ONLY_COVERED={eval_only}")
    for k in ks_needed:
        print(f"TopK@{k:<3}= {metrics[k]['Hit@k']*100:.2f}%  NDCG@{k:<3}= {metrics[k]['NDCG@k']*100:.2f}%")
    print(f"MRR@10  = {mrr10*100:.2f}%")
    print(f"Total queries (eval/all): {total_queries_eval}/{total_queries_all}")
    print(f"Total candidates encoded: {len(c_metas)}")
    print("=================================================")

                
    if CONFIG.get("EXPORT_JSON", True):
        metrics_block = {
            "MRR@10": mrr10 * 100.0,
            "coverage_pos_in_index": cover_rate * 100.0,
            "eval_only_covered": eval_only,
            "num_queries_eval": total_queries_eval,
            "num_queries_all": total_queries_all,
            "index_total_vectors": len(c_metas),
        }
        for k in ks_needed:
            metrics_block[f"TopK@{k}"] = metrics[k]["Hit@k"] * 100.0
            metrics_block[f"NDCG@{k}"] = metrics[k]["NDCG@k"] * 100.0

        out_obj = {
            "config": {
                "model_dir": CONFIG["MODEL_ROOT"],
                "queries_path": CONFIG["QUERIES"],
                "candidates_path": CONFIG["CANDIDATES"],
                "conv_template": CONFIG["CONV_TEMPLATE"],
                "topk_list": CONFIG["TOPK_LIST"],
                "eval_only_covered": CONFIG["EVAL_ONLY_COVERED"],
            },
            "metrics": metrics_block,
            "results": dump_records
        }

        out_path = Path(CONFIG["DUMP_TOPK_JSON"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)
        print(f"[save] json dump -> {out_path}")

                      
    if bool(CONFIG.get("EXPORT_TOPK_JSONL", False)):
        out_jsonl = Path(CONFIG.get("EXPORT_TOPK_JSONL_PATH"))
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for rec in jsonl_out_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[save] topk-jsonl -> {out_jsonl}  records={len(jsonl_out_records)}  K={int(CONFIG.get('EXPORT_TOPK_K', 100))}")
        
        if jsonl_out_records:
            head_line = json.dumps(jsonl_out_records[0], ensure_ascii=False)
            head_file = out_jsonl.with_suffix(out_jsonl.suffix + ".head1")
            with open(head_file, "w", encoding="utf-8") as hf:
                hf.write(head_line + "\n")
            print(f"[preview] head(1) file -> {head_file}")

if __name__ == "__main__":
    main()
