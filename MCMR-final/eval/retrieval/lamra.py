import os
import json
import math
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from contextlib import nullcontext

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor, AutoTokenizer, AutoImageProcessor
from transformers.utils.versions import require_version

                    
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

                                        
try:
    require_version(
        "transformers<4.52.0",
        "GME has limited compatibility with transformers>=4.52.0. Recommended: pip install 'transformers==4.51.3'"
    )
except Exception:
    pass


                                                              
CONFIG: Dict[str, Any] = dict(
          
    MODEL_DIR = "/code/.cache/huggingface/gme-Qwen2-VL-7B-Instruct",

                        
    CANDIDATES = [
        "/code/multimodal_retrieval/MCMR-final/candidates.jsonl",
    ],
    IMAGE_DIR = "/code/multimodal_retrieval/MCMR-final/images",

          
    QUERIES = [
        "/code/multimodal_retrieval/MCMR-final/query.jsonl",
    ],
    T2I_PROMPT = "Find an image-text pair that matches the given query",
    MAX_QUERIES = None,                            
    
             
    TOPK_LIST = [1, 5, 10, 50, 100],
    EVAL_ONLY_COVERED = True,                               

          
    BATCH_SIZE = 32,                               
    AMP = True,                      
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu",
    OFFLINE = True,

               
    MAX_TXT_CHARS = 2048,
    MIN_IMAGE_HW = 28,

          
    DUMP_TOPK_JSON = "gme_direct_eval_top10.json",
    EXPORT_JSON = True,
    PREVIEW_FIRST_N = 3,
)
                                                                  

                
if CONFIG["OFFLINE"]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

DEVICE = CONFIG["DEVICE"]
AMP_CTX = torch.autocast("cuda", dtype=torch.float16) if (DEVICE == "cuda" and CONFIG["AMP"]) else nullcontext()


                                                    
def json_lines(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s: yield json.loads(s)

def iter_candidates(paths):
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        p = Path(path)
        if not p.exists(): continue
        
        if p.suffix.lower() == ".jsonl":
            yield from json_lines(path)
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, list):
                for it in obj: yield it
            elif isinstance(obj, dict):
                if isinstance(obj.get("data"), list):
                    for it in obj["data"]: yield it
                elif isinstance(obj.get("items"), list):
                    for it in obj["items"]: yield it
                else: yield obj
            else:
                yield from json_lines(path)
        except Exception:
            yield from json_lines(path)

def extract_filename_from_url(url: str) -> Optional[str]:
    if not url: return None
    return url.split("/")[-1] or None

def pick_first_image_path(obj: Dict[str, Any], image_dir: str) -> Optional[Path]:
    imgs = obj.get("images") or []
    url = None
    for it in imgs:
        url = it.get("url") or it.get("large") or it.get("hi_res")
        if url: break
    if not url: return None
    fn = extract_filename_from_url(url)
    if not fn: return None
    p = Path(image_dir) / fn
    return p if p.exists() else None

def render_text(obj: Dict[str, Any], max_chars: int) -> str:
    """Build product text: title | description | features + price/date."""
    title = str(obj.get("title") or "").strip()
    desc  = obj.get("description") or []
    if isinstance(desc, list): desc = " ".join([str(x) for x in desc if x])
    else: desc = str(desc or "")
    feats = obj.get("features") or []
    if isinstance(feats, list): feats = " ".join([str(x) for x in feats if x])
    else: feats = str(feats or "")
    price = obj.get("price")
    price_str = f" price: {price}" if (price is not None) else ""
    dfa = obj.get("Date First Available") or obj.get("date") or ""
    dfa_str = f" date: {dfa}" if dfa else ""
    txt = " | ".join([s for s in [title, desc, feats] if s])
    txt = (txt + price_str + dfa_str).strip()
    return txt[:max_chars]

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


                                                    
def load_gme(model_dir: str):
    """Load the GME model using the official trust_remote_code interface."""
    print(f"[model] Loading GME from: {model_dir}")
    
                                      
    try:
        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    except Exception:
        processor = None

    model = AutoModel.from_pretrained(
        model_dir, 
        trust_remote_code=True, 
        local_files_only=True,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None
    )
    model.to(DEVICE).eval()

    if not hasattr(model, "get_fused_embeddings") or not hasattr(model, "get_text_embeddings"):
        raise RuntimeError("Model weights do not expose get_fused_embeddings/get_text_embeddings. Please verify this is a GME checkpoint.")
        
    return model


                                                                
def encode_fused_batch_gme(model, texts: List[str], pil_images: List[Image.Image]) -> np.ndarray:
    """Extract fused image-text embeddings for candidates."""
    with torch.no_grad(), AMP_CTX:
        emb = model.get_fused_embeddings(texts=texts, images=pil_images)
        if isinstance(emb, (list, tuple)):
            emb = emb[0]
        if not isinstance(emb, torch.Tensor):
            raise RuntimeError("get_fused_embeddings did not return a tensor.")
        vec = emb.detach().float().cpu().numpy()
        return l2_normalize_np(vec)

def encode_queries_gme(model, texts: List[str], instruction: str, batch_size: int) -> np.ndarray:
    """Extract text-only embeddings for queries."""
    all_vecs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding Queries"):
        batch = texts[i:i + batch_size]
        with torch.no_grad(), AMP_CTX:
            emb = model.get_text_embeddings(texts=batch, instruction=instruction)
            if isinstance(emb, (list, tuple)):
                emb = emb[0]
            if isinstance(emb, np.ndarray):
                vec = emb
            else:
                vec = emb.detach().float().cpu().numpy()
            all_vecs.append(l2_normalize_np(vec))
    return np.vstack(all_vecs)


                                                  
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
        out[k] = {"Hit@k": hits / M if M else 0.0, "NDCG@k": float(ndcgs / M) if M else 0.0}
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
    return s / M if M else 0.0


                                                  
def load_queries_any(qspec: Any, max_q: Optional[int]):
    queries, pos_sets, qids, origins = [], [], [], []
    if not qspec: return queries, pos_sets, qids, origins
    paths = qspec if isinstance(qspec, (list, tuple)) else [qspec]
    
    for p in paths:
        if not p or not Path(p).exists(): continue
        src_tag = Path(p).name
        for obj in json_lines(p):
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

def load_and_encode_candidates(model) -> Tuple[np.ndarray, List[Dict]]:
    """Load and batch-encode all valid candidates with display metadata."""
    c_vecs = []
    c_metas = []
    
    batch_texts, batch_imgs, batch_meta_tmp = [], [], []
    skipped_small = 0
    total_seen = 0
    
    def _flush():
        nonlocal batch_texts, batch_imgs, batch_meta_tmp
        if not batch_texts: return
        vecs = encode_fused_batch_gme(model, batch_texts, batch_imgs)
        c_vecs.append(vecs)
        c_metas.extend(batch_meta_tmp)
        batch_texts.clear(); batch_imgs.clear(); batch_meta_tmp.clear()

    print("[candidates] Reading and Encoding candidates...")
    for obj in tqdm(iter_candidates(CONFIG["CANDIDATES"]), desc="Encoding Candidates"):
        total_seen += 1
        cid = str(obj.get("candidate_id") or "")
        if not cid: continue

        img_path = pick_first_image_path(obj, CONFIG["IMAGE_DIR"])
        if img_path is None: continue

        text = render_text(obj, CONFIG["MAX_TXT_CHARS"])
        if not text: continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception: continue

                               
        w, h = img.size
        if min(w, h) < CONFIG.get("MIN_IMAGE_HW", 28):
            skipped_small += 1
            continue
            
                    
        imgs_meta = obj.get("images") or []
        first_url = ""
        for it in imgs_meta:
            url = it.get("url") or it.get("large") or it.get("hi_res")
            if url: 
                first_url = str(url).strip()
                break

        batch_texts.append(text)
        batch_imgs.append(img)
        batch_meta_tmp.append({
            "candidate_id": cid,
            "title": obj.get("title"),
            "price": obj.get("price"),
            "date": obj.get("Date First Available") or obj.get("date"),
            "image_url": first_url,
        })

        if len(batch_texts) >= CONFIG["BATCH_SIZE"]:
            _flush()

    _flush()
    print(f"[candidates] Seen: {total_seen}, Valid encoded: {len(c_metas)}, Skipped small images: {skipped_small}")
    
    if len(c_vecs) == 0:
        raise RuntimeError("No valid candidates were loaded!")
        
    return np.vstack(c_vecs), c_metas


                                                 
def main():
    print("===== GME Direct Eval Started (FAISS) =====")
    
             
    model = load_gme(CONFIG["MODEL_DIR"])
    
                 
    C, c_metas = load_and_encode_candidates(model)
    in_index_cids = set(m["candidate_id"] for m in c_metas)
    
                 
    queries, pos_sets, qids, origins = load_queries_any(CONFIG["QUERIES"], CONFIG["MAX_QUERIES"])
    if not queries:
        print("[warn] No valid queries loaded. Exit.")
        return

    cover_flags = [(len(ps & in_index_cids) > 0) for ps in pos_sets]
    cover_rate = sum(cover_flags) / len(cover_flags) if cover_flags else 0.0
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

               
    Q = encode_queries_gme(model, queries, CONFIG["T2I_PROMPT"], CONFIG["BATCH_SIZE"])
    
                                    
    print("[search] Building Faiss index and retrieving top-K...")
    max_k = max(10, max(CONFIG["TOPK_LIST"]))
    top_k_req = min(max_k, len(c_metas))
    faiss_index = build_faiss_ip_index(C)
    scores_np, indices_np = faiss_topk_search(faiss_index, Q, top_k_req)

                
    all_rels = []
    dump_records = []
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

    print("\n===== Retrieval Metrics (GME Direct Eval) =====")
    print(f"[config] EVAL_ONLY_COVERED={eval_only}")
    for k in ks_needed:
        print(f"TopK@{k:<3}= {metrics[k]['Hit@k']*100:.2f}%  NDCG@{k:<3}= {metrics[k]['NDCG@k']*100:.2f}%")
    print(f"MRR@10  = {mrr10*100:.2f}%")
    print(f"Total queries (eval/all): {total_queries_eval}/{total_queries_all}")
    print(f"Total candidates encoded: {len(c_metas)}")
    print("===============================================")

                
    if CONFIG.get("EXPORT_JSON", True):
        metrics_block = {
            "MRR@10": mrr10 * 100.0,
            "coverage_pos_in_index": (float(sum(1 for s in pos_sets if (s & in_index_cids))) / len(pos_sets) * 100.0) if pos_sets else 0.0,
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
                "model_dir": CONFIG["MODEL_DIR"],
                "queries_path": CONFIG["QUERIES"],
                "candidates_path": CONFIG["CANDIDATES"],
                "t2i_prompt": CONFIG["T2I_PROMPT"],
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

if __name__ == "__main__":
    main()
