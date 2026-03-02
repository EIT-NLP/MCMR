import os
import sys
import json
import math
from typing import List, Dict, Any, Optional, Tuple, Iterable
from pathlib import Path
from contextlib import nullcontext

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/code/.cache/huggingface")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

                                                                                                           
try:
    from transformers import dynamic_module_utils as _dm
    _orig_get_cached_module_file = _dm.get_cached_module_file
    def _local_get_cached_module_file(path_or_repo_id, filename, *args, **kwargs):
        try:
            if str(path_or_repo_id) == "microsoft/Phi-3.5-vision-instruct":
                local_dir = "/code/.cache/huggingface/Phi-3.5-vision-instruct"
                local_path = os.path.join(local_dir, filename)
                if os.path.isfile(local_path):
                    return local_path
        except Exception:
            pass
        return _orig_get_cached_module_file(path_or_repo_id, filename, *args, **kwargs)
    _dm.get_cached_module_file = _local_get_cached_module_file
    print("[hf] Patched dynamic_module_utils.get_cached_module_file for microsoft/Phi-3.5-vision-instruct")
except Exception as e:
    print(f"[hf] Patch dynamic module resolver failed: {e}")



CONFIG: Dict[str, Any] = dict(

    VLM2VEC_REPO="/code/multimodal_retrieval/VLM2Vec",  
    MODEL_DIR="/code/.cache/huggingface/VLM2Vec-Full",
    PROCESSOR=None,                                     
    

    CANDIDATES = [
        "/code/multimodal_retrieval/MCMR-final/candidates.jsonl",
    ],
    IMAGE_DIR = "/code/multimodal_retrieval/MCMR-final/images",

    TEXT_KEYS=dict(
        title="title",
        desc="description",
        feat="features",
        price="price",
        date="Date First Available"
    ),

          
    QUERIES = [
        "/code/multimodal_retrieval/MCMR-final/query.jsonl",
    ],
    MAX_QUERIES=None,                            
    
            
    TORCH_DTYPE="bfloat16",
    MAX_LENGTH=128,               
    NUM_CROPS_TGT=4,                  
    NUM_CROPS_QRY=1,                          
    POOLING="last",
    NORMALIZE=True,

             
    TOPK_LIST=[1, 5, 10, 50, 100],
    EVAL_ONLY_COVERED=True,

          
    BATCH_SIZE=2,                                      
    DEVICE="cuda" if torch.cuda.is_available() else "cpu",

          
    DUMP_TOPK_JSON="vlm2vec_direct_eval_top10.json",
    EXPORT_JSON=True,
    PREVIEW_FIRST_N=3,
)
                                                                  

DEVICE = CONFIG["DEVICE"]
amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16 if CONFIG["TORCH_DTYPE"] == "bfloat16" else torch.float16) if DEVICE == "cuda" else nullcontext()

          
if CONFIG.get("VLM2VEC_REPO") and CONFIG["VLM2VEC_REPO"] not in sys.path:
    sys.path.insert(0, CONFIG["VLM2VEC_REPO"])
from src.arguments import ModelArguments as _VArgs
from src.model import MMEBModel as _MMEBModel
from src.model_utils import load_processor as _load_processor, process_vlm_inputs_fns as _process_vlm_inputs_fns, vlm_image_tokens as _vlm_image_tokens


                                                    
def jsonl_reader(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            yield json.loads(ln)

def iter_candidates(paths):
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        p = Path(path)
        if not p.exists(): continue
        
        if p.suffix.lower() == ".jsonl":
            yield from jsonl_reader(path)
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
                yield from jsonl_reader(path)
        except Exception:
            yield from jsonl_reader(path)

def l2_normalize_np(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return (v / (n + eps)).astype("float32")

def move_tensors_to_device(batch: Dict, device: str) -> Dict:
    """仅将张量移动到 device，保持 list/tuple 原样（与官方实现兼容）。"""
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        elif isinstance(v, (list, tuple)):
            out[k] = v
        else:
            out[k] = v
    return out

def extract_filename_from_url(url: str) -> Optional[str]:
    if not url: return None
    base = url.split("?")[0]
    return os.path.basename(base)

def extract_image_source(obj: Dict) -> str:
    """尝试从多个键提取图片资源路径/URL"""
    keys_try = ["image", "image_url", "img", "image_path", "imagePath", "imageName", "image_name", "images"]
    for k in keys_try:
        if k not in obj: continue
        v = obj.get(k)
        if not v: continue
        if isinstance(v, str): return v
        if isinstance(v, list):
            for it in v:
                if isinstance(it, str) and it: return it
                if isinstance(it, dict):
                    for kk in ("url", "image", "path", "image_url", "large", "hi_res"):
                        if isinstance(it.get(kk), str) and it.get(kk): return it.get(kk)
        if isinstance(v, dict):
            for kk in ("url", "image", "path", "image_url", "large", "hi_res"):
                if isinstance(v.get(kk), str) and v.get(kk): return v.get(kk)
    return ""

def find_image_gme(image_root: str, url_or_name: str) -> Optional[str]:
    """GME 风格找图，支持递归子目录匹配。"""
    if not url_or_name: return None
    
                 
    if ("/" in url_or_name or os.sep in url_or_name):
        cand = url_or_name if os.path.isabs(url_or_name) else os.path.join(image_root, url_or_name)
        if os.path.isfile(cand): return cand

    basename = extract_filename_from_url(url_or_name) or url_or_name
    p = os.path.join(image_root, basename)
    if os.path.isfile(p): return p

               
    for root, _, files in os.walk(image_root):
        if basename in files:
            fp = os.path.join(root, basename)
            if os.path.isfile(fp): return fp

                     
    stem, _ = os.path.splitext(basename)
    if stem:
        for root, _, files in os.walk(image_root):
            for fn in files:
                if os.path.splitext(fn)[0] == stem:
                    fp = os.path.join(root, fn)
                    if os.path.isfile(fp): return fp
    return None

def render_text(obj: Dict, keys: Dict[str, str]) -> str:
    """将候选集字段拼成一段纯文本"""
    title = str(obj.get(keys["title"], "") or "")
    desc  = obj.get(keys["desc"], []) or []
    if isinstance(desc, str): desc = [desc]
    feat  = obj.get(keys["feat"], []) or []
    if isinstance(feat, str): feat = [feat]
    price = obj.get(keys["price"], None)
    date  = obj.get(keys["date"], None)

    parts = []
    if title: parts.append(f"Title: {title}")
    if desc: parts.append("Description: " + " ".join([x.strip() for x in desc if x]))
    if feat: parts.append("Features: " + " ; ".join([x.strip() for x in feat if x]))
    if price is not None: parts.append(f"Price: {price}")
    if date: parts.append(f"FirstAvailable: {date}")
    return " ".join(parts).strip()


                                                             
class VLM2VecManager:
    """包装 VLM2Vec 模型，以便后续同时支持 query (纯文本) 和 target (图文) 的特征提取"""
    def __init__(self, model_path: str, processor_name: str, device="cuda", pooling="last", normalize=True, num_crops_tgt=4, num_crops_qry=1):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dummy_img = Image.new("RGB", (4, 4), 0)
        
                              
        margs_tgt = _VArgs(
            model_name=model_path,
            processor_name=processor_name,
            pooling=pooling,
            normalize=normalize,
            num_crops=num_crops_tgt,
        )
        
                             
        margs_qry = _VArgs(
            model_name=model_path,
            processor_name=processor_name,
            pooling=pooling,
            normalize=normalize,
            num_crops=num_crops_qry,
        )
        
        print(f"[model] Loading VLM2Vec model from: {model_path}")
        self.model = _MMEBModel.load(margs_tgt).to(self.device).eval()
        
        self.processor_tgt = _load_processor(margs_tgt)
        self.processor_qry = _load_processor(margs_qry)
        
        self.backbone = margs_tgt.model_backbone
        self.process_fn = _process_vlm_inputs_fns[self.backbone]
        self.image_token = _vlm_image_tokens.get(self.backbone, "")
        
                                                              
        self.amp_ctx_qry = nullcontext() if self.backbone == "phi3_v" else amp_ctx

    @torch.no_grad()
    def encode_tgt_batch(self, texts: List[str], images: List[Image.Image], max_length: int) -> np.ndarray:
        """提取候选集图文融合特征 (Target 侧)"""
                          
        if self.image_token:
            texts_in = [f"{self.image_token} {t}".strip() for t in texts]
        else:
            texts_in = texts
            
        inputs = self.process_fn(
            model_inputs={"text": texts_in, "image": images},
            processor=self.processor_tgt,
            max_length=max_length
        )
        inputs = move_tensors_to_device(inputs, str(self.device))
        
        with amp_ctx:
            out = self.model(tgt=inputs)              
            reps = out["tgt_reps"]
            vec = reps.detach().cpu().to(torch.float32).numpy()
            return l2_normalize_np(vec)

    @torch.no_grad()
    def encode_qry_batch(self, texts: List[str], max_length: int) -> np.ndarray:
        """提取查询集文本特征 (Query 侧)"""
        dummy_imgs = [self.dummy_img for _ in texts]
        
        if self.image_token:
            texts_in = [f"{self.image_token} {t}".strip() for t in texts]
        else:
            texts_in = texts
            
        inputs = self.process_fn(
            model_inputs={"text": texts_in, "image": dummy_imgs},
            processor=self.processor_qry,
            max_length=max_length
        )
        inputs = move_tensors_to_device(inputs, str(self.device))
        
        with self.amp_ctx_qry:
            out = self.model(qry=inputs)              
            reps = out["qry_reps"]
            vec = reps.detach().cpu().to(torch.float32).numpy()
            return l2_normalize_np(vec)

                                                  
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
        for obj in jsonl_reader(p):
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

def load_and_encode_candidates(manager: VLM2VecManager) -> Tuple[np.ndarray, List[Dict]]:
    """读取并批量编码全部合法候选集，保留展示用的元数据"""
    c_vecs = []
    c_metas = []
    
    batch_texts, batch_imgs, batch_meta_tmp = [], [], []
    skipped_miss = 0
    skipped_err = 0
    total_seen = 0
    
    def _flush():
        nonlocal batch_texts, batch_imgs, batch_meta_tmp
        if not batch_texts: return
                         
        vecs = manager.encode_tgt_batch(batch_texts, batch_imgs, max_length=CONFIG["MAX_LENGTH"])
        c_vecs.append(vecs)
        c_metas.extend(batch_meta_tmp)
        batch_texts.clear(); batch_imgs.clear(); batch_meta_tmp.clear()

    print("[candidates] Reading and Encoding candidates...")
    for obj in tqdm(iter_candidates(CONFIG["CANDIDATES"]), desc="Encoding Candidates"):
        total_seen += 1
        cid = str(obj.get("candidate_id") or obj.get("id") or obj.get("asin") or "")
        if not cid: continue

        img_src = extract_image_source(obj)
        img_path = find_image_gme(CONFIG["IMAGE_DIR"], img_src)
        
        if not img_path:
            skipped_miss += 1
            continue

        text = render_text(obj, CONFIG["TEXT_KEYS"])

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception: 
            skipped_err += 1
            continue

        batch_texts.append(text)
        batch_imgs.append(img)
        batch_meta_tmp.append({
            "candidate_id": cid,
            "title": obj.get("title"),
            "price": obj.get("price"),
            "date": obj.get("Date First Available") or obj.get("date"),
            "image_url": img_src,
        })

        if len(batch_texts) >= CONFIG["BATCH_SIZE"]:
            _flush()

    _flush()
    print(f"[candidates] Seen: {total_seen}, Valid encoded: {len(c_metas)}")
    print(f"[candidates] Skipped missing image: {skipped_miss}, Skipped loading error: {skipped_err}")
    
    if len(c_vecs) == 0:
        raise RuntimeError("No valid candidates were loaded!")
        
    return np.vstack(c_vecs), c_metas


                                                 
def main():
    print("===== VLM2Vec Direct Eval Started (No FAISS) =====")
    
                
    processor_name = CONFIG["PROCESSOR"] or CONFIG["MODEL_DIR"]
    manager = VLM2VecManager(
        model_path=CONFIG["MODEL_DIR"],
        processor_name=processor_name,
        device=CONFIG["DEVICE"],
        pooling=CONFIG["POOLING"],
        normalize=CONFIG["NORMALIZE"],
        num_crops_tgt=CONFIG["NUM_CROPS_TGT"],
        num_crops_qry=CONFIG["NUM_CROPS_QRY"]
    )
    
                            
    C, c_metas = load_and_encode_candidates(manager)
    in_index_cids = set(m["candidate_id"] for m in c_metas)
    
                 
    queries, pos_sets, qids, origins = load_queries_any(CONFIG["QUERIES"], CONFIG["MAX_QUERIES"])
    if not queries:
        print("[warn] 未读取到有效的 query, 退出.")
        return

    cover_flags = [(len(ps & in_index_cids) > 0) for ps in pos_sets]
    cover_rate = sum(cover_flags) / (len(cover_flags) or 1)
    print(f"[coverage] Queries with >=1 positive in candidates: {sum(cover_flags)}/{len(cover_flags)} ({cover_rate:.1%})")

    eval_only = bool(CONFIG.get("EVAL_ONLY_COVERED", False))
    total_queries_all = len(cover_flags)
    
    if eval_only:
        kept = [(q, ps, qid, org) for (q, ps, qid, org, keep) in zip(queries, pos_sets, qids, origins, cover_flags) if keep]
        if not kept:
            print("[filter] WARN: 没有可评测的 queries（正样本均不在候选集中）。")
            return
        queries, pos_sets, qids, origins = zip(*kept)
        queries, pos_sets, qids, origins = list(queries), list(pos_sets), list(qids), list(origins)
        print(f"[filter] Kept {len(queries)}/{total_queries_all} covered queries.")
        
    total_queries_eval = len(queries)

                         
    print("[search] Encoding queries...")
    all_q_vecs = []
    for i in tqdm(range(0, len(queries), CONFIG["BATCH_SIZE"]), desc="Encoding Queries"):
        batch_q = queries[i:i + CONFIG["BATCH_SIZE"]]
        q_vec = manager.encode_qry_batch(batch_q, max_length=CONFIG["MAX_LENGTH"])
        all_q_vecs.append(q_vec)
    Q = np.vstack(all_q_vecs)
    
                                    
    print("[search] Computing similarities and retrieving top-K...")
    max_k = max(10, max(CONFIG["TOPK_LIST"]))
    top_k_req = min(max_k, len(c_metas))
    
    q_tensor = torch.tensor(Q, device=DEVICE)
    c_tensor = torch.tensor(C, device=DEVICE)
    
                                                
    sims = q_tensor @ c_tensor.T 
    scores_tensor, indices_tensor = torch.topk(sims, k=top_k_req, dim=1)
    
    scores_np = scores_tensor.cpu().numpy()
    indices_np = indices_tensor.cpu().numpy()

                
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

    print("\n===== Retrieval Metrics (VLM2Vec Direct Eval) =====")
    print(f"[config] EVAL_ONLY_COVERED={eval_only}")
    for k in ks_needed:
        print(f"TopK@{k:<3}= {metrics[k]['Hit@k']*100:.2f}%  NDCG@{k:<3}= {metrics[k]['NDCG@k']*100:.2f}%")
    print(f"MRR@10  = {mrr10*100:.2f}%")
    print(f"Total queries (eval/all): {total_queries_eval}/{total_queries_all}")
    print(f"Total candidates encoded: {len(c_metas)}")
    print("=====================================================")

                
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
                "model_dir": CONFIG["MODEL_DIR"],
                "queries_path": CONFIG["QUERIES"],
                "candidates_path": CONFIG["CANDIDATES"],
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
    torch.set_grad_enabled(False)
    main()