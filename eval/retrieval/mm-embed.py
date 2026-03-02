import os
import sys
import json
import math
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from contextlib import nullcontext

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from transformers import AutoModel

                      
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

def die(msg: str, code: int = 2):
    print(f"[FATAL] {msg}")
    sys.exit(code)

                                                              
CONFIG: Dict[str, Any] = dict(
          
    MODEL_DIR="/code/.cache/huggingface/MM-Embed",
    RETRIEVER_DIR="/code/.cache/huggingface/NV-Embed-v2",                            
    
                        
    CANDIDATES = [
        "/code/multimodal_retrieval/MCMR-final/candidates.jsonl",
    ],
    IMAGE_DIR = "/code/multimodal_retrieval/MCMR-final/images",

          
    QUERIES = [
        "/code/multimodal_retrieval/MCMR-final/query.jsonl",
    ],
    INSTRUCTION="Retrieve the product that best matches the query description.",
    MAX_LENGTH=4096,                   
    MAX_QUERIES=None,                            
    
             
    TOPK_LIST=[1, 5, 10, 50, 100],
    EVAL_ONLY_COVERED=True,                               

          
    BATCH_SIZE=32,
    MIN_IMAGE_HW=28,
    DEVICE="cuda" if torch.cuda.is_available() else "cpu",
    OFFLINE=True,

          
    DUMP_TOPK_JSON="mm_embed_direct_eval_top10.json",
    EXPORT_JSON=True,
    PREVIEW_FIRST_N=3,
)
                                                                  

                
if CONFIG["OFFLINE"]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

DEVICE = CONFIG["DEVICE"]

                                                    
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

def extract_filename_from_url(url: Optional[str]) -> Optional[str]:
    if not url or not isinstance(url, str): return None
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

def render_text(obj: Dict[str, Any], max_chars: int = 2048) -> str:
    title = str(obj.get("title") or "").strip()
    desc = obj.get("description") or []
    if isinstance(desc, list):
        desc = " ".join([str(x) for x in desc if x])
    else:
        desc = str(desc or "")
    feats = obj.get("features") or []
    if isinstance(feats, list):
        feats = " ".join([str(x) for x in feats if x])
    else:
        feats = str(feats or "")
    price = obj.get("price")
    price_str = f" price: {price}" if (price is not None) else ""
    date = obj.get("Date First Available") or obj.get("date") or ""
    date_str = f" date: {date}" if date else ""
    txt = " | ".join([s for s in [title, desc, feats] if s])
    txt = (txt + price_str + date_str).strip()
    return txt[:max_chars]

def l2_normalize_np(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return (v / (n + eps)).astype("float32")


                                                      
def check_mmembed_config(model_dir: str, retriever_dir: str):
    cfg_path = Path(model_dir) / "config.json"
    if not cfg_path.exists():
        die(f"未找到 {cfg_path}，请确认 MM-Embed 已正确下载。")
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as e:
        die(f"解析 {cfg_path} 失败：{e}")

    retriever = cfg.get("retriever")
    if retriever != retriever_dir:
        die(
            "MM-Embed/config.json 的 'retriever' 未指向本地 NV-Embed-v2 根目录。\n"
            f"  当前: {retriever}\n"
            f"  期望: {retriever_dir}\n"
            "请修改后重试（保持离线环境）。"
        )

    text_cfg = cfg.get("text_config") or {}
    t_path = text_cfg.get("_name_or_path")
    if t_path != model_dir:
        die(
            "MM-Embed/config.json 的 'text_config._name_or_path' 必须改为 MM-Embed 根目录（使用本地分词器）。\n"
            f"  当前: {t_path}\n"
            f"  期望: {model_dir}\n"
            "请修改后重试（保持离线环境）。"
        )

    tok_json = Path(model_dir) / "tokenizer.json"
    if not tok_json.exists():
        die(f"缺少本地分词器文件：{tok_json}，请确保 MM-Embed 根目录内含 tokenizer.json。")

    if not Path(retriever_dir).exists():
        die(f"未找到本地 NV-Embed-v2 目录：{retriever_dir}")


                                                    
def load_mmembed(model_dir: str):
    """加载 MM-Embed 模型并包含热修复逻辑"""
    print(f"[model] load: {model_dir}")
    model = AutoModel.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype="auto",
        attn_implementation="eager",
    )
    
    if DEVICE == "cuda":
        model = model.to("cuda")
        
    if not hasattr(model, "encode"):
        die("当前模型不支持 model.encode(...) 接口。请检查 MM-Embed 版本/权重。")

                                                              
    try:
        pp = getattr(model, "preprocess_fn", None)
        if pp is not None:
            ps = getattr(pp, "patch_size", None)
            if ps is None:
                vcfg = getattr(model, "config", None)
                vcfg = getattr(vcfg, "vision_config", None)
                ps_val = getattr(vcfg, "patch_size", None) or 14
                setattr(pp, "patch_size", int(ps_val))
                print(f"[hotfix] set preprocess_fn.patch_size = {int(ps_val)}")

            img_size = None
            vcfg = getattr(model, "config", None)
            vcfg = getattr(vcfg, "vision_config", None)
            if vcfg is not None:
                img_size = getattr(vcfg, "image_size", None)
            ip = getattr(pp, "image_processor", None)
            if (ip is not None) and (img_size is not None):
                try:
                    size = getattr(ip, "size", None)
                    if isinstance(size, dict):
                        if size.get("shortest_edge") != img_size:
                            size["shortest_edge"] = img_size
                            print(f"[hotfix] set image_processor.size.shortest_edge = {img_size}")
                except Exception:
                    pass
    except Exception as e:
        print(f"[hotfix] preprocess_fn warmup skipped: {e}")

    model.eval()
    return model


                                                                
@torch.no_grad()
def encode_fused_batch_mmembed(model, texts: List[str], images: List[Image.Image]) -> np.ndarray:
    """提取候选集图文融合向量"""
    if len(texts) != len(images):
        die(f"编码批大小不一致：texts={len(texts)} vs images={len(images)}")

    samples = [{"txt": t, "img": im} for t, im in zip(texts, images)]
    out = model.encode(samples, is_query=False, max_length=CONFIG["MAX_LENGTH"])
    
    if not isinstance(out, dict) or ("hidden_states" not in out):
        die("model.encode(...) 返回值中未找到 'hidden_states' 字段。请检查 MM-Embed 版本/接口。")

    emb = out["hidden_states"]
    if isinstance(emb, torch.Tensor):
        emb = emb.detach().float().cpu().numpy()
    elif isinstance(emb, np.ndarray):
        pass
    else:
        die(f"'hidden_states' 类型不支持：{type(emb)}，需为 torch.Tensor 或 np.ndarray")

                            
    return l2_normalize_np(emb)

@torch.no_grad()
def encode_queries_mmembed(model, queries: List[str], batch_size: int, instruction: str, max_length: int) -> np.ndarray:
    """提取纯文本 Query 向量"""
    all_vecs = []
    for i in tqdm(range(0, len(queries), batch_size), desc="Encoding Queries"):
        batch = queries[i:i+batch_size]
        samples = [{"txt": q} for q in batch]
        
        out = model.encode(samples, is_query=True, instruction=instruction, max_length=max_length)
        
        if isinstance(out, dict) and "hidden_states" in out:
            vec = out["hidden_states"]
            if isinstance(vec, torch.Tensor):
                vec = vec.detach().float().cpu().numpy()
            elif isinstance(vec, np.ndarray):
                vec = vec.astype("float32")
            else:
                raise TypeError("Unexpected hidden_states type from MM-Embed.encode")
        else:
            raise RuntimeError("MM-Embed.encode didn't return 'hidden_states'")
            
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
    """读取并批量编码全部合法候选集，保留展示用的元数据"""
    c_vecs = []
    c_metas = []
    
    batch_texts, batch_imgs, batch_meta_tmp = [], [], []
    skipped_small = 0
    skipped_ioerr = 0
    skipped_emptytxt = 0
    skipped_nourl = 0
    total_seen = 0
    
    def _flush():
        nonlocal batch_texts, batch_imgs, batch_meta_tmp
        if not batch_texts: return
        vecs = encode_fused_batch_mmembed(model, batch_texts, batch_imgs)
        c_vecs.append(vecs)
        c_metas.extend(batch_meta_tmp)
        batch_texts.clear(); batch_imgs.clear(); batch_meta_tmp.clear()

    print("[candidates] Reading and Encoding candidates...")
    for obj in tqdm(iter_candidates(CONFIG["CANDIDATES"]), desc="Encoding Candidates"):
        total_seen += 1
        cid = str(obj.get("candidate_id") or "")
        if not cid: continue

        img_path = pick_first_image_path(obj, CONFIG["IMAGE_DIR"])
        if img_path is None: 
            skipped_nourl += 1
            continue

        text = render_text(obj, max_chars=2048)
        if not text: 
            skipped_emptytxt += 1
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception: 
            skipped_ioerr += 1
            continue

        w, h = img.size
        if min(w, h) < CONFIG["MIN_IMAGE_HW"]:
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
    print(f"[candidates] Seen: {total_seen}, Valid encoded: {len(c_metas)}")
    print(f"[candidates] Skipped: small={skipped_small}, ioerr={skipped_ioerr}, empty_txt={skipped_emptytxt}, no_url={skipped_nourl}")
    
    if len(c_vecs) == 0:
        raise RuntimeError("No valid candidates were loaded!")
        
    return np.vstack(c_vecs), c_metas


                                                 
def main():
    print("===== MM-Embed Direct Eval Started (No FAISS) =====")
    
              
    check_mmembed_config(CONFIG["MODEL_DIR"], CONFIG["RETRIEVER_DIR"])
    
             
    model = load_mmembed(CONFIG["MODEL_DIR"])
    
                 
    C, c_metas = load_and_encode_candidates(model)
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

               
    Q = encode_queries_mmembed(
        model=model,
        queries=queries,
        batch_size=int(CONFIG.get("BATCH_SIZE") or 32),
        instruction=str(CONFIG.get("INSTRUCTION") or ""),
        max_length=int(CONFIG.get("MAX_LENGTH") or 4096),
    )
    
                                    
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

    print("\n===== Retrieval Metrics (MM-Embed Direct Eval) =====")
    print(f"[config] EVAL_ONLY_COVERED={eval_only}")
    for k in ks_needed:
        print(f"TopK@{k:<3}= {metrics[k]['Hit@k']*100:.2f}%  NDCG@{k:<3}= {metrics[k]['NDCG@k']*100:.2f}%")
    print(f"MRR@10  = {mrr10*100:.2f}%")
    print(f"Total queries (eval/all): {total_queries_eval}/{total_queries_all}")
    print(f"Total candidates encoded: {len(c_metas)}")
    print("====================================================")

                
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
                "instruction": CONFIG["INSTRUCTION"],
                "max_length": CONFIG["MAX_LENGTH"],
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