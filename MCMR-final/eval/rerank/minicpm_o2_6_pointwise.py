import os, json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, AutoConfig
from transformers.tokenization_utils_base import BatchEncoding
from transformers.image_processing_utils import BatchFeature

                                                        
CONFIG = dict(
    MODEL_DIR = "/code/.cache/huggingface/MiniCPM-o-2_6",
    INPUT_TOPK_JSONL = "llave_eval_topk50.jsonl",
    OUT_POINTWISE_JSONL = "output.jsonl",

                                
    IMAGE_DIRS = [
    "/code/multimodal_retrieval/MCMR-final/images",
    ],
    PREFIX_IMAGE_DIRS = {
        "TOP": [
"/code/multimodal_retrieval/MCMR-final/images",
        ],
        "BOTTOM": [
"/code/multimodal_retrieval/MCMR-final/images",
        ],
        "SHO": [
"/code/multimodal_retrieval/MCMR-final/images",
        ],
        "Jewelry": [
"/code/multimodal_retrieval/MCMR-final/images",
        ],
        "Furniture": [
"/code/multimodal_retrieval/MCMR-final/images",
        ],
    },

    BATCH_SIZE = 16,                                           
    MAX_QUERIES = None,                                
    MAX_PAIRS_PER_QUERY = None,                                 

    AMP = True,                                      
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu",
    STRICT_SINGLE_TOKEN = True,
    DIAG_FIRST_BATCH = True,            
    DIAG_TOPK = 5,                         

    SYSTEM_PROMPT = (
        "You are a strict e-commerce relevance judge. "
        "Given a user query and a candidate (image + textual attributes), "
        "answer with a single token: True or False. "
        "True means the candidate matches the query faithfully. "
        "False otherwise. No other words, no punctuation."
    ),
)
                                                        

      
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = CONFIG["DEVICE"]
AMP_DTYPE = torch.bfloat16 if (CONFIG["AMP"] and DEVICE == "cuda") else None

                                                        
def json_lines(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                yield json.loads(s)

def append_jsonl(path: str, obj: Dict[str, Any]):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def extract_filename_from_url(url: Optional[str]) -> Optional[str]:
    if not url: return None
    return url.split("/")[-1] or None

def find_local_image_path(candidate: Dict[str, Any], fallback_url: Optional[str], candidate_id: Optional[str]) -> Optional[Path]:
    url = None
    imgs = (candidate or {}).get("images") or []
    for it in imgs:
        url = it.get("url") or it.get("large") or it.get("hi_res")
        if url: break
    if not url:
        url = fallback_url
    fn = extract_filename_from_url(url) if url else None
    if not fn: return None

    search_dirs: List[str] = []
    if candidate_id:
        if candidate_id.startswith("TOP"): search_dirs += CONFIG["PREFIX_IMAGE_DIRS"].get("TOP", [])
        elif candidate_id.startswith("BOTTOM"): search_dirs += CONFIG["PREFIX_IMAGE_DIRS"].get("BOTTOM", [])
        elif candidate_id.startswith("SHO"): search_dirs += CONFIG["PREFIX_IMAGE_DIRS"].get("SHO", [])
        elif candidate_id.startswith("Jewelry"): search_dirs += CONFIG["PREFIX_IMAGE_DIRS"].get("Jewelry", [])
        elif candidate_id.startswith("Furniture"): search_dirs += CONFIG["PREFIX_IMAGE_DIRS"].get("Furniture", [])
    search_dirs += CONFIG["IMAGE_DIRS"]
        
    seen=set(); ordered=[]
    for d in search_dirs:
        if d not in seen:
            seen.add(d); ordered.append(d)
    for d in ordered:
        p = Path(d) / fn
        if p.exists(): return p
    return None

def render_candidate_text(cand: Dict[str, Any]) -> str:
    title = str(cand.get("title") or "").strip()
    desc = cand.get("description") or []
    if isinstance(desc, list): desc = " ".join([str(x) for x in desc if x])
    else: desc = str(desc or "")
    feats = cand.get("features") or []
    if isinstance(feats, list): feats = " ".join([str(x) for x in feats if x])
    else: feats = str(feats or "")
    price = cand.get("price")
    date = cand.get("Date First Available") or cand.get("date") or ""
    s = " | ".join([x for x in [title, desc, feats] if x]).strip()
    if price is not None: s += f" | price: {price}"
    if date: s += f" | date: {date}"
    return s[:2048]

                                                     
def _move_to_device(x, device, amp_dtype):
    """
    Recursively move tensors/containers to `device`. 
    Cast ONLY floating-point tensors to `amp_dtype` if provided.
    Do NOT cast integer id tensors. 
    Safely handle HF BatchEncoding/BatchFeature by moving to device first,
    then selectively casting float tensors.
    """
    import torch
                          
    if isinstance(x, torch.Tensor):
        if amp_dtype is not None and x.is_floating_point():
            return x.to(device=device, dtype=amp_dtype)
        return x.to(device=device)
                                
    if isinstance(x, (BatchEncoding, BatchFeature)):
        x = x.to(device)                                           
        if amp_dtype is not None:
            for k, v in x.items():
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    x[k] = v.to(dtype=amp_dtype)
        return x
                                     
    if isinstance(x, (list, tuple)):
        return type(x)(_move_to_device(y, device, amp_dtype) for y in x)
    if isinstance(x, dict):
        return {k: _move_to_device(v, device, amp_dtype) for k, v in x.items()}
                            
    return x

def load_minicpm(model_dir: str):
                                                                                     
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    proc = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
                                                                                          
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    if not hasattr(cfg, "init_tts") or cfg.init_tts:
        try:
            cfg.init_tts = False
        except Exception:
            pass
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        config=cfg,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=(torch.bfloat16 if AMP_DTYPE is not None else torch.float32),
        device_map=None,
    )
    if DEVICE == "cuda":
        model.to(DEVICE)
    model.eval()
    try:
        print(f"[config] init_tts={getattr(model.config, 'init_tts', None)}")
    except Exception:
        pass
    return tok, proc, model

def build_messages(query_text: str, candidate_text: str, pil_img: Image.Image) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": CONFIG["SYSTEM_PROMPT"]},
        {"role": "user", "content": [
            {"type": "image", "image": pil_img},
            {"type": "text", "text": (
                f"Query: {query_text}\n"
                f"Candidate: {candidate_text}\n"
                "Does the candidate match the query? True or False"
            )},
        ]},
    ]

def render_prompt_with_fallback(processor, messages: List[Dict[str, Any]]) -> str:
    sys_t = ""
    for m in messages:
        if m.get("role") == "system":
            c = m.get("content")
            sys_t = c if isinstance(c, str) else ""
    user_t = ""
    for c in messages[-1].get("content", []):
        if isinstance(c, dict) and c.get("type") == "text":
            user_t = c.get("text") or ""
            break
                                                                                         
    user_t = "(./)\n" + user_t
    merged = (sys_t.strip() + "\n" + user_t.strip()).strip()
    return merged

                                                                  
                                                                  
class LabelScorer:
    def __init__(self, tokenizer: AutoTokenizer, processor: AutoProcessor, model: AutoModelForCausalLM):
        self.tok = tokenizer
        self.proc = processor
        self.model = model
        self._diag_done = False
        self.true_ids = self.tok("True", add_special_tokens=False).input_ids
        self.false_ids = self.tok("False", add_special_tokens=False).input_ids
        if not self.true_ids or not self.false_ids:
            raise RuntimeError("Label tokenization is empty: tokenizer cannot tokenize 'True' / 'False' correctly.")
        if CONFIG["STRICT_SINGLE_TOKEN"] and (len(self.true_ids)!=1 or len(self.false_ids)!=1):
            raise RuntimeError(f"STRICT_SINGLE_TOKEN=True, but labels are not single-token: len(True)={len(self.true_ids)}, len(False)={len(self.false_ids)}")

    def _pack_for_model(self, messages, padding: bool):
        """
        Pack inputs for MiniCPM‑O by converting chat-style `messages` (which include a PIL image)
        into the processor's expected signature: `processor(text=[...], images=[...], ...)`.

        Accepts either a single sample (list[dict]) or a batch (list[list[dict]]).
        Returns a dict {"data": BatchFeature} suitable for MiniCPMO.forward(data=...).
        """
                                                         
        if messages and isinstance(messages[0], dict):
            batch_msgs = [messages]
        else:
            batch_msgs = messages

        texts = []
        images = []

                                          
        for one in batch_msgs:
                                                                                                       
            prompt_text = render_prompt_with_fallback(self.proc, one)

                                                                      
            img = None
            try:
                user_content = one[-1].get("content", [])
                for c in user_content:
                    if isinstance(c, dict) and c.get("type") == "image":
                        img = c.get("image", None)
                        if img is not None:
                            break
            except Exception:
                img = None

            if img is None:
                raise RuntimeError("MiniCPM‑O input packing error: no PIL image found in messages.")

            texts.append(prompt_text)
            images.append(img)

                                                                                                         
        batch: BatchFeature = self.proc(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True if padding else False,
        )
        pos_src = "processor"

                                                                             
        if "position_ids" not in batch:
            try:
                _input_ids = batch.get("input_ids", None)
                _attn_mask = batch.get("attention_mask", None)
                if _input_ids is not None and hasattr(_input_ids, "shape"):
                    bsz, seqlen = _input_ids.shape
                    pos = torch.arange(seqlen, dtype=torch.long).unsqueeze(0).expand(bsz, -1).clone()
                                                                      
                    if _attn_mask is not None and hasattr(_attn_mask, "shape"):
                        lengths = _attn_mask.long().sum(dim=1)       
                        for i in range(bsz):
                            valid = int(lengths[i].item())
                            if valid <= 0:
                                pos[i].fill_(0)
                            elif valid < seqlen:
                                pos[i, valid:] = valid - 1
                    batch["position_ids"] = pos
                    pos_src = "fallback"
            except Exception as _e:
                print(f"[warn] failed to auto-create position_ids: {repr(_e)}")

        try:
            batch["__pos_src"] = pos_src
        except Exception:
            pass

                                                                      
        need = {"input_ids", "position_ids", "pixel_values", "image_bound", "tgt_sizes"}
        got = set(batch.keys())
        missing = need - got
        if missing:
            raise RuntimeError(
                f"Processor output missing keys required by MiniCPM‑O: {sorted(list(missing))}. "
                f"Got keys={sorted(list(got))}."
            )

        return {"data": batch}

    def _score_one(self, messages) -> Tuple[float, float, float]:
        """Single-sample scoring using chat-style messages (with image embedded)."""
        call_kwargs = self._pack_for_model(messages, padding=True)
        try:
            _target_device = self.model.llm.model.embed_tokens.weight.device
        except Exception:
            try:
                _target_device = next(self.model.parameters()).device
            except Exception:
                _target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        call_kwargs = _move_to_device(call_kwargs, _target_device, AMP_DTYPE)

                            
        try:
            self.last_call_data = call_kwargs.get("data", {})
        except Exception:
            self.last_call_data = None

        with torch.no_grad():
            out = self.model(**call_kwargs)
        logits = out.logits             
        step_logits = logits[:, -1, :][0]       

        if CONFIG.get("DIAG_FIRST_BATCH", True) and (not self._diag_done):
            try:
                k = int(CONFIG.get("DIAG_TOPK", 5))
                topk_vals, topk_idx = torch.topk(step_logits, k)
                topk_toks = self.tok.convert_ids_to_tokens(topk_idx.tolist())
                sorted_idx = torch.argsort(step_logits, descending=True)
                tid = self.true_ids[0]; fid = self.false_ids[0]
                r_true = int((sorted_idx == tid).nonzero(as_tuple=False)[0].item()) + 1
                r_false = int((sorted_idx == fid).nonzero(as_tuple=False)[0].item()) + 1
                pos_src = "unknown"
                has_image_bound = None
                try:
                    data = self.last_call_data or {}
                    pos_src = data.get("__pos_src", pos_src)
                    has_image_bound = ("image_bound" in data) and (data["image_bound"] is not None)
                except Exception:
                    pass
                topk_fmt = [f"{tok}:{int(idx)}" for tok, idx in zip(topk_toks, topk_idx.tolist())]
                print(f"[diag] pos_src={pos_src}  has_image_bound={has_image_bound}  bs=1")
                print(f"[diag] first_token_top{k}={topk_fmt}")
                print(f"[diag] rank(True)={r_true}  rank(False)={r_false}")
            except Exception as _e:
                print(f"[diag] failed to print single-sample diagnostics: {repr(_e)}")
            finally:
                self._diag_done = True

        tid = self.true_ids[0]; fid = self.false_ids[0]
        pair = torch.stack([step_logits[fid], step_logits[tid]], dim=0)
        probs = torch.softmax(pair, dim=0)
        p_true = float(probs[1].item())

        step_logprobs = torch.log_softmax(step_logits, dim=-1)
        t_lp = float(step_logprobs[tid].item())
        f_lp = float(step_logprobs[fid].item())
        return p_true, t_lp, f_lp

    @torch.no_grad()
    def score_batch(self, msgs_list: List[List[dict]]) -> Tuple[List[float], List[float], List[float]]:
        """Batch scoring. Each element of `msgs_list` is a chat-style messages list for one sample."""
        try:
            call_kwargs = self._pack_for_model(msgs_list, padding=True)

            try:
                _target_device = self.model.llm.model.embed_tokens.weight.device
            except Exception:
                try:
                    _target_device = next(self.model.parameters()).device
                except Exception:
                    _target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            call_kwargs = _move_to_device(call_kwargs, _target_device, AMP_DTYPE)

            data = call_kwargs.get("data", {})
            bs = len(data["input_ids"]) if "input_ids" in data else len(msgs_list)
            pix = data.get("pixel_values", None)

            ok_shape = False
            if isinstance(pix, (list, tuple)):
                ok_shape = (len(pix) == bs)
            elif isinstance(pix, torch.Tensor):
                ok_shape = (pix.shape[0] == bs)
            if not ok_shape:
                got_len = (len(pix) if isinstance(pix, (list, tuple)) else (pix.shape[0] if isinstance(pix, torch.Tensor) else 'None'))
                print(f"[guard] fallback to single-sample (pixel_values shape mismatch: {type(pix)}; bs={bs}, got_len={got_len})")
                p_true_list, lt_list, lf_list = [], [], []
                for msgs in msgs_list:
                    p_true, lt, lf = self._score_one(msgs)
                    p_true_list.append(p_true); lt_list.append(lt); lf_list.append(lf)
                return p_true_list, lt_list, lf_list

            out = self.model(**call_kwargs)
            logits = out.logits             
            first_logits = logits[:, -1, :]

                                  
            if CONFIG.get("DIAG_FIRST_BATCH", True) and (not self._diag_done):
                try:
                    data = call_kwargs.get("data", {})
                    pos_src = data.get("__pos_src", "unknown")
                    has_image_bound = ("image_bound" in data) and (data["image_bound"] is not None)
                    bs = first_logits.shape[0]
                    v = first_logits[0]
                    k = int(CONFIG.get("DIAG_TOPK", 5))
                    topk_vals, topk_idx = torch.topk(v, k)
                    topk_toks = self.tok.convert_ids_to_tokens(topk_idx.tolist())
                    sorted_idx = torch.argsort(v, descending=True)
                    tid = self.true_ids[0]; fid = self.false_ids[0]
                    r_true = int((sorted_idx == tid).nonzero(as_tuple=False)[0].item()) + 1
                    r_false = int((sorted_idx == fid).nonzero(as_tuple=False)[0].item()) + 1
                    topk_fmt = [f"{tok}:{int(idx)}" for tok, idx in zip(topk_toks, topk_idx.tolist())]
                    print(f"[diag] pos_src={pos_src}  has_image_bound={has_image_bound}  bs={bs}")
                    print(f"[diag] first_token_top{k}={topk_fmt}")
                    print(f"[diag] rank(True)={r_true}  rank(False)={r_false}")
                except Exception as _e:
                    print(f"[diag] failed to print first-batch diagnostics: {repr(_e)}")
                finally:
                    self._diag_done = True

            tid = self.true_ids[0]; fid = self.false_ids[0]
            logprobs = torch.log_softmax(first_logits, dim=-1)
            t_lp = logprobs[:, tid]
            f_lp = logprobs[:, fid]
            pair = torch.stack([f_lp, t_lp], dim=1)
            p_true = torch.softmax(pair, dim=1)[:, 1]

            return p_true.tolist(), t_lp.tolist(), f_lp.tolist()

        except (RuntimeError, IndexError) as e:
            msg = str(e)
            if ("Sizes of tensors must match" in msg) or ("hstack" in msg) or ("list index out of range" in msg):
                print(f"[guard] fallback to single-sample scoring due to: {repr(e)}")
                p_true_list, lt_list, lf_list = [], [], []
                for msgs in msgs_list:
                    p_true, lt, lf = self._score_one(msgs)
                    p_true_list.append(p_true); lt_list.append(lt); lf_list.append(lf)
                return p_true_list, lt_list, lf_list
            raise

                                                   
def main():
    cfg = CONFIG
    tokenizer, processor, model = load_minicpm(cfg["MODEL_DIR"])
    scorer = LabelScorer(tokenizer, processor, model)

          
    seen = set()
    out_p = Path(cfg["OUT_POINTWISE_JSONL"])
    if out_p.exists():
        for obj in json_lines(str(out_p)):
            seen.add(f"{obj['qid']}|{obj['candidate_id']}")

    total_pairs=0; done_pairs=0
    stats = dict(skipped_seen=0, image_missing=0, image_open_fail=0, image_too_small=0, empty_text=0)
    debug_prints = 0

         
    buf_msgs: List[List[Dict[str, Any]]] = []
    buf_meta: List[Dict[str, Any]] = []

    def flush_batch():
        nonlocal done_pairs, buf_msgs, buf_meta
        if not buf_msgs: return
        p_list, t_list, f_list = scorer.score_batch(buf_msgs)
        for (meta, p, lt, lf) in zip(buf_meta, p_list, t_list, f_list):
            out = {
                "qid": meta["qid"], "candidate_id": meta["cid"],
                "rank": meta["rank"], "is_pos": meta["is_pos"],
                "p_true": float(p),
                "label_true_logprob": float(lt),
                "label_false_logprob": float(lf),
                "retrieval_score": meta["ret_score"],
                "image_path": meta["img_path"],
            }
            append_jsonl(cfg["OUT_POINTWISE_JSONL"], out)
            done_pairs += 1
            if done_pairs % 32 == 0:
                print(f"[progress] written={done_pairs}")
        buf_msgs=[]; buf_meta=[]

    q_c = 0
    for rec in json_lines(cfg["INPUT_TOPK_JSONL"]):
        if cfg["MAX_QUERIES"] is not None and q_c >= cfg["MAX_QUERIES"]:
            break
        q_c += 1

        qid = rec.get("qid")
        query = rec.get("query") or ""
        topk = rec.get("topk") or []
        if cfg["MAX_PAIRS_PER_QUERY"] is not None:
            topk = topk[: cfg["MAX_PAIRS_PER_QUERY"]]

        for item in topk:
            total_pairs += 1
            cid = item.get("candidate_id")
            key = f"{qid}|{cid}"
            if key in seen:
                stats["skipped_seen"] += 1
                continue

            cand_raw = item.get("candidate") or {}
            img_path = find_local_image_path(cand_raw, item.get("image_url"), cid)
            if img_path is None:
                stats["image_missing"] += 1
                if debug_prints < 3:
                    debug_prints += 1
                    url = None
                    imgs = (cand_raw or {}).get("images") or []
                    for it in imgs:
                        url = it.get("url") or it.get("large") or it.get("hi_res")
                        if url: break
                    if url is None: url = item.get("image_url")
                    fn = extract_filename_from_url(url) if url else None
                                    
                    debug_dirs=[]
                    if cid:
                        if cid.startswith("TOP"): debug_dirs += cfg["PREFIX_IMAGE_DIRS"].get("TOP", [])
                        elif cid.startswith("BOTTOM"): debug_dirs += cfg["PREFIX_IMAGE_DIRS"].get("BOTTOM", [])
                        elif cid.startswith("SHO"): debug_dirs += cfg["PREFIX_IMAGE_DIRS"].get("SHO", [])
                        elif cid.startswith("Jewelry"): debug_dirs += cfg["PREFIX_IMAGE_DIRS"].get("Jewelry", [])
                        elif cid.startswith("Furniture"): debug_dirs += cfg["PREFIX_IMAGE_DIRS"].get("Furniture", [])
                    debug_dirs += cfg["IMAGE_DIRS"]
                    _s=set(); ordered=[]
                    for d in debug_dirs:
                        if d not in _s: _s.add(d); ordered.append(d)
                    print(f"[debug] missing image for {cid}: url={url} filename={fn} searched_dirs={ordered}")
                continue

            try:
                pil = Image.open(img_path).convert("RGB")
            except Exception as e:
                stats["image_open_fail"] += 1
                if debug_prints < 3:
                    debug_prints += 1
                    print(f"[debug] PIL open fail for {cid}: path={img_path} err={repr(e)}")
                continue

            w, h = pil.size
            if w < 28 or h < 28:                           
                stats["image_too_small"] += 1
                if debug_prints < 3:
                    debug_prints += 1
                    print(f"[debug] too small image for {cid}: path={img_path} size=({w},{h}) < 28")
                continue

            cand_text = render_candidate_text(cand_raw)
            if not cand_text.strip():
                stats["empty_text"] += 1
                continue

                                                                                    
            msgs = build_messages(query, cand_text, pil)
            buf_msgs.append(msgs)
            buf_meta.append(dict(
                qid=qid, cid=cid,
                rank=int(item.get("rank") or 0),
                is_pos=int(item.get("is_pos") or 0),
                ret_score=float(item.get("score") or 0.0),
                img_path=str(img_path),
            ))

            if len(buf_msgs) >= cfg["BATCH_SIZE"]:
                flush_batch()

        
    flush_batch()

    print(f"[done] pairs computed: {done_pairs}/{total_pairs}.")
    print(f"[stats] skipped_seen={stats['skipped_seen']}  image_missing={stats['image_missing']}  image_open_fail={stats['image_open_fail']}  image_too_small={stats['image_too_small']}  empty_text={stats['empty_text']}")
    print(f"[save] -> {cfg['OUT_POINTWISE_JSONL']}")

if __name__ == "__main__":
    main()
