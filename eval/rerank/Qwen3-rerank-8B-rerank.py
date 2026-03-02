import os, json, math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer

                                                               
CONFIG = dict(
             
    MODEL_DIR = "/code/.cache/huggingface/Qwen3-VL-Reranker-8B",
    
                             
    INPUT_TOPK_JSONL = "llave_eval_topk50.jsonl",
    OUT_POINTWISE_JSONL = "",

                                
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

    BATCH_SIZE = 4,                   
    MAX_QUERIES = None,               
    MAX_PAIRS_PER_QUERY = None,                
    AMP = True,                                
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu",
    STRICT_SINGLE_TOKEN = True, 
    
                      
                                                                          
    SYSTEM_PROMPT = "Retrieval relevant image or text with user's query",
    
                                   
    POSITIVE_LABEL = "Yes",
    NEGATIVE_LABEL = "No",
)
                                                                  

      
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = CONFIG["DEVICE"]
AMP_DTYPE = torch.bfloat16 if CONFIG["AMP"] and DEVICE == "cuda" else None

                                                       
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

def find_local_image_path(any_candidate_obj: Dict[str, Any], fallback_url: Optional[str], candidate_id: Optional[str] = None) -> Optional[Path]:
    """复用原有逻辑：在 IMAGE_DIRS 中查找图片"""
    url = None
    imgs = (any_candidate_obj or {}).get("images") or []
    for it in imgs:
        url = it.get("url") or it.get("large") or it.get("hi_res")
        if url: break
    if not url:
        url = fallback_url
    fn = extract_filename_from_url(url) if url else None
    if not fn:
        return None
    
    search_dirs: List[str] = []
    if candidate_id:
        prefix = None
        if candidate_id.startswith("TOP"): prefix = "TOP"
        elif candidate_id.startswith("BOTTOM"): prefix = "BOTTOM"
        elif candidate_id.startswith("SHO"): prefix = "SHO"
        elif candidate_id.startswith("Jewelry"): prefix = "Jewelry"
        elif candidate_id.startswith("Furniture"): prefix = "Furniture"
        
        if prefix and "PREFIX_IMAGE_DIRS" in CONFIG:
            search_dirs.extend(CONFIG["PREFIX_IMAGE_DIRS"].get(prefix, []))
    
    search_dirs.extend(CONFIG["IMAGE_DIRS"])
    
    seen_dirs = set()
    ordered_dirs = []
    for d in search_dirs:
        if d not in seen_dirs:
            seen_dirs.add(d)
            ordered_dirs.append(d)
            
    for d in ordered_dirs:
        p = Path(d) / fn
        if p.exists():
            return p
    return None

def render_candidate_text(cand: Dict[str, Any]) -> str:
    """将 candidate 字典转换为文本描述"""
    title = str(cand.get("title") or "").strip()
    desc = cand.get("description") or []
    if isinstance(desc, list): desc = " ".join([str(x) for x in desc if x])
    else: desc = str(desc or "")
    feats = cand.get("features") or []
    if isinstance(feats, list): feats = " ".join([str(x) for x in feats if x])
    else: feats = str(feats or "")
    price = cand.get("price")
    date = cand.get("Date First Available") or cand.get("date") or ""
    
    parts = [title, desc, feats]
    s = " | ".join([x for x in parts if x]).strip()
    if price is not None: s += f" | price: {price}"
    if date: s += f" | date: {date}"
    return s[:2048]

                                                     
def load_qwen(model_dir: str):
    print(f"Loading Qwen3-VL-Reranker from: {model_dir}")
                                      
    try:
        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True, use_fast=False)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True, use_fast=True)
                                                                    
                                                      
                                                      
        model = AutoModelForVision2Seq.from_pretrained(
            model_dir, 
            trust_remote_code=True, 
            local_files_only=True,
            torch_dtype=(torch.bfloat16 if AMP_DTYPE is not None else torch.float32),
            device_map="auto" if DEVICE == "cuda" else None
        )
        model.eval()
        return tokenizer, processor, model
    except Exception as e:
        raise RuntimeError(f"Failed to load model. Error: {e}")

def build_messages(query_text: str, candidate_text: str, pil_img: Image.Image) -> List[Dict[str, Any]]:
                                     
                                                                  
                                                  
    return [
        {"role": "system", "content": CONFIG["SYSTEM_PROMPT"]},
        {"role": "user", "content": [
            {"type": "image", "image": pil_img},
            {"type": "text", "text": f"<image>\nQuery: {query_text}\nDocument: {candidate_text}"}
        ]},
    ]

                                                             
class LabelScorer:
    def __init__(self, tokenizer: AutoTokenizer, processor: AutoProcessor, model: AutoModelForVision2Seq):
        self.tok = tokenizer
        self.proc = processor
        self.model = model
        
                                
        self.pos_label = CONFIG["POSITIVE_LABEL"]
        self.neg_label = CONFIG["NEGATIVE_LABEL"]
        
        self.pos_ids = self.tok(self.pos_label, add_special_tokens=False).input_ids
        self.neg_ids = self.tok(self.neg_label, add_special_tokens=False).input_ids
        
        print(f"Label Tokens: '{self.pos_label}'->{self.pos_ids}, '{self.neg_label}'->{self.neg_ids}")

        if not self.pos_ids or not self.neg_ids:
            raise RuntimeError(f"标签分词为空。请检查 '{self.pos_label}'/'{self.neg_label}' 是否在词表中。")
            
        if bool(CONFIG.get("STRICT_SINGLE_TOKEN", False)):
            if len(self.pos_ids) != 1 or len(self.neg_ids) != 1:
                raise RuntimeError(f"STRICT_SINGLE_TOKEN=True，但标签不为单 token。")

    @torch.no_grad()
    def score_pair(self, query_text: str, cand_text: str, pil_img: Image.Image) -> Tuple[float, float, float]:
        """
        返回 (p_pos, pos_logprob, neg_logprob)
        p_pos = p(Yes)
        """
        messages = build_messages(query_text, cand_text, pil_img)

                      
        prompt_text = self.proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                    
        batch = self.proc(text=[prompt_text], images=[pil_img], return_tensors="pt")
        input_ids_prompt = batch["input_ids"].to(self.model.device)
        attn_prompt = batch["attention_mask"].to(self.model.device) if "attention_mask" in batch else None
        
        mm_kwargs = {}
        for k, v in batch.items():
            if k not in ("input_ids", "attention_mask"):
                mm_kwargs[k] = v.to(self.model.device)
        prompt_len = int(input_ids_prompt.shape[1])

                           
        def branch_logprob(label_ids: List[int]) -> float:
            label_t = torch.tensor([label_ids], device=self.model.device, dtype=input_ids_prompt.dtype)
            input_ids = torch.cat([input_ids_prompt, label_t], dim=1)
            if attn_prompt is not None:
                attn = torch.cat([attn_prompt, torch.ones_like(label_t)], dim=1)
            else:
                attn = None

            with torch.cuda.amp.autocast(enabled=(AMP_DTYPE is not None), dtype=AMP_DTYPE):
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attn,
                    **mm_kwargs,
                    use_cache=False,
                    output_hidden_states=False,
                    output_attentions=False
                )
                logits = out.logits
                logprobs = F.log_softmax(logits, dim=-1)

            lp = 0.0
            for i, tid in enumerate(label_ids):
                pos = prompt_len - 1 + i
                lp += float(logprobs[0, pos, tid].item())
            return lp

        pos_lp = branch_logprob(self.pos_ids)
        neg_lp = branch_logprob(self.neg_ids)

                    
                                                  
        logits_2 = torch.tensor([pos_lp, neg_lp], dtype=torch.float32)
        probs_2 = torch.softmax(logits_2, dim=-1)
        p_pos = float(probs_2[0].item())
        
        return p_pos, float(pos_lp), float(neg_lp)

                                                   
def main():
    cfg = CONFIG
    print(f"Start processing. Output will be saved to: {cfg['OUT_POINTWISE_JSONL']}")
    
    tokenizer, processor, model = load_qwen(cfg["MODEL_DIR"])
    scorer = LabelScorer(tokenizer, processor, model)

    seen = set()
    out_p = Path(cfg["OUT_POINTWISE_JSONL"])
    if out_p.exists():
        print(f"Resuming from {out_p}...")
        for obj in json_lines(str(out_p)):
            seen.add(f"{obj['qid']}|{obj['candidate_id']}")

    total_pairs = 0
    done_pairs = 0
    stats = {
        "skipped_seen": 0,
        "skipped_image_missing": 0,
        "skipped_image_open_fail": 0,
        "skipped_image_too_small": 0,
        "skipped_empty_text": 0,
    }
    
             
    debug_prints = 0

    q_count = 0
    for rec in json_lines(cfg["INPUT_TOPK_JSONL"]):
        qid = rec.get("qid")
        query = rec.get("query") or ""
        topk = rec.get("topk") or []
        
        if cfg["MAX_QUERIES"] is not None and q_count >= cfg["MAX_QUERIES"]:
            break
        q_count += 1

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
            img_path = find_local_image_path(cand_raw, item.get("image_url"), candidate_id=cid)
            
            if img_path is None:
                stats["skipped_image_missing"] += 1
                if debug_prints < 3:
                    print(f"[Warn] Missing image for {cid}. Check paths.")
                    debug_prints += 1
                continue

            try:
                pil = Image.open(img_path).convert("RGB")
            except Exception as e:
                stats["skipped_image_open_fail"] += 1
                continue
                
            w, h = pil.size
            if w < 28 or h < 28:
                stats["skipped_image_too_small"] += 1
                continue

            cand_text = render_candidate_text(cand_raw)
            if not cand_text.strip():
                stats["skipped_empty_text"] += 1
                continue

                
                                 
            p_pos, lp_pos, lp_neg = scorer.score_pair(query, cand_text, pil)

            out_obj = {
                "qid": qid,
                "candidate_id": cid,
                "rank": int(item.get("rank") or 0),
                "is_pos": int(item.get("is_pos") or 0),
                "p_true": p_pos,                                        
                "label_true_logprob": lp_pos,                    
                "label_false_logprob": lp_neg,                  
                "retrieval_score": float(item.get("score") or 0.0),
                "image_path": str(img_path),
            }
            append_jsonl(cfg["OUT_POINTWISE_JSONL"], out_obj)
            seen.add(key)
            done_pairs += 1
            
            if done_pairs % 32 == 0:
                print(f"[progress] written={done_pairs}")

    print(f"[done] pairs computed: {done_pairs}/{total_pairs}.")
    print(f"[stats] {json.dumps(stats, indent=2)}")
    print(f"[save] -> {cfg['OUT_POINTWISE_JSONL']}")

if __name__ == "__main__":
    main()