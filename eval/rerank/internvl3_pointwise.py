import os, json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor, AutoTokenizer,
    AutoModelForCausalLM, AutoModelForVision2Seq,
    AutoImageProcessor, AutoConfig
)
import numpy as np

                                                        
CONFIG = dict(
    MODEL_DIR = "/code/.cache/huggingface/InternVL3-8B-Instruct",
    MODEL_TYPE = "causallm",                                                     
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

    MAX_QUERIES = None,                                  
    MAX_PAIRS_PER_QUERY = None,                                   
    AMP = True,                                        
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu",
    STRICT_SINGLE_TOKEN = True,                                 
    IMG_MIN_SIDE = 28,                                              
                                          
    BATCH_SIZE = 16,
                     
    PROGRESS_EVERY = 32,
                  
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
        if candidate_id.startswith("TOP"):
            prefix = "TOP"
        elif candidate_id.startswith("BOTTOM"):
            prefix = "BOTTOM"
        elif candidate_id.startswith("SHO"):
            prefix = "SHO"
        elif candidate_id.startswith("Jewelry"):
            prefix = "Jewelry"
        elif candidate_id.startswith("Furniture"):
            prefix = "Furniture"
        if prefix and "PREFIX_IMAGE_DIRS" in CONFIG:
            search_dirs.extend(CONFIG["PREFIX_IMAGE_DIRS"].get(prefix, []))
          
    search_dirs.extend(CONFIG["IMAGE_DIRS"])
            
    seen = set(); ordered = []
    for d in search_dirs:
        if d not in seen:
            seen.add(d); ordered.append(d)
    for d in ordered:
        p = Path(d) / fn
        if p.exists():
            return p
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
    parts = [title, desc, feats]
    s = " | ".join([x for x in parts if x]).strip()
    if price is not None: s += f" | price: {price}"
    if date: s += f" | date: {date}"
    return s[:2048]

                                                  
def load_internvl(model_dir: str, model_type: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True, use_fast=True)

                                                                        
    image_processor = None
    try:
        image_processor = AutoImageProcessor.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    except Exception:
        try:
            cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
            vision_name = None
            if hasattr(cfg, "vision_config") and isinstance(cfg.vision_config, dict):
                vision_name = cfg.vision_config.get("_name_or_path")
            elif hasattr(cfg, "vision_config") and getattr(cfg.vision_config, "_name_or_path", None):
                vision_name = cfg.vision_config._name_or_path
            if vision_name:
                image_processor = AutoImageProcessor.from_pretrained(vision_name, trust_remote_code=True, local_files_only=True)
        except Exception:
            image_processor = None
    if model_type == "vision2seq":
        model = AutoModelForVision2Seq.from_pretrained(
            model_dir, trust_remote_code=True, local_files_only=True,
            torch_dtype=(torch.bfloat16 if AMP_DTYPE is not None else torch.float32),
            device_map="auto" if DEVICE == "cuda" else None
        )
    elif model_type == "causallm":
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True, local_files_only=True,
            torch_dtype=(torch.bfloat16 if AMP_DTYPE is not None else torch.float32),
            device_map="auto" if DEVICE == "cuda" else None
        )
    else:
        raise ValueError(f"Unsupported MODEL_TYPE={model_type}. Use 'causallm' or 'vision2seq'.")
    model.eval()
    return tokenizer, image_processor, model

def build_messages(query_text: str, candidate_text: str, pil_img: Image.Image) -> List[Dict[str, Any]]:
    """
    InternVL3 的 chat_template（template="internvl2_5"）期望 message["content"] 为字符串。
    因此这里不再传入 [ {type:"image"}, {type:"text"} ] 列表；
    图片通过 processor(..., images=[pil_img]) 传入，文本里只保留占位符 <image>。
    """
    user_text = (
        f"<image>\n"
        f"Query: {query_text}\n"
        f"Candidate: {candidate_text}\n"
        f"Does the candidate match the query? True or False"
    )
    return [
        {"role": "system", "content": CONFIG["SYSTEM_PROMPT"]},
        {"role": "user", "content": user_text},
    ]


                                                                                
def to_prompt_text(tokenizer: AutoTokenizer, messages: List[Dict[str, Any]], model) -> str:
    """Render chat template to a string and replace `<image>` with `<img> <IMG_CONTEXT>*K </img>`.
    K = model.num_image_token * num_patches; here we assume num_patches=1 (448x448 path).
    Also sets model.img_context_token_id for downstream use.
    """
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    num_patches = 1
    num_img_tok = int(getattr(model, 'num_image_token', 256)) * num_patches
    image_tokens = IMG_START_TOKEN + (IMG_CONTEXT_TOKEN * num_img_tok) + IMG_END_TOKEN
    if '<image>' in prompt:
        prompt = prompt.replace('<image>', image_tokens, 1)
                                                        
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    return prompt

                                                                    
class LabelScorer:
    def __init__(self, tokenizer: AutoTokenizer, image_processor: Optional[AutoImageProcessor], model):
        self.tok = tokenizer
        self.img_proc = image_processor
        self.model = model
        self.true_ids = self.tok("True", add_special_tokens=False).input_ids
        self.false_ids = self.tok("False", add_special_tokens=False).input_ids
        if not self.true_ids or not self.false_ids:
            raise RuntimeError("标签分词为空：请检查 tokenizer 是否能正确分词 'True' / 'False'。")
        if bool(CONFIG.get("STRICT_SINGLE_TOKEN", False)):
            if len(self.true_ids) != 1 or len(self.false_ids) != 1:
                raise RuntimeError(f"STRICT_SINGLE_TOKEN=True，但分词结果不为单 token：len(True)={len(self.true_ids)}, len(False)={len(self.false_ids)}。请更换模型/词表或关闭该开关。")
                                                                         
        if self.tok.pad_token_id is None and getattr(self.tok, "eos_token_id", None) is not None:
            self.tok.pad_token = self.tok.eos_token

                                                                   
        self._fallback_img_size = 448
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def _prepare_images(self, pil_list: List[Image.Image]) -> Dict[str, torch.Tensor]:
        if self.img_proc is not None:
            return self.img_proc(images=pil_list, return_tensors="pt")
                                    
        batch = []
        for im in pil_list:
            im = im.convert("RGB")
            im = im.resize((self._fallback_img_size, self._fallback_img_size), resample=Image.BICUBIC)
            arr = np.array(im).astype("float32") / 255.0         
            arr = np.transpose(arr, (2, 0, 1))         
            batch.append(arr)
        x = torch.from_numpy(np.stack(batch, axis=0))           
        x = (x - self._mean) / self._std
        return {"pixel_values": x}

    @torch.no_grad()
    def score_pair(self, query_text: str, cand_text: str, pil_img: Image.Image) -> Tuple[float, float, float]:
        messages = build_messages(query_text, cand_text, pil_img)
        prompt_text = to_prompt_text(self.tok, messages, self.model)

                                             
        enc = self.tok([prompt_text], return_tensors="pt", padding=False)
        input_ids_prompt = enc["input_ids"].to(self.model.device)
        attn_prompt = enc.get("attention_mask", None)
        if attn_prompt is not None:
            attn_prompt = attn_prompt.to(self.model.device)

        img_inputs = self._prepare_images([pil_img])
               
        for k in list(img_inputs.keys()):
            img_inputs[k] = img_inputs[k].to(self.model.device)
        mm_kwargs = img_inputs

        prompt_len = int(input_ids_prompt.shape[1])

        img_context_id = getattr(self.model, 'img_context_token_id', self.tok.convert_tokens_to_ids('<IMG_CONTEXT>'))
        n_img_tok = int((input_ids_prompt == img_context_id).sum().item())
        if n_img_tok <= 0:
            raise RuntimeError("No <IMG_CONTEXT> tokens found in prompt; cannot build image_flags.")
        image_flags = torch.ones((1, n_img_tok), dtype=torch.long, device=self.model.device)                    

        def branch_logprob(label_ids: List[int]) -> float:
            label_t = torch.tensor([label_ids], device=self.model.device, dtype=input_ids_prompt.dtype)
            input_ids = torch.cat([input_ids_prompt, label_t], dim=1)
            attn = torch.cat([attn_prompt, torch.ones_like(label_t)], dim=1) if attn_prompt is not None else None
            with torch.cuda.amp.autocast(enabled=(AMP_DTYPE is not None), dtype=AMP_DTYPE):
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attn,
                    image_flags=image_flags,           
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

        true_lp = branch_logprob(self.true_ids)
        false_lp = branch_logprob(self.false_ids)
        probs_2 = torch.softmax(torch.tensor([true_lp, false_lp], dtype=torch.float32), dim=-1)
        return float(probs_2[0].item()), float(true_lp), float(false_lp)

    @torch.no_grad()
    def score_batch(self, prompt_texts: List[str], pil_imgs: List[Image.Image]) -> Tuple[List[float], List[float], List[float]]:
        """
        对一批样本进行 True/False 二分类打分（与 score_pair 等价的批量版）：
          返回三个等长列表：p_true_list, true_lp_list, false_lp_list
        """
        assert len(prompt_texts) == len(pil_imgs) and len(prompt_texts) > 0
        B = len(prompt_texts)

                                                                    
        enc = self.tok(prompt_texts, return_tensors="pt", padding=True, truncation=False)
        input_ids_prompt = enc["input_ids"].to(self.model.device)
        attn_prompt = enc.get("attention_mask", None)
        if attn_prompt is not None:
            attn_prompt = attn_prompt.to(self.model.device)

        img_inputs = self._prepare_images(pil_imgs)
        for k in list(img_inputs.keys()):
            img_inputs[k] = img_inputs[k].to(self.model.device)
        mm_kwargs = img_inputs

                                                                           
        img_context_id = getattr(self.model, 'img_context_token_id', self.tok.convert_tokens_to_ids('<IMG_CONTEXT>'))
        counts = (input_ids_prompt == img_context_id).sum(dim=1)       
                                                      
        n_img_tok = int(counts.max().item())
        if not torch.all(counts == counts[0]):
            raise RuntimeError(f"Batched prompts have different numbers of <IMG_CONTEXT> tokens: {counts.tolist()}. Ensure fixed num_patches or disable batching.")
        if n_img_tok <= 0:
            raise RuntimeError("No <IMG_CONTEXT> tokens found in prompts; cannot build image_flags.")
        image_flags = torch.ones((B, n_img_tok), dtype=torch.long, device=self.model.device)                    

                                    
        if attn_prompt is not None:
            prompt_lens = attn_prompt.sum(dim=1).tolist()                  
        else:
                                                                             
            pad_id = getattr(self.tok, "pad_token_id", None)
            if pad_id is None:
                raise RuntimeError("tokenizer 未提供 attention_mask，且 pad_token_id 不可用，无法确定每条样本的 prompt 长度。")
            prompt_lens = (input_ids_prompt != pad_id).sum(dim=1).tolist()

        def branch_logprob_batched(label_ids: List[int]) -> List[float]:
            T = len(label_ids)
            label = torch.tensor(label_ids, dtype=input_ids_prompt.dtype, device=self.model.device).unsqueeze(0).repeat(B, 1)          
            input_ids = torch.cat([input_ids_prompt, label], dim=1)                 
            attn = torch.cat([attn_prompt, torch.ones((B, T), dtype=attn_prompt.dtype, device=self.model.device)], dim=1) if attn_prompt is not None else None

            with torch.cuda.amp.autocast(enabled=(AMP_DTYPE is not None), dtype=AMP_DTYPE):
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attn,
                    image_flags=image_flags,           
                    **mm_kwargs,
                    use_cache=False,
                    output_hidden_states=False,
                    output_attentions=False
                )
                logits = out.logits                   
                logprobs = F.log_softmax(logits, dim=-1)

            lp_list: List[float] = []
            for b in range(B):
                lp = 0.0
                base = int(prompt_lens[b]) - 1            
                for j, tid in enumerate(label_ids):
                    pos = base + j
                    lp += float(logprobs[b, pos, tid].item())
                lp_list.append(lp)
            return lp_list

        true_lp_list = branch_logprob_batched(self.true_ids)
        false_lp_list = branch_logprob_batched(self.false_ids)

                               
        p_true_list: List[float] = []
        for tlp, flp in zip(true_lp_list, false_lp_list):
            probs_2 = torch.softmax(torch.tensor([tlp, flp], dtype=torch.float32), dim=-1)
            p_true_list.append(float(probs_2[0].item()))
        return p_true_list, true_lp_list, false_lp_list

                                                   
def main():
    cfg = CONFIG
    tokenizer, image_processor, model = load_internvl(cfg["MODEL_DIR"], cfg["MODEL_TYPE"])
    scorer = LabelScorer(tokenizer, image_processor, model)

               
    print(f"[labels] len(True)={len(scorer.true_ids)} len(False)={len(scorer.false_ids)}  STRICT_SINGLE_TOKEN={cfg['STRICT_SINGLE_TOKEN']}")

                          
    seen = set()
    out_p = Path(cfg["OUT_POINTWISE_JSONL"])
    if out_p.exists():
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

            
    buf_prompts: List[str] = []
    buf_imgs: List[Image.Image] = []
    buf_meta: List[Dict[str, Any]] = []

    def flush_batch():
        nonlocal buf_prompts, buf_imgs, buf_meta, done_pairs
        if not buf_prompts:
            return
        p_list, t_list, f_list = scorer.score_batch(buf_prompts, buf_imgs)
        for meta, p_true, lp_t, lp_f in zip(buf_meta, p_list, t_list, f_list):
            out_obj = {
                "qid": meta["qid"],
                "candidate_id": meta["cid"],
                "rank": int(meta["rank"]),
                "is_pos": int(meta["is_pos"]),
                "p_true": float(p_true),
                "label_true_logprob": float(lp_t),
                "label_false_logprob": float(lp_f),
                "retrieval_score": float(meta["retrieval_score"]),
                "image_path": meta["image_path"],
            }
            append_jsonl(cfg["OUT_POINTWISE_JSONL"], out_obj)
            seen.add(f"{meta['qid']}|{meta['cid']}")
            done_pairs += 1
            if done_pairs % cfg.get("PROGRESS_EVERY", 32) == 0:
                print(f"[progress] written={done_pairs}")
              
        buf_prompts.clear(); buf_imgs.clear(); buf_meta.clear()

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
                    debug_prints += 1
                    url = None
                    imgs = (cand_raw or {}).get("images") or []
                    for it in imgs:
                        url = it.get("url") or it.get("large") or it.get("hi_res")
                        if url: break
                    if url is None: url = item.get("image_url")
                    fn = extract_filename_from_url(url) if url else None
                                           
                    debug_dirs = []
                    if cid and "PREFIX_IMAGE_DIRS" in cfg:
                        if cid.startswith("TOP"):
                            debug_dirs += cfg["PREFIX_IMAGE_DIRS"].get("TOP", [])
                        elif cid.startswith("BOTTOM"):
                            debug_dirs += cfg["PREFIX_IMAGE_DIRS"].get("BOTTOM", [])
                        elif cid.startswith("SHO"):
                            debug_dirs += cfg["PREFIX_IMAGE_DIRS"].get("SHO", [])
                        elif cid.startswith("Jewelry"):
                            debug_dirs += cfg["PREFIX_IMAGE_DIRS"].get("Jewelry", [])
                        elif cid.startswith("Furniture"):
                            debug_dirs += cfg["PREFIX_IMAGE_DIRS"].get("Furniture", [])
                    debug_dirs += cfg["IMAGE_DIRS"]
                    _seen = set(); ordered = []
                    for d in debug_dirs:
                        if d not in _seen:
                            _seen.add(d); ordered.append(d)
                    print(f"[debug] missing image for {cid}: url={url} filename={fn} searched_dirs={ordered}")
                continue

            try:
                pil = Image.open(img_path).convert("RGB")
            except Exception as e:
                stats["skipped_image_open_fail"] += 1
                if debug_prints < 3:
                    debug_prints += 1
                    print(f"[debug] PIL open fail for {cid}: path={img_path} err={repr(e)}")
                continue

            w, h = pil.size
            if w < cfg["IMG_MIN_SIDE"] or h < cfg["IMG_MIN_SIDE"]:
                stats["skipped_image_too_small"] += 1
                if debug_prints < 3:
                    debug_prints += 1
                    print(f"[debug] too small image for {cid}: path={img_path} size=({w},{h}) < {cfg['IMG_MIN_SIDE']}")
                continue

            cand_text = render_candidate_text(cand_raw)
            if not cand_text.strip():
                stats["skipped_empty_text"] += 1
                continue

                        
            messages = build_messages(query, cand_text, pil)
            prompt_text = to_prompt_text(tokenizer, messages, model)
            buf_prompts.append(prompt_text)
            buf_imgs.append(pil)
            buf_meta.append({
                "qid": qid,
                "cid": cid,
                "rank": int(item.get("rank") or 0),
                "is_pos": int(item.get("is_pos") or 0),
                "retrieval_score": float(item.get("score") or 0.0),
                "image_path": str(img_path),
            })
                               
            if len(buf_prompts) >= cfg.get("BATCH_SIZE", 8):
                flush_batch()

    flush_batch()
    print(f"[done] pairs computed: {done_pairs}/{total_pairs}.")
    print(f"[stats] skipped_seen={stats['skipped_seen']} image_missing={stats['skipped_image_missing']} image_open_fail={stats['skipped_image_open_fail']} image_too_small={stats['skipped_image_too_small']} empty_text={stats['skipped_empty_text']}")
    print(f"[save] -> {cfg['OUT_POINTWISE_JSONL']}")

if __name__ == "__main__":
    main()