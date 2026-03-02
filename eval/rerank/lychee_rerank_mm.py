import os, json, math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info

                     
try:
    from transformers import AutoModelForImageTextToText as AutoModelForVision
except ImportError:
    from transformers import AutoModelForVision2Seq as AutoModelForVision

                                                               
CONFIG = dict(
          
    MODEL_DIR = "/code/.cache/huggingface/lychee-rerank-mm",
    
              
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

    BATCH_SIZE = 1,
    MAX_QUERIES = None,
    MAX_PAIRS_PER_QUERY = None,
    AMP = True,
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu",
    
                 
                                  
    INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query",
    
                         
    SYSTEM_PROMPT = "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".",
    
    POSITIVE_LABEL = "yes",
    NEGATIVE_LABEL = "no",
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
    """在 IMAGE_DIRS 中查找图片"""
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

                                                    
def load_model(model_dir: str):
    print(f"Loading Lychee-rerank-mm from: {model_dir}")
    try:
        min_pixels = 4*28*28
        max_pixels = 1280*28*28
                                                         
        processor = AutoProcessor.from_pretrained(
            model_dir, 
            min_pixels=min_pixels, 
            max_pixels=max_pixels, 
            trust_remote_code=True
        )
        
                                                
                                                     
        model = AutoModelForVision.from_pretrained(
            model_dir, 
            trust_remote_code=True, 
            local_files_only=True,
            torch_dtype=(torch.bfloat16 if AMP_DTYPE is not None else torch.float32),
            device_map="auto" if DEVICE == "cuda" else None,
            attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager"
        )
        model.eval()
        return processor, model
    except Exception as e:
        raise RuntimeError(f"Failed to load model. Error: {e}")

                                                                
def format_content(text: Optional[str], image_path: Optional[str], prefix: str = 'Query:') -> List[Dict[str, Any]]:
    """Helper: 格式化单个部分（Query 或 Document）"""
    content = []
          
    content.append({'type': 'text', 'text': prefix})
    
                                      
    if not text and not image_path:
        return content
        
                                           
    if image_path:
        content.append({'type': 'image', 'image': 'file://' + str(image_path)})
    
    if text:
        content.append({'type': 'text', 'text': text})
        
    return content

def build_lychee_messages(
    instruction: str, 
    query_text: str, 
    query_image_path: Optional[str], 
    doc_text: str, 
    doc_image_path: Optional[str]
) -> List[Dict[str, Any]]:
    """
    构建符合 Lychee 要求的 messages 列表。
    结构:
    System: Judge whether...
    User: <Instruct>: ... <Query>: ... \n<Document>: ...
    """
    inputs = []
    
                      
    inputs.append({
        "role": "system",
        "content": [{
            "type": "text",
            "text": CONFIG["SYSTEM_PROMPT"]
        }]
    })
    
                        
    user_contents = []
    
                
    user_contents.append({
        "type": "text",
        "text": '<Instruct>: ' + instruction
    })
    
                                                         
                                            
                                  
    q_content = format_content(query_text, query_image_path, prefix='<Query>:')
    user_contents.extend(q_content)
    
                
    d_content = format_content(doc_text, doc_image_path, prefix='\n<Document>:')
    user_contents.extend(d_content)
    
    inputs.append({
        "role": "user",
        "content": user_contents
    })
    
    return inputs

                                                    
class LabelScorer:
    def __init__(self, processor: AutoProcessor, model: AutoModelForVision):
        self.proc = processor
        self.model = model
        self.tokenizer = processor.tokenizer                              
        
        self.pos_label = CONFIG["POSITIVE_LABEL"]      
        self.neg_label = CONFIG["NEGATIVE_LABEL"]     
        
                      
        vocab = self.tokenizer.get_vocab()
        self.pos_id = vocab.get(self.pos_label)
        self.neg_id = vocab.get(self.neg_label)
        
        print(f"Label Tokens: '{self.pos_label}'->{self.pos_id}, '{self.neg_label}'->{self.neg_id}")

        if self.pos_id is None or self.neg_id is None:
            raise RuntimeError(f"标签分词为空。请检查 '{self.pos_label}'/'{self.neg_label}' 是否在词表中。")

    @torch.no_grad()
    def score_pair(self, query_text: str, cand_text: str, img_path: str) -> Tuple[float, float, float]:
        """
        返回 (p_yes, yes_logprob, no_logprob)
        """
                     
        instruction = CONFIG["INSTRUCTION"]
                                                                       
        messages = build_lychee_messages(instruction, query_text, None, cand_text, img_path)

                                               
                                                                                                     
        text_input = self.proc.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
                                              
                                                      
        image_inputs, video_inputs = process_vision_info(messages)
        
                   
                                                                                                     
                                                             
        inputs = self.proc(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
                
        inputs = inputs.to(self.model.device)

                   
        if DEVICE == "cuda":
            context = torch.amp.autocast("cuda", enabled=(AMP_DTYPE is not None), dtype=AMP_DTYPE)
        else:
            context = torch.autocast("cpu", enabled=(AMP_DTYPE is not None), dtype=AMP_DTYPE)
            
        with context:
            outputs = self.model(**inputs)
                                  
                                                          
            logits = outputs.logits[:, -1, :]
        
                                  
               
                                                      
                                                        
                                                                        
                                                                             
        
        yes_logit = logits[0, self.pos_id]
        no_logit = logits[0, self.neg_id]
        
        target_logits = torch.stack([no_logit, yes_logit])            
        log_probs = F.log_softmax(target_logits, dim=0)
        
        lp_no = float(log_probs[0].item())
        lp_yes = float(log_probs[1].item())
        
        p_yes = math.exp(lp_yes)
        
        return p_yes, lp_yes, lp_no

                                                   
def main():
    cfg = CONFIG
    print(f"Start processing. Output will be saved to: {cfg['OUT_POINTWISE_JSONL']}")
    
         
    processor, model = load_model(cfg["MODEL_DIR"])
    scorer = LabelScorer(processor, model)

          
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
            
                    
            cand_text = render_candidate_text(cand_raw)
            if not cand_text.strip():
                stats["skipped_empty_text"] += 1
                continue

                
                                                                  
                                                                            
            try:
                p_yes, lp_yes, lp_no = scorer.score_pair(query, cand_text, str(img_path))
            except Exception as e:
                print(f"[Error] Failed to score {cid}: {e}")
                continue

            out_obj = {
                "qid": qid,
                "candidate_id": cid,
                "rank": int(item.get("rank") or 0),
                "is_pos": int(item.get("is_pos") or 0),
                "p_true": p_yes,                             
                "label_true_logprob": lp_yes,                 
                "label_false_logprob": lp_no,                
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