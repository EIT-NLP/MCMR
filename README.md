<div align="center">
  <h1>Beyond Global Similarity: Towards Fine-Grained, Multi-Condition Multimodal Retrieval (CVPR 2026)</h1>
  Xuan Lu<sup>1,2,3</sup>, Kangle Li<sup>1,2*</sup>, Haohang Huang<sup>3</sup>, Rui Meng, Wenjun Zeng<sup>2,3</sup>, Xiaoyu Shen<sup>2,3</sup><br/>
  <sup>1</sup> Shanghai Jiao Tong University (SJTU)<br/>
  <sup>2</sup> Institute of Digital Twin, Eastern Institute of Technology (EIT), Ningbo<br/>
  <sup>3</sup> Ningbo Key Laboratory of Spatial Intelligence and Digital Derivative
</div>

<p align="center">
  <a href="https://arxiv.org/abs/2603.01082"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b.svg" alt="Paper"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-green.svg" alt="License"></a>
  <a href="https://huggingface.co/datasets/Lux1997/MCMR" target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace%20Datasets-27b3b4.svg></a>
</p>

<!-- <p align="center" width="100%">
  <a target="_blank">
    <img src="./assets/mcmr_overview.png" alt="MCMR Overview" style="width: 90%; min-width: 200px; display: block; margin: auto;">
  </a>
</p> -->


## **🌟 Introduction**

[**MCMR (Multi-Conditional Multimodal Retrieval)** is a large-scale, high-difficulty benchmark designed to evaluate fine-grained and multi-condition cross-modal retrieval. Unlike traditional retrieval tasks that rely on coarse-grained global similarity, MCMR emphasizes the alignment of complex, interdependent constraints across visual and textual modalities.
](https://arxiv.org/abs/2603.01082)

- **Diverse Domain Coverage:** Spans five distinct product domains: Upper Clothing, Bottom Clothing, Jewelry, Shoes, and Furniture.
- **Fine-grained Multi-Condition Queries:** Each query integrates complementary visual cues and textual attributes, requiring models to satisfy all specified constraints simultaneously.


- **Rich Contextual Metadata:** Preserves long-form textual metadata to facilitate research on compositional matching and complex attribute reasoning.
- **Standardized Evaluation Framework:**  Provides a unified implementation for both MLLM-based retrievers (e.g., CORAL, VLM2Vec) and vision-language rerankers (e.g., Qwen-VL, InternVL).

---

## **🛠️ Installation**

We recommend using Conda to manage your environment. Follow the steps below to set up the necessary dependencies:

```bash
# 1) Create and activate a dedicated conda environment
conda create -n mcmr python=3.10 -y
conda activate mcmr

# 2) Upgrade pip
pip install --upgrade pip

# 3) Install project dependencies
pip install -r requirements.txt
```

---

## **📊 Dataset Preparation**

### **1) Data Acquisition**
The MCMR dataset is hosted on Hugging Face: https://huggingface.co/datasets/Lux1997/MCMR

### **2) Directory Structure**
Ensure your data directory follows this hierarchy:

```text
data/
└── mcmr/
    ├── images/             # Extracted product images (from images.tar.gz)
    ├── candidate.jsonl     # Catalog of candidate items with metadata
    └── query.jsonl         # Multi-condition retrieval queries
```

---

## **🚀 Evaluation and Usage**

The MCMR evaluation pipeline consists of two sequential phases: Dense Retrieval and Fine-grained Reranking.

### **Phase 1: Dense Retrieval**

The retrieval stage aims to efficiently narrow down millions of candidates to a manageable top-$K$ subset. We provide implementations for several state-of-the-art retrievers.

To run a retrieval script (e.g., using CORAL):

```bash
python eval/retrieval/coral.py
```
### **Customization**

You can modify the following variables within the scripts to point to your custom file locations:


- **CANDIDATES:** path to candidate metadata
- **IMAGE_DIR:** path to the image root directory
- **QUERIES:** path to the query file

### **Phase 2: Fine-grained Reranking**

Reranking uses powerful **Vision-Language Models (VLMs)** to re-evaluate the retrieved top‑K results.

#### **Step A — Generate Top‑K Candidates**
First, generate a top‑50 candidate file ( topk50.jsonl).  
We recommend using `llave.py`:

```bash
python eval/retrieval/llave.py
```

#### **Step B — Execute Reranking**
Once `topk50.jsonl` is ready, run a reranking script (e.g., **InternVL3** pointwise):

```bash
python eval/rerank/internvl3_pointwise.py
```

---

### **Common Path Configuration**


The following variables are available for customization in reranking scripts:


**Reranking scripts**
- **INPUT_TOPK_JSONL:** Path to the generated top‑K file
- **OUT_POINTWISE_JSONL:** Destination path for reranking results
- **IMAGE_DIRS:** Path to the image assets

## 📚 Citation
```bibtex
@misc{lu2026globalsimilarityfinegrainedmulticondition,
      title={Beyond Global Similarity: Towards Fine-Grained, Multi-Condition Multimodal Retrieval}, 
      author={Xuan Lu and Kangle Li and Haohang Huang and Rui Meng and Wenjun Zeng and Xiaoyu Shen},
      year={2026},
      eprint={2603.01082},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.01082}, 
}
```
