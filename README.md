# Smart MCQ Solver Challenge

**Name:** Rajveer Kharade
**Roll number:** 23f3000717
**Course:** Deep Learning and Generative AI Project

Kaggle competition `smart-mcq-solver-challenge`: given a question and five
options A-E, submit the top three options ranked. Scored with mAP@3.




## Setup

```bash
pip install -r requirements.txt
python -m pytest tests          # 47 tests, ~20s
streamlit run app.py            # the demo
```

The notebook is meant to run on Kaggle with a T4 and installs its own extra
dependencies (`peft`, `datasets`, `faiss-cpu`, `bitsandbytes`) in the cells that
need them. The W&B key is set in Section 6; the notebook is private and shared
only with the competition admin.

## Models

| Section | Model | Role |
| --- | --- | --- |
| 7 | TF-IDF, Word2Vec, random | Milestone 1 baselines |
| 8 | MiniLM sentence embeddings | pretrained encoder, cosine ranking |
| 9 | DeBERTa-v3-large NLI | zero-shot entailment ranking |
| 10 | MiniLM + FAISS retrieval | Milestone 3, RAG |
| 11 | **BiLSTM ranker, built from scratch** | own vocab, dataset, architecture, training loop |
| 12 | **LoRA multiple-choice DeBERTa-v3-base** | Milestone 4, fine-tuned |
| 13 | Qwen2.5-7B-Instruct (4-bit) | answer-letter logits, strongest label-free ranker |
| 14 | Rank-average ensemble | Milestone 5 |

Both trained models rank the five options jointly — one score per option and a
softmax across them — rather than classifying each (question, option) pair
independently. Every run is logged to W&B project `23f3000717-dl-genai-project`
with mAP@3, accuracy and macro-F1, and the two trained models log a per-epoch
curve.




## Improving the zero-shot ranker

`scripts/improve_zeroshot.py` searches the zero-shot NLI ranker along three
axes - which checkpoint, what goes in the premise, and which logits are read -
and scores every variant with `conditional_rank23_gain`. Each forward pass is
cached as a raw `(n_questions, 5, 3)` logit tensor, so only the first run pays
for the model.



| Ranker | rank-2/3 gain | mAP@3 |
| --- | --- | --- |
| random | 0.2083 | - |
| `zeroshot_v2 \| stem \| entail` (previously shipped) | 0.3313 | 0.6053 |
| `mnli_fever_anli \| stem \| entail` | 0.3660 | 0.6609 |
| `zeroshot_v2 \| rag \| entail` | 0.3767 | 0.7153 |
| **`mnli_fever_anli \| stem \| entail_minus_contra`** | **0.4062** | **0.7569** |


