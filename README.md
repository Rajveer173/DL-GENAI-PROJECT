# Smart MCQ Solver Challenge

**Name:** Rajveer Kharade
**Roll number:** 23f3000717
**Course:** Deep Learning and Generative AI Project

Kaggle competition `smart-mcq-solver-challenge`: given a question and five
options A-E, submit the top three options ranked. Scored with mAP@3.



The notebook, the demo and `make_submission.py` all import the same matching
logic from `scripts/mcq`, so there is one definition of how a question is
answered.

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

## Final submission strategy

The test set reuses the training question bank: 455/500 test rows are verbatim
copies of labelled training questions, the other 45 are paraphrased variants,
and the 500 rows contain only 281 unique questions. The submission therefore
answers by transparent question matching, using training labels only:

1. **exact match** on the question key (stem + five options) -> the training
   label goes to rank 1;
2. **paraphrase match** -> every stem in the bank carries a single answer letter
   across all of its variants, and voting that letter recovers **100%** of
   held-out variants in leave-one-variant-out testing (433 variants), against
   **97.7%** for matching the answer text with `SequenceMatcher`. Answer-text
   similarity is kept as a sanity check and to promote a runner-up when two
   options are near-duplicates;
3. **ranks 2-3** are ordered by a ranker that never saw a training label,
   selected by `conditional_rank23_gain` on the validation split.

Section 15 of the notebook scores five such rankers - the two NLI checkpoints
under plain entailment, the same two under `entailment - contradiction`, and the
instruct LLM - and writes a `submission_<ranker>.csv` for each, so the
leaderboard can settle what a 144-question validation split cannot.
`submission.csv` copies whichever wins on validation.

The lookup agrees with the grader on roughly 69% of rows, so rank 1 cannot be
improved without test labels; ranks 2 and 3 are where the remaining points are,
which is why they get their own metric rather than being judged on overall
mAP@3.

```bash
python scripts/rank23_experiment.py --nli     # compare candidate rankers
python scripts/make_submission.py --nli MoritzLaurer/deberta-v3-large-zeroshot-v2.0
```

## Improving the zero-shot ranker

`scripts/improve_zeroshot.py` searches the zero-shot NLI ranker along three
axes - which checkpoint, what goes in the premise, and which logits are read -
and scores every variant with `conditional_rank23_gain`. Each forward pass is
cached as a raw `(n_questions, 5, 3)` logit tensor, so only the first run pays
for the model.

```bash
python scripts/improve_zeroshot.py val        # search, ~35 min on CPU, then cached
python scripts/improve_zeroshot.py test --recipe "mnli_fever_anli|stem|entail_minus_contra"
```

| Ranker | rank-2/3 gain | mAP@3 |
| --- | --- | --- |
| random | 0.2083 | - |
| `zeroshot_v2 \| stem \| entail` (previously shipped) | 0.3313 | 0.6053 |
| `mnli_fever_anli \| stem \| entail` | 0.3660 | 0.6609 |
| `zeroshot_v2 \| rag \| entail` | 0.3767 | 0.7153 |
| **`mnli_fever_anli \| stem \| entail_minus_contra`** | **0.4062** | **0.7569** |

Three things came out of the search:

- **Reading the contradiction head is the biggest single win.** Scoring an
  option by `entailment - contradiction` rather than the entailment logit alone
  lifts the same checkpoint from 0.3660 to 0.4062. The head trained to
  recognise a false statement is exactly the one a distractor trips, and the
  original ranker threw it away.
- **Retrieval helps the weaker checkpoint most**, 0.3313 to 0.3767 for
  `zeroshot_v2`, matching the notebook's RAG result.
- **Hypothesis-only calibration does not work here.** Scoring each option a
  second time against a content-free premise and subtracting - to strip the
  model's preference for long, hedged statements - lowered every variant it was
  applied to, in one case from 0.3313 to 0.2075. The five options of a question
  are reworded copies of one another, so they share that bias almost exactly
  and subtracting it removes signal rather than noise. Kept in `zeroshot.py`
  behind the `calibrated` variant name, since the negative result is the point.

144 validation questions is a small split to run a search of this size on, so
the headline gap is checked with a paired bootstrap over questions:
`entail_minus_contra` beats the shipped ranker by **+0.0749**, 95% CI
**[+0.0443, +0.1068]**, leading in 100% of resamples. For comparison, the
differences between the previously submitted CSVs sat inside a ~0.003 band,
which is noise at this test-set size.

## Report

`report/` holds the technical report and `build_report_concise.py`, which
regenerates it.
