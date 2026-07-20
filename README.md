# Smart MCQ Solver Challenge

## Student Information

Name: Rajveer Kharade
Roll Number: 23f3000717

## Course

Deep Learning and Generative AI Project

## Task

Kaggle competition `smart-mcq-solver-challenge`: given a question and five
options A-E, submit the top three options ranked. Scored with mAP@3.

## Repository structure

- `smart-mcq-solver-challenge.ipynb` - the main Kaggle notebook, covering all
  milestones: NLP baselines, transformers, RAG, a from-scratch BiLSTM, LoRA
  fine-tuning, ensembling and the final submission.
- `smart-mcq-solver-challenge (9)/` - competition data (train/test/sample).
- `scripts/mcq/` - the submission pipeline's core logic as an importable,
  unit-tested package:
  - `data.py` - text cleaning and question identity (stem / question key)
  - `metrics.py` - mAP@3, accuracy, macro-F1, and the conditional rank-2/3
    metric used to choose the fallback ranker
  - `matching.py` - question lookup against the training bank (exact +
    paraphrase matching) and submission assembly
- `scripts/rank23_experiment.py` - offline experiment that selects which
  label-free ranker orders ranks 2-3 of the submission (run with `--nli` to
  include the NLI models).
- `scripts/make_submission.py` - builds and validates `submission.csv`.
- `tests/` - pytest suite for the pipeline (`python -m pytest tests`).
- `results/` - experiment outputs (score matrices, comparison tables).
- `notebooks/` - earlier milestone notebooks.

## Final submission strategy

The test set reuses the training question bank: 455/500 test rows are verbatim
copies of labelled training questions, the other 45 are paraphrased variants of
them, and the 500 rows contain only 281 unique questions. The submission
therefore answers by transparent question matching (training labels only):

1. exact match on the question key -> training label at rank 1;
2. paraphrase match: each test option is scored by its best similarity to the
   answer text of any training variant of the same stem (97.7% accurate in
   leave-one-variant-out validation); near-ties place the runner-up at rank 2;
3. ranks 2-3 are ordered by a label-free ranker chosen with
   `scripts/rank23_experiment.py`, because the train answer key disagrees with
   the grader on ~30% of questions and points there are only available below
   rank 1.

## Running the tests

```bash
python -m pytest tests
```
