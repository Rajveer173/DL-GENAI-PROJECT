import re
import string
from collections import defaultdict
from pathlib import Path

import pandas as pd
import streamlit as st
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer

OPTIONS = ["A", "B", "C", "D", "E"]
DATA_DIR = Path(__file__).parent / "data"
ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MIN_MATCH_SIM = 0.60
AMBIGUOUS_MARGIN = 0.05

PROMPT_PREFIXES = [
    r"^pick the best possible answer\s*:?\s*",
    r"^select the most accurate option\s*:?\s*",
    r"^determine the correct option\s*:?\s*",
    r"^identify the correct statement\s*:?\s*",
    r"^choose the correct answer\s*:?\s*",
    r"^which of the following is correct\??\s*:?\s*",
    r"^which of the following statements is true (about|regarding)\s*:?\s*",
    r"^which of the following statements accurately (describes|depicts)\s*:?\s*",
]

PROMPT_SUFFIXES = [
    r"\s*among the listed options\.?$",
    r"\s*from the following choices\.?$",
    r"\s*based on the given context\.?$",
    r"\s*carefully\.?$",
]


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def extract_query(prompt):
    text = str(prompt).strip()
    for pat in PROMPT_PREFIXES:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    for pat in PROMPT_SUFFIXES:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    return text.strip()


def question_stem(prompt):
    return clean_text(extract_query(prompt))


def question_key(prompt, option_text):
    opts = "|".join(clean_text(option_text[o]) for o in OPTIONS)
    return f"{question_stem(prompt)}||{opts}"


@st.cache_resource(show_spinner="Loading the training question bank...")
def load_bank():
    df = pd.read_csv(DATA_DIR / "train.csv")
    df["stem"] = df["prompt"].apply(question_stem)
    df["qkey"] = df.apply(
        lambda r: question_key(r["prompt"], {o: r[o] for o in OPTIONS}), axis=1
    )
    df = df.drop_duplicates(subset="qkey", keep="first").reset_index(drop=True)

    qkey_to_answer = df.set_index("qkey")["answer"].to_dict()
    stem_to_rows = defaultdict(list)
    for row in df.to_dict("records"):
        stem_to_rows[row["stem"]].append(row)
    return df, qkey_to_answer, dict(stem_to_rows)


@st.cache_resource(show_spinner="Loading the MiniLM ranker...")
def load_encoder():
    return SentenceTransformer(ENCODER_NAME)


@st.cache_data(show_spinner=False)
def load_examples():
    path = DATA_DIR / "test.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def match_rank1(prompt, option_text, qkey_to_answer, stem_to_rows):
    qkey = question_key(prompt, option_text)
    if qkey in qkey_to_answer:
        return "exact", qkey_to_answer[qkey], None, {}

    variants = stem_to_rows.get(question_stem(prompt), [])
    if not variants:
        return "none", None, None, {}

    cleaned = {o: clean_text(option_text[o]) for o in OPTIONS}
    answer_text = [clean_text(v[v["answer"]]) for v in variants]
    sims = {
        o: max(SequenceMatcher(None, cleaned[o], t).ratio() for t in answer_text)
        for o in OPTIONS
    }
    ranked = sorted(((sims[o], o) for o in OPTIONS), reverse=True)
    (best_sim, best_opt), (second_sim, second_opt) = ranked[0], ranked[1]

    if best_sim < MIN_MATCH_SIM:
        return "none", None, None, sims
    runner_up = second_opt if best_sim - second_sim < AMBIGUOUS_MARGIN else None
    return "paraphrase", best_opt, runner_up, sims


def ranker_scores(prompt, option_text, encoder):
    texts = [extract_query(prompt)] + [str(option_text[o]) for o in OPTIONS]
    emb = encoder.encode(texts, normalize_embeddings=True)
    return dict(zip(OPTIONS, emb[1:] @ emb[0]))


def solve(prompt, option_text, bank, encoder):
    _, qkey_to_answer, stem_to_rows = bank
    kind, rank1, runner_up, match_sims = match_rank1(
        prompt, option_text, qkey_to_answer, stem_to_rows
    )
    scores = ranker_scores(prompt, option_text, encoder)
    model_order = sorted(OPTIONS, key=lambda o: -scores[o])

    head = [] if rank1 is None else ([rank1] if runner_up is None else [rank1, runner_up])
    rest = [o for o in model_order if o not in head]
    return (head + rest)[:3], kind, match_sims, scores


st.set_page_config(page_title="Smart MCQ Solver", layout="wide")

st.title("Smart MCQ Solver")
st.caption(
    "Ranks the top three options for a multiple-choice question, using the "
    "question-matching pipeline from the Kaggle submission."
)

bank = load_bank()
bank_df = bank[0]

with st.sidebar:
    st.header("Pipeline")
    st.markdown(
        "**Rank 1** — question matching against the labelled training bank:\n\n"
        "1. *exact* — the question key (stem + five options) is already in the bank, "
        "so its training label is used verbatim.\n"
        "2. *paraphrase* — same stem, reworded options: each option is scored by its "
        "best similarity to the answer text of any training variant.\n"
        "3. *none* — no trusted match, so the ranker alone decides.\n\n"
        "**Ranks 2-3** — ordered by MiniLM cosine similarity, which never sees a label."
    )
    st.divider()
    st.metric("Unique questions in bank", f"{len(bank_df):,}")
    st.metric("Unique stems", f"{bank_df['stem'].nunique():,}")
    st.caption(f"Ranker: `{ENCODER_NAME}`")

examples = load_examples()
if not examples.empty:
    cols = st.columns([3, 1])
    with cols[0]:
        idx = st.selectbox(
            "Load an example from the test set",
            options=range(len(examples)),
            format_func=lambda i: f"{examples.iloc[i]['id']} — {examples.iloc[i]['prompt'][:90]}",
        )
    with cols[1]:
        st.write("")
        st.write("")
        if st.button("Load example", use_container_width=True):
            row = examples.iloc[idx]
            st.session_state["prompt"] = row["prompt"]
            for o in OPTIONS:
                st.session_state[f"opt_{o}"] = row[o]

prompt = st.text_area("Question", key="prompt", height=100)

st.write("**Options**")
option_text = {}
for o in OPTIONS:
    option_text[o] = st.text_area(o, key=f"opt_{o}", height=80)

if st.button("Solve", type="primary"):
    if not prompt.strip() or any(not option_text[o].strip() for o in OPTIONS):
        st.warning("Fill in the question and all five options.")
    else:
        encoder = load_encoder()
        top3, kind, match_sims, scores = solve(prompt, option_text, bank, encoder)

        st.subheader("Prediction")
        st.markdown("## " + "  ".join(f"`{i}. {o}`" for i, o in enumerate(top3, 1)))

        badge = {
            "exact": ("Exact match", "This question is in the training bank; rank 1 is its label."),
            "paraphrase": ("Paraphrase match", "Same stem as a training question, options reworded."),
            "none": ("No match", "Not found in the bank; all three ranks come from the ranker."),
        }[kind]
        st.info(f"**{badge[0]}** — {badge[1]}")

        for rank, o in enumerate(top3, 1):
            with st.expander(f"Rank {rank}: option {o}", expanded=rank == 1):
                st.write(option_text[o])

        table = pd.DataFrame(
            {
                "Option": OPTIONS,
                "Answer-text similarity": [match_sims.get(o, float("nan")) for o in OPTIONS],
                "MiniLM cosine": [scores[o] for o in OPTIONS],
                "Predicted rank": [
                    top3.index(o) + 1 if o in top3 else None for o in OPTIONS
                ],
            }
        )
        st.dataframe(
            table.style.format(
                {"Answer-text similarity": "{:.3f}", "MiniLM cosine": "{:.3f}"},
                na_rep="—",
            ),
            hide_index=True,
            use_container_width=True,
        )
        if kind == "exact":
            st.caption(
                "Answer-text similarity is not computed for exact matches — the "
                "training label is used directly."
            )
