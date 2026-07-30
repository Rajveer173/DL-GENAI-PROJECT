import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "scripts"))

from mcq.data import OPTIONS, add_question_keys, extract_query, question_key, question_stem
from mcq.matching import answer_similarity, build_lookup, match_rank1

DATA_DIR = ROOT / "data"
ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@st.cache_resource(show_spinner="Loading the training question bank...")
def load_bank():
    df = add_question_keys(pd.read_csv(DATA_DIR / "train.csv"))
    df = df.drop_duplicates(subset="qkey", keep="first").reset_index(drop=True)
    qkey_to_answer, stem_to_rows = build_lookup(df)
    return df, qkey_to_answer, dict(stem_to_rows)


@st.cache_resource(show_spinner="Loading the MiniLM ranker...")
def load_encoder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(ENCODER_NAME)


@st.cache_data(show_spinner=False)
def load_examples():
    path = DATA_DIR / "test.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def ranker_scores(prompt, option_text, encoder):
    """MiniLM cosine similarity between the question and each option."""
    texts = [extract_query(prompt)] + [str(option_text[o]) for o in OPTIONS]
    emb = encoder.encode(texts, normalize_embeddings=True)
    return dict(zip(OPTIONS, emb[1:] @ emb[0]))


def solve(prompt, option_text, bank, encoder):
    _, qkey_to_answer, stem_to_rows = bank
    row = {"prompt": prompt, **option_text}
    row["stem"] = question_stem(row)
    row["qkey"] = question_key(row)

    kind, rank1, runner_up = match_rank1(row, qkey_to_answer, stem_to_rows)
    variants = stem_to_rows.get(row["stem"], [])
    similarity = answer_similarity(row, variants) if kind == "paraphrase" else {}
    letters = Counter(v["answer"] for v in variants) if variants else Counter()

    scores = ranker_scores(prompt, option_text, encoder)
    model_order = sorted(OPTIONS, key=lambda o: -scores[o])
    head = [] if rank1 is None else [r for r in (rank1, runner_up) if r]
    rest = [o for o in model_order if o not in head]
    return (head + rest)[:3], kind, similarity, scores, letters


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
        "**Rank 1** — matched against the labelled training bank:\n\n"
        "1. *exact* — the question key (stem + five options) is already in the "
        "bank, so its training label is used verbatim.\n"
        "2. *paraphrase* — same stem, reworded options. Every stem carries one "
        "answer letter across all its variants, so that letter is voted; "
        "answer-text similarity is kept as a sanity check and to promote a "
        "runner-up when two options are near-duplicates.\n"
        "3. *none* — no trusted match, so the ranker alone decides.\n\n"
        "**Ranks 2-3** — ordered by MiniLM cosine similarity, which never sees "
        "a label."
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
        if st.button("Load example", width="stretch"):
            row = examples.iloc[idx]
            st.session_state["prompt"] = row["prompt"]
            for o in OPTIONS:
                st.session_state[f"opt_{o}"] = row[o]

prompt = st.text_area("Question", key="prompt", height=100)

st.write("**Options**")
option_text = {o: st.text_area(o, key=f"opt_{o}", height=80) for o in OPTIONS}

if st.button("Solve", type="primary"):
    if not prompt.strip() or any(not option_text[o].strip() for o in OPTIONS):
        st.warning("Fill in the question and all five options.")
    else:
        top3, kind, similarity, scores, letters = solve(
            prompt, option_text, bank, load_encoder()
        )

        st.subheader("Prediction")
        st.markdown("## " + "  ".join(f"`{i}. {o}`" for i, o in enumerate(top3, 1)))

        explanation = {
            "exact": "This question is in the training bank; rank 1 is its label.",
            "paraphrase": (
                "Same stem as a training question with reworded options. "
                f"Variants in the bank answer: {dict(letters)}."
            ),
            "none": "Not found in the bank; all three ranks come from the ranker.",
        }[kind]
        st.info(f"**{kind.capitalize()} match** — {explanation}")

        for rank, o in enumerate(top3, 1):
            with st.expander(f"Rank {rank}: option {o}", expanded=rank == 1):
                st.write(option_text[o])

        table = pd.DataFrame({
            "Option": OPTIONS,
            "Answer-text similarity": [similarity.get(o, float("nan")) for o in OPTIONS],
            "MiniLM cosine": [scores[o] for o in OPTIONS],
            "Predicted rank": [top3.index(o) + 1 if o in top3 else None for o in OPTIONS],
        })
        st.dataframe(
            table.style.format(
                {"Answer-text similarity": "{:.3f}", "MiniLM cosine": "{:.3f}"},
                na_rep="—",
            ),
            hide_index=True,
            width="stretch",
        )
        if kind != "paraphrase":
            st.caption(
                "Answer-text similarity is only computed for paraphrase matches."
            )
