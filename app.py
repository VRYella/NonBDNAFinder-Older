
# ======================
# utils.py
# ======================
import re
import numpy as np

def parse_fasta(fasta_str: str) -> str:
    lines = fasta_str.strip().splitlines()
    seq = [line.strip() for line in lines if not line.startswith(">")]
    return "".join(seq).upper().replace(" ", "").replace("U", "T")

def wrap(seq: str, width=60) -> str:
    return "\n".join([seq[i:i+width] for i in range(0, len(seq), width)])

def g4hunter_score(seq: str) -> float:
    vals = []
    n = len(seq)
    i = 0
    while i < n:
        s = seq[i]
        if s == 'G':
            run_len = 1
            while i + run_len < n and seq[i + run_len] == 'G':
                run_len += 1
            score = min(run_len, 4)
            for _ in range(run_len):
                vals.append(score)
            i += run_len
        elif s == 'C':
            run_len = 1
            while i + run_len < n and seq[i + run_len] == 'C':
                run_len += 1
            score = -min(run_len, 4)
            for _ in range(run_len):
                vals.append(score)
            i += run_len
        else:
            vals.append(0)
            i += 1
    return np.mean(np.array(vals)) if vals else 0.0

def zseeker_score(seq: str) -> float:
    dinucs = re.findall(r"(GC|CG|GT|TG|AC|CA)", seq)
    return len(dinucs) / (len(seq)/2) if len(seq) >= 2 else 0.0

# ======================
# Streamlit App
# ======================
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
from datetime import datetime

EXAMPLE_FASTA = ">Example\nATCGATCGATCGAAAATTTTATTTAAATTTAAATTTGGGTTAGGGTTAGGGTTAGGGCCCCCTCCCCCTCCCCCTCCCC\nATCGATCGCGCGCGCGATCGCACACACACAGCTGCTGCTGCTTGGGAAAGGGGAAGGGTTAGGGAAAGGGGTTT\nGGGTTTAGGGGGGAGGGGCTGCTGCTGCATGCGGGAAGGGAGGGTAGAGGGTCCGGTAGGAACCCCTAACCCCTAA\nGAAAGAAGAAGAAGAAGAAGAAAGGAAGGAAGGAGGAGGAGGAGGAGGAGGAGGAGGAGGAGGAGGAGGAGGG"

st.set_page_config(page_title="Non-B DNA Motif Finder", layout="wide")
st.title("Non-B DNA Motif Finder (Single File Version)")

uploaded = st.file_uploader("Upload FASTA", type=["fa", "fasta", "txt"])
sequence_input = ""

if uploaded:
    try:
        sequence_input = parse_fasta(uploaded.read().decode())
        st.success("Sequence uploaded successfully.")
    except:
        st.error("Invalid FASTA format.")
else:
    if st.button("Use Example Sequence"):
        sequence_input = parse_fasta(EXAMPLE_FASTA)

if sequence_input:
    st.text_area("Input Sequence", sequence_input, height=150)

    def non_overlapping_finditer(pattern, seq):
        regex = re.compile(pattern)
        pos = 0
        while pos < len(seq):
            match = regex.search(seq, pos)
            if not match:
                break
            yield match
            pos = match.end()

    def create_motif_dict(cls, subtype, match, seq, score_method="None", score="0", group=0):
        sequence = match.group(group)
        return {
            "Class": cls, "Subtype": subtype, "Start": match.start()+1,
            "End": match.start()+len(sequence), "Length": len(sequence),
            "Sequence": wrap(sequence), "ScoreMethod": score_method, "Score": score
        }

    def find_motif(seq, pattern, cls, subtype, score_method="None", score_func=None, group=0):
        results = []
        for m in non_overlapping_finditer(pattern, seq):
            score = f"{score_func(m.group(group)):.2f}" if score_func else "0"
            results.append(create_motif_dict(cls, subtype, m, seq, score_method, score, group))
        return results

    motifs = []
    motifs += find_motif(sequence_input, r"(?=(G{3,}([ATGC]{1,7}G{3,}){3}))", "Quadruplex", "Canonical_G-Quadruplex", "G4Hunter", g4hunter_score)
    motifs += find_motif(sequence_input, r"(?=((?:CG){6,}))", "Z-DNA", "CG_Repeat", "ZSeeker", zseeker_score, 1)

    df = pd.DataFrame(motifs)
    if not df.empty:
        st.success(f"{len(df)} motifs found.")
        st.dataframe(df)

        fig, ax = plt.subplots(figsize=(10, 4))
        sns.countplot(data=df, x="Subtype", ax=ax)
        ax.set_title("Motif Type Distribution")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📄 Download CSV", data=csv, file_name="motif_results.csv", mime="text/csv")
    else:
        st.warning("No motifs found.")
