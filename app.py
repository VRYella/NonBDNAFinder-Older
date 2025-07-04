# app.py
import streamlit as st
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

import matplotlib
matplotlib.use("Agg")

st.set_page_config(page_title="Non-B DNA Motif Finder", layout="wide")

EXAMPLE_FASTA = """>Example_NonB_DNA
ATCGATCGATCGAAAATTTTATTTAAATTTAAATTTGGGTTAGGGTTAGGGTTAGGGCCCCCTCCCCCTCCCCCTCCCC
ATCGATCGCGCGCGCGATCGCACACACACAGCTGCTGCTGCTGAAAGAAAGAAAGAAAGAAATTCCTTCCTTCCTTCC
AGAGAGAGAGAGAGAGAGAGAGAGAGTTTTCCTGCCAGAGAGAGCAGAGAGAGGAGAGGGGAGAGGGGAATATAAAT
TTTTTTAATTTTAAAATATAATTTTAATAAAAAATTAAAGGGGTTAGGGTTAGGGTTAGGGTAAAGGGATGCGGGATG
CGGGATGCGGGTTAAAGCTAGCTAGCTAGCTAGCTAGCTTAAAGCTAGCTAGCTAGCTAGCTAGCTTAAAGCTAGCT
AGCTAGCTAGCTAGCTTAAAAAAATAAAAACTAAAAAAGTTAAAAATTAAGGGTTAGGGTTAGGGTTAGGGTTAGGGT
TAAAGGGTTAGGGTTAGGGTTAGGGTTAAAGGGTTAGGGTTAGGGTTAGGGTTAAGCTGCTGCTGCTTTAAAGCTAGC
TAGCTTAAAGGGGGGGGGGTTAAGGGTTAGGGTTAGGGTTAGGGTTAAACCCCTAACCCCTAACCCCTAA
"""

MOTIFS = [
    ("Quadruplex", r"(G{3,}[ATGC]{1,12}){3}G{3,}", "G-Quadruplex"),
    ("Quadruplex", r"(C{3,}[ATGC]{1,12}){3}C{3,}", "i-Motif"),
    ("Quadruplex", r"(G{3,}[ATGC]{1,12}G{3,})([ATGC]{0,100})(G{3,}[ATGC]{1,12}G{3,})", "Bipartite_G-Quadruplex"),
    ("Quadruplex Dimer", r"(G{3,}[ATGC]{1,12}G{3,})([ATGC]{0,100})(G{3,}[ATGC]{1,12}G{3,})", "Asymmetric_Dimeric_G-Quadruplex"),
    ("Quadruplex", r"(G{3,}[AUGC]{1,12}){3}G{3,}", "DNA-RNA_Hybrid_G-Quadruplex"),
    ("Quadruplex", r"(G{3,}[ATGC]{1,12}){2}G{3,}", "G-Triplex"),
    ("Z-DNA", r"((GC|CG|GT|TG|AC|CA){6,})", "Z-DNA"),
    ("Triplex", r"([AG]{10,}|[CT]{10,})([ATGC]{0,100})([AG]{10,}|[CT]{10,})", "H-DNA"),
    ("Triplex", r"(GAA){5,}|(TTC){5,}", "Sticky_DNA"),
    ("Direct Repeat", r"([ATGC]{10,25})([ATGC]{0,10})\1", "Slipped_DNA"),
    ("Inverted Repeat", r"([ATGC]{10,})([ATGC]{0,100})\1", "Cruciform_DNA"),
    ("Hairpin", r"([ATGC]{6,})([ATGC]{0,100})\1", "DNA_Hairpin"),
    ("Hairpin", r"(C{3,10})([ATGC]{0,100})\1", "i-Motif_Hairpin"),
    ("Hairpin", r"(G{3,10})([ATGC]{0,100})\1", "G-Hairpin"),
    ("Hairpin", r"(C{3,10})([ATGC]{0,100})\1", "C-Hairpin"),
    ("Hairpin", r"([ATGC]{6,10})([ATGC]{0,100})\1([ATGC]{0,100})\1", "Hairpin-Loop-Duplex"),
    ("Bent DNA", r"([AT]{3,9})([ATGC]{1,7})\1([ATGC]{1,7})\1", "Bent_DNA"),
    ("Quadruplex-Triplex Hybrid", r"(G{3,}[ATGC]{1,12}){3}G{3,}([ATGC]{0,100})([AG]{10,}|[CT]{10,})", "Quadruplex-Triplex_Hybrid"),
    ("Cruciform-Triplex Junction", r"([ATGC]{10,})([ATGC]{0,100})([ATGC]{10,})([ATGC]{0,100})([AG]{10,}|[CT]{10,})", "Cruciform-Triplex_Junctions"),
    ("G-Quadruplex_i-Motif_Hybrid", r"(G{3,}[ATGC]{1,12}){3}G{3,}([ATGC]{0,100})(C{3,}[ATGC]{1,12}){3}C{3,}", "G-Quadruplex_i-Motif_Hybrid"),
]

def wrap(seq: str, width=40) -> str:
    return "\n".join([seq[i:i+width] for i in range(0, len(seq), width)])

def gc_content(seq: str) -> float:
    seq = seq.upper()
    return 100.0 * (seq.count("G") + seq.count("C")) / max(1, len(seq))

def propensity_score(name: str, seq: str) -> float:
    if name in ["G-Quadruplex", "DNA-RNA_Hybrid_G-Quadruplex"]:
        vals = []
        run = 0
        for base in seq:
            if base == 'G':
                run += 1
                vals.append(min(run, 4))
            else:
                run = 0
                vals.append(0)
        prop = np.mean(vals) if vals else 0
        return round(prop, 2)
    if name == "i-Motif":
        vals = []
        run = 0
        for base in seq:
            if base == 'C':
                run += 1
                vals.append(min(run, 4))
            else:
                run = 0
                vals.append(0)
        prop = np.mean(vals) if vals else 0
        return round(prop, 2)
    if name == "G-Triplex":
        g4_val = propensity_score("G-Quadruplex", seq)
        return round(g4_val * 0.6, 2)
    if name == "Bipartite_G-Quadruplex":
        matches = re.findall(r"(G{3,}[ATGC]{1,12}G{3,})", seq)
        props = [propensity_score("G-Quadruplex", m) for m in matches]
        return round(min(props) if props else 0, 2)
    if name == "Asymmetric_Dimeric_G-Quadruplex":
        matches = re.findall(r"(G{3,}[ATGC]{1,12}G{3,})", seq)
        props = [propensity_score("G-Quadruplex", m) for m in matches]
        return round(np.mean(props) if props else 0, 2)
    if name == "Sticky_DNA":
        count = len(re.findall(r"(GAA|TTC)", seq))
        return round(min(1.0, 0.2 + 0.05 * count), 2)
    if name == "H-DNA":
        matches = re.findall(r"[AG]{10,}|[CT]{10,}", seq)
        prop = 0.3 + 0.02 * min([len(m) for m in matches]) if matches else 0.3
        return round(min(1.0, prop), 2)
    if name == "Z-DNA":
        repeats = len(re.findall(r"(GC|CG|GT|TG|AC|CA)", seq))
        prop = 0.2 + 0.04 * repeats
        return round(min(1.0, prop), 2)
    if name in ["DNA_Hairpin", "i-Motif_Hairpin", "G-Hairpin", "C-Hairpin"]:
        return "NA"
    if name == "Bent_DNA":
        return "NA"
    if name in ["Slipped_DNA", "Cruciform_DNA"]:
        return "NA"
    if name in ["Quadruplex-Triplex_Hybrid", "Cruciform-Triplex_Junctions", "G-Quadruplex_i-Motif_Hybrid"]:
        return "NA"
    return "NA"

def parse_fasta(fasta_text: str) -> str:
    seq = ""
    for line in fasta_text.splitlines():
        line = line.strip()
        if not line or line.startswith(">"):
            continue
        seq += line
    return seq.upper()

def find_motifs_all(seq: str) -> list:
    found = []
    for motif_class, regex, name in MOTIFS:
        for m in re.finditer(regex, seq):
            start, end = m.start(), m.end()-1
            region = seq[start:end+1]
            found.append({
                "class": motif_class,
                "name": name,
                "start": start+1,
                "end": end+1,
                "length": end-start+1,
                "propensity": propensity_score(name, region),
                "gc_content": f"{gc_content(region):.1f}",
                "wrapped": wrap(region),
                "sequence": region,
            })
    return found

def print_longest_of_each_class(motifs: list):
    df = pd.DataFrame(motifs)
    idx = df.groupby("class")["length"].idxmax().dropna()
    longest = df.loc[idx].sort_values("class")
    st.markdown("### Longest motif for each subclass")
    st.dataframe(longest[["class", "name", "start", "end", "length", "propensity", "gc_content", "wrapped"]],
                 use_container_width=True)
    return longest

def print_table(motifs: list):
    df = pd.DataFrame(motifs)
    st.markdown("### All predicted motifs")
    st.dataframe(df[["class", "name", "start", "end", "length", "propensity", "gc_content", "wrapped"]],
                 use_container_width=True)
    return df

def visualize_motifs(motifs: list, seq_len: int):
    st.markdown("### Motif visualization")
    plt.figure(figsize=(15, 8))
    class_names = [cl for cl in [
        "Quadruplex", "Quadruplex Dimer", "Z-DNA", "Triplex", "Direct Repeat",
        "Inverted Repeat", "Hairpin", "Bent DNA", "Quadruplex-Triplex Hybrid",
        "Cruciform-Triplex Junction", "G-Quadruplex_i-Motif_Hybrid"
    ] if any(m['class'] == cl for m in motifs)]
    colormap = plt.colormaps['tab20']
    class2y = {cl: i+1 for i, cl in enumerate(class_names)}
    for m in motifs:
        y = class2y.get(m['class'])
        if y:
            plt.plot([m['start'], m['end']], [y, y], lw=10,
                     color=colormap((y-1) % 20),
                     label=m['class'] if m['start']==min([mm['start'] for mm in motifs if mm['class']==m['class']]) else "")
    plt.yticks(list(class2y.values()), list(class2y.keys()), fontsize=11)
    plt.xlabel("Sequence Position (bp)")
    plt.ylabel("Motif Class")
    plt.title("Non-B DNA Motif Locations by Class", fontsize=14)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    st.pyplot(plt.gcf())

def motif_summary(motifs: list, seq_len: int):
    df = pd.DataFrame(motifs)
    total_motifs = len(df)
    motif_class_counts = df['class'].value_counts().to_dict()
    motif_cov = 0
    covered = set()
    for _, row in df.iterrows():
        covered.update(range(row['start'], row['end']+1))
    motif_cov = len(covered)
    st.markdown("### Motif Summary")
    st.write(f"**Total motifs detected:** {total_motifs}")
    st.write(f"**Motif class counts:** {motif_class_counts}")
    st.write(f"**Total coverage:** {motif_cov} bp ({motif_cov*100/seq_len:.1f}% of sequence)")

    # Pie chart for motif class frequency
    if motif_class_counts:
        chart_df = pd.DataFrame(list(motif_class_counts.items()), columns=["Motif Class", "Count"])
        st.pyplot(chart_df.set_index("Motif Class").plot.pie(y="Count", legend=False, autopct='%1.0f%%', figsize=(4,4)).get_figure())

def csv_download_button(df, name):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"Download {name} as CSV",
        data=csv,
        file_name=f"{name}.csv",
        mime="text/csv"
    )

def excel_download_button(df, name):
    output = StringIO()
    df.to_excel(output, index=False, engine='openpyxl')
    st.download_button(
        label=f"Download {name} as Excel",
        data=output.getvalue(),
        file_name=f"{name}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

st.title("Non-B DNA Motif Finder")
st.write("Paste or upload a DNA sequence in FASTA format. This tool predicts all scientifically described non-B DNA motifs, their location, length, and non-B propensity (where possible). Download results as CSV or Excel.")

col1, col2 = st.columns([1,1])
with col1:
    fasta_text = st.text_area(
        "Paste your FASTA sequence here (single or multiline)", 
        value="", 
        height=180,
        placeholder=">seq1\nATGCATGCATGC...\n"
    )
    if st.button("Load Example Sequence"):
        fasta_text = EXAMPLE_FASTA

with col2:
    fasta_file = st.file_uploader("Or upload a FASTA file", type=["fa", "fasta", "txt"])
    if fasta_file:
        fasta_bytes = fasta_file.read()
        fasta_text = fasta_bytes.decode("utf-8")

if not fasta_text.strip():
    st.info("Please paste or upload a sequence. Example: click 'Load Example Sequence' button.")
    st.stop()

seq = parse_fasta(fasta_text)
if not seq or not all(c in "ATGCUatgcu" for c in seq):
    st.error("No valid DNA sequence detected (sequence should contain only A,T,G,C,U).")
    st.stop()

st.success(f"Loaded sequence: {len(seq)} bases")
st.code(wrap(seq), language="text")

motifs = find_motifs_all(seq)
if not motifs:
    st.warning("No non-B DNA motifs found in the input sequence.")
else:
    summary_placeholder = st.empty()
    longest = print_longest_of_each_class(motifs)
    df = print_table(motifs)
    csv_download_button(df, "all_motifs")
    excel_download_button(df, "all_motifs")
    csv_download_button(longest, "longest_per_class")
    excel_download_button(longest, "longest_per_class")
    motif_summary(motifs, len(seq))
    visualize_motifs(motifs, len(seq))
