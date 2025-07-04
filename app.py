import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import matplotlib.pyplot as plt

st.set_page_config(page_title="Non-B DNA Motif Finder", layout="wide")

MOTIFS = [
    # G-quadruplex (canonical: 4 G-runs separated by 1-12 nt)
    ("Quadruplex", r"(G{3,}[ATGC]{1,12}){3}G{3,}", "G-Quadruplex"),
    # i-Motif (C-rich counterpart, similar loop size)
    ("Quadruplex", r"(C{3,}[ATGC]{1,12}){3}C{3,}", "i-Motif"),
    # Bipartite G-quadruplex (stricter: 2 G4 blocks separated by up to 100nt, both G4s)
    ("Quadruplex", r"(G{3,}[ATGC]{1,12}){3}G{3,}[ATGC]{0,100}(G{3,}[ATGC]{1,12}){3}G{3,}", "Bipartite_G-Quadruplex"),
    # G-triplex (3 G runs, 1–12nt loops)
    ("Quadruplex", r"(G{3,}[ATGC]{1,12}){2}G{3,}", "G-Triplex"),
    # Z-DNA (alternating purine/pyrimidine, 6+ repeats, e.g. GC, CG, GT, TG, AC, CA)
    ("Z-DNA", r"((GC|CG|GT|TG|AC|CA){6,})", "Z-DNA"),
    # Triplex-forming mirror repeat (H-DNA, purine or pyrimidine arms, 10+nt each, ≤100nt loop)
    ("Triplex", r"([AG]{10,}|[CT]{10,})([ATGC]{0,100})([AG]{10,}|[CT]{10,})", "H-DNA"),
    # Sticky DNA (GAA/TTC repeats, 5+ units)
    ("Triplex", r"(GAA){5,}|(TTC){5,}", "Sticky_DNA"),
    # Direct Repeat (slipped DNA)
    ("Direct Repeat", r"([ATGC]{10,25})([ATGC]{0,10})\1", "Slipped_DNA"),
    # Inverted Repeat (cruciform)
    ("Inverted Repeat", r"([ATGC]{10,})([ATGC]{0,100})\1", "Cruciform_DNA"),
    # Hairpin (6+ nt stem, loop up to 100 nt)
    ("Hairpin", r"([ATGC]{6,})([ATGC]{0,100})\1", "DNA_Hairpin"),
    ("Hairpin", r"(C{3,10})([ATGC]{0,100})\1", "i-Motif_Hairpin"),
    ("Hairpin", r"(G{3,10})([ATGC]{0,100})\1", "G-Hairpin"),
    ("Hairpin", r"(C{3,10})([ATGC]{0,100})\1", "C-Hairpin"),
    ("Hairpin", r"([ATGC]{6,10})([ATGC]{0,100})\1([ATGC]{0,100})\1", "Hairpin-Loop-Duplex"),
    # Bent DNA (A4-6 or T4-6 phased by 7–11 nt, at least three phased tracts)
    ("Bent DNA", r"(A{4,6}|T{4,6})([ATGC]{7,11})(A{4,6}|T{4,6})([ATGC]{7,11})(A{4,6}|T{4,6})", "Bent_DNA"),
    # Quadruplex-triplex hybrid
    ("Quadruplex-Triplex Hybrid", r"(G{3,}[ATGC]{1,12}){3}G{3,}[ATGC]{0,100}([AG]{10,}|[CT]{10,})", "Quadruplex-Triplex_Hybrid"),
    # Cruciform-Triplex junction
    ("Cruciform-Triplex Junction", r"([ATGC]{10,})([ATGC]{0,100})([ATGC]{10,})([ATGC]{0,100})([AG]{10,}|[CT]{10,})", "Cruciform-Triplex_Junctions"),
    # G4/i-motif hybrid (G4 + i-motif in proximity, up to 100nt apart)
    ("G-Quadruplex_i-Motif_Hybrid", r"(G{3,}[ATGC]{1,12}){3}G{3,}[ATGC]{0,100}(C{3,}[ATGC]{1,12}){3}C{3,}", "G-Quadruplex_i-Motif_Hybrid"),
]

EXAMPLE_FASTA = """>Example
ATCGATCGATCGAAAATTTTATTTAAATTTAAATTTGGGTTAGGGTTAGGGTTAGGGCCCCCTCCCCCTCCCCCTCCCC
ATCGATCGCGCGCGCGATCGCACACACACAGCTGCTGCTGCTTGGGAAAGGGGAAGGGTTAGGGAAAGGGGTTT
GGGTTTAGGGGGGAGGGGCTGCTGCTGCATGCGGGAAGGGAGGGTAGAGGGTCCGGTAGGAACCCCTAACCCCTAA
GAAAGAAGAAGAAGAAGAAGAAAGGAAGGAAGGAGGAGGAGGAGGAGGAGGAGGAGGAGGAGGG
"""

def parse_fasta(fasta_str: str) -> str:
    lines = fasta_str.strip().splitlines()
    seq = [line.strip() for line in lines if not line.startswith(">")]
    return "".join(seq).upper().replace(" ", "")

def gc_content(seq: str) -> float:
    seq = seq.upper()
    return 100.0 * (seq.count("G") + seq.count("C")) / max(1, len(seq))

def wrap(seq: str, width=60) -> str:
    return "\n".join([seq[i:i+width] for i in range(0, len(seq), width)])

def g4hunter_score(seq: str) -> float:
    vals = []
    seq = seq.upper()
    i = 0
    while i < len(seq):
        if seq[i] == 'G':
            run_len = 1
            while i + run_len < len(seq) and seq[i + run_len] == 'G':
                run_len += 1
            score = min(run_len, 4)
            vals.extend([score]*run_len)
            i += run_len
        elif seq[i] == 'C':
            run_len = 1
            while i + run_len < len(seq) and seq[i + run_len] == 'C':
                run_len += 1
            score = -min(run_len, 4)
            vals.extend([score]*run_len)
            i += run_len
        else:
            vals.append(0)
            i += 1
    return np.mean(vals) if vals else 0.0

def motif_propensity(name: str, seq: str) -> str:
    if name == "G-Quadruplex":
        return f"{g4hunter_score(seq):.2f}"
    if name == "i-Motif":
        seq = seq.replace("G", "C")
        return f"{-g4hunter_score(seq):.2f}"
    if name == "G-Triplex":
        # Chen et al. NAR 2018: Triplex is less stable than G4 (~0.7x)
        score = g4hunter_score(seq)
        return f"{score * 0.7:.2f}"
    if name == "Z-DNA":
        n = len(re.findall(r"(GC|CG|GT|TG|AC|CA)", seq))
        return f"{n:d}" if n > 0 else "NA"
    if name == "Sticky_DNA":
        n = len(re.findall(r"(GAA|TTC)", seq))
        return f"{n:d}" if n > 0 else "NA"
    if name == "H-DNA":
        return "NA"
    if name == "Bipartite_G-Quadruplex":
        blocks = re.findall(r"(G{3,}[ATGC]{1,12}){3}G{3,}", seq)
        if blocks:
            score = np.mean([g4hunter_score(b) for b in blocks])
            return f"{score:.2f}"
        return "NA"
    if name in ["DNA_Hairpin", "G-Hairpin", "C-Hairpin", "i-Motif_Hairpin"]:
        stems = re.findall(r"([ATGC]{6,})", seq)
        if stems:
            return f"{max([len(s) for s in stems])}bp-stem"
        return "NA"
    if name == "Slipped_DNA":
        dr = re.findall(r"([ATGC]{10,25})", seq)
        return f"{len(dr)}" if dr else "NA"
    if name == "Cruciform_DNA":
        ir = re.findall(r"([ATGC]{10,})", seq)
        return f"{max([len(i) for i in ir])}bp-arm" if ir else "NA"
    if name == "Bent_DNA":
        tracts = re.findall(r"A{4,6}|T{4,6}", seq)
        return f"{max([len(t) for t in tracts])}bp-tract" if tracts else "NA"
    # For other/hybrid/multi-conformational cases
    return "NA"

def find_motifs(seq: str) -> list:
    results = []
    for motif_class, regex, name in MOTIFS:
        for m in re.finditer(regex, seq):
            region = seq[m.start():m.end()]
            results.append({
                "Motif Class": motif_class,
                "Subtype": name,
                "Start": m.start() + 1,
                "End": m.end(),
                "Length": len(region),
                "GC (%)": f"{gc_content(region):.1f}",
                "Propensity/Score": motif_propensity(name, region),
                "Sequence": wrap(region, 60),
            })
    return results

def find_multiconformational(results: list, max_gap=10) -> list:
    df = pd.DataFrame(results)
    df = df.sort_values("Start")
    mcrs = []
    for i in range(len(df)-1):
        curr, nxt = df.iloc[i], df.iloc[i+1]
        if nxt["Start"] - curr["End"] <= max_gap and curr["Subtype"] != nxt["Subtype"]:
            mcr_seq = curr["Sequence"].replace("\n", "") + nxt["Sequence"].replace("\n", "")
            mcrs.append({
                "Motif Class": "Multi-Conformational",
                "Subtype": f"{curr['Subtype']}/{nxt['Subtype']}",
                "Start": curr["Start"],
                "End": nxt["End"],
                "Length": nxt["End"] - curr["Start"] + 1,
                "GC (%)": f"{gc_content(mcr_seq):.1f}",
                "Propensity/Score": "NA",
                "Sequence": wrap(mcr_seq, 60)
            })
    return mcrs

def visualize_motifs(results: list, seq_len: int):
    plt.figure(figsize=(15, 6))
    motif_types = sorted(set(r['Motif Class'] for r in results))
    class2y = {cl: i+1 for i, cl in enumerate(motif_types)}
    colormap = plt.colormaps['tab20']
    for r in results:
        y = class2y[r['Motif Class']]
        plt.plot([r['Start'], r['End']], [y, y], lw=10,
                 color=colormap((y-1) % 20),
                 label=r['Motif Class'] if r['Start']==min(rr['Start'] for rr in results if rr['Motif Class']==r['Motif Class']) else "")
    plt.yticks(list(class2y.values()), list(class2y.keys()))
    plt.xlabel("Sequence Position (bp)")
    plt.ylabel("Motif Class")
    plt.title("Non-B DNA Motif Locations")
    plt.tight_layout()
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    st.pyplot(plt)

def csv_download_button(df, label="Download CSV"):
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode('utf-8'),
        file_name="motifs.csv",
        mime="text/csv"
    )

def excel_download_button(df, label="Download Excel"):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    output.seek(0)
    st.download_button(
        label=label,
        data=output,
        file_name="motifs.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.title("🧬 Non-B DNA Motif Finder")
st.write(
    "Upload a FASTA file or paste your sequence below. "
    "Supports single or multiline FASTA. Example sequence provided."
)

col1, col2 = st.columns(2)
with col1:
    fasta_file = st.file_uploader("Upload FASTA file", type=["fa", "fasta", "txt"])
with col2:
    if st.button("Use Example Sequence"):
        st.session_state["input_seq"] = EXAMPLE_FASTA
    input_seq = st.text_area("Paste sequence in FASTA format", st.session_state.get("input_seq", EXAMPLE_FASTA), height=150)

seq = None
if fasta_file is not None:
    seq = parse_fasta(fasta_file.read().decode("utf-8"))
elif input_seq.strip():
    seq = parse_fasta(input_seq.strip())
else:
    st.stop()

if not seq or not re.match("^[ATGCUatgcu]+$", seq):
    st.error("No valid DNA sequence detected. Please upload or paste a valid FASTA.")
    st.stop()

st.markdown(f"**Sequence length:** {len(seq):,} bp")

results = find_motifs(seq)
multi_conf = find_multiconformational(results)
all_results = results + multi_conf

if not results:
    st.warning("No non-B DNA motifs detected in this sequence.")
    st.stop()

df = pd.DataFrame(all_results)
st.markdown("### 🧬 Predicted Non-B DNA Motifs")
st.dataframe(df, use_container_width=True, hide_index=True)

with st.expander("Motif Class Summary", expanded=True):
    motif_counts = df["Motif Class"].value_counts().reset_index()
    motif_counts.columns = ["Motif Class", "Count"]
    st.dataframe(motif_counts, use_container_width=True, hide_index=True)

col_csv, col_excel = st.columns(2)
with col_csv:
    csv_download_button(df, "Download Results as CSV")
with col_excel:
    excel_download_button(df, "Download Results as Excel")

st.markdown("### 📊 Motif Visualization")
visualize_motifs(results, len(seq))
