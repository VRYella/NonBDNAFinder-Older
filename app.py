# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# --- Logo Handling ---
def show_logo():
    try:
        st.image("nbd.png", use_column_width=True)
    except Exception:
        st.warning("Logo image not found. Please add nbd.png to your app directory.")

# --- FASTA Parsing ---
def parse_fasta(fasta_str: str) -> str:
    lines = fasta_str.strip().splitlines()
    seq = [line.strip() for line in lines if not line.startswith(">")]
    return "".join(seq).upper().replace(" ", "").replace("U", "T")

# --- GC Content ---
def gc_content(seq: str) -> float:
    seq = seq.upper()
    gc = seq.count("G") + seq.count("C")
    return 100.0 * gc / max(1, len(seq))

def wrap(seq: str, width=60) -> str:
    return "\n".join([seq[i:i+width] for i in range(0, len(seq), width)])

# --- Motif Definitions ---
MOTIF_INFO = [
    ("G-Quadruplex", "G-rich, four-stranded DNA, four runs of ≥3 Gs, loops 1–12 nt. [Bedrat 2016]"),
    ("i-Motif", "C-rich quadruplex, forms at low pH. [Abou Assi 2018]"),
    ("Bipartite_G-Quadruplex", "Two G4s ≤100 nt apart. [Matsugami 2001]"),
    ("G-Triplex", "Three G runs, triple-stranded. [Chen NAR 2018]"),
    ("Multimeric_G-Quadruplex", "Several tandem G4s, ≤50 nt apart."),
    ("Z-DNA", "Left-handed helix, alternating purine-pyrimidine. [Herbert 1999]"),
    ("H-DNA", "Triplex-forming mirror repeats. [Buske 2012]"),
    ("Sticky_DNA", "GAA/TTC repeats ≥5 units. [Sakamoto 1999]"),
    ("Slipped_DNA", "Direct repeats, unit 10–300 bp, ≤100 bp spacer. [Wells 1988]"),
    ("Cruciform_DNA", "Inverted repeats, arms ≥6 bp, ≤100 bp loop. [Pearson 1996]"),
    ("Bent_DNA", "APR: at least 3 A-tracts (3–11 bp) with 10–11 nt periodicity. [Marini 1982]"),
    ("STR", "Short tandem repeat: unit 1–9 bp, total length ≥10 bp."),
    ("Quadruplex-Triplex_Hybrid", "G4 and triplex potential overlap. [Siddiqui-Jain 2002]"),
    ("Cruciform-Triplex_Junctions", "Junction: cruciform+triplex motifs. [Cer 2011]"),
    ("G-Quadruplex_i-Motif_Hybrid", "G4 and i-motif nearby. [Abou Assi 2018]"),
    ("Local Flexible Region", "CA or TG dinucleotide repeats (≥4)"),
    ("Local Curved Motif", "A6,7 or T6,7 tracts"),
]

# --- Motif Regexes grouped by type ---
def gquadruplex_motifs(seq):
    motifs = []
    # G4 canonical
    for m in re.finditer(r"(G{3,}[ATGC]{1,12}){3}G{3,}", seq):
        motifs.append(("Quadruplex", "G-Quadruplex", m.start()+1, m.end(), m.group()))
    # Bipartite G4
    for m in re.finditer(r"(G{3,}[ATGC]{1,12}){3}G{3,}[ATGC]{0,100}(G{3,}[ATGC]{1,12}){3}G{3,}", seq):
        motifs.append(("Quadruplex", "Bipartite_G-Quadruplex", m.start()+1, m.end(), m.group()))
    # G-Triplex
    for m in re.finditer(r"(G{3,}[ATGC]{1,12}){2}G{3,}", seq):
        motifs.append(("Quadruplex", "G-Triplex", m.start()+1, m.end(), m.group()))
    # Multimeric G4
    for m in re.finditer(r"((G{3,}[ATGC]{1,12}){3}G{3,}([ATGC]{1,50}(G{3,}[ATGC]{1,12}){3}G{3,})+)", seq):
        motifs.append(("Quadruplex", "Multimeric_G-Quadruplex", m.start()+1, m.end(), m.group()))
    return motifs

def i_motif_motifs(seq):
    motifs = []
    for m in re.finditer(r"(C{3,}[ATGC]{1,12}){3}C{3,}", seq):
        motifs.append(("Quadruplex", "i-Motif", m.start()+1, m.end(), m.group()))
    return motifs

def z_dna_motifs(seq):
    motifs = []
    for m in re.finditer(r"((GC|CG|GT|TG|AC|CA){6,})", seq):
        motifs.append(("Z-DNA", "Z-DNA", m.start()+1, m.end(), m.group()))
    return motifs

def triplex_motifs(seq):
    motifs = []
    # H-DNA: Mirror repeats (arms ≥10 bp, loop ≤8 bp)
    for m in re.finditer(r"([AG]{10,}|[CT]{10,})([ATGC]{0,8})([AG]{10,}|[CT]{10,})", seq):
        motifs.append(("Triplex", "H-DNA", m.start()+1, m.end(), m.group()))
    # Sticky DNA (GAA/TTC) ≥5
    for m in re.finditer(r"(GAA){5,}|(TTC){5,}", seq):
        motifs.append(("Triplex", "Sticky_DNA", m.start()+1, m.end(), m.group()))
    return motifs

def repeat_motifs(seq):
    motifs = []
    # Direct repeat (unit 10–300 bp, ≤100 bp spacer)
    for m in re.finditer(r"([ATGC]{10,300})([ATGC]{0,100})\1", seq):
        motifs.append(("Direct Repeat", "Slipped_DNA", m.start()+1, m.end(), m.group()))
    # Inverted repeat (arms ≥6 bp, ≤100 bp loop)
    for m in re.finditer(r"([ATGC]{6,})([ATGC]{0,100})\1", seq):
        motifs.append(("Inverted Repeat", "Cruciform_DNA", m.start()+1, m.end(), m.group()))
    return motifs

def str_motifs(seq):
    motifs = []
    # STRs: repeat units 1–9 bp, total length ≥10 bp
    for unit_len in range(1, 10):
        pattern = fr"((?:[ATGC]{{{unit_len}}})\1{{1,}})"
        for m in re.finditer(pattern, seq):
            if len(m.group()) >= 10:
                motifs.append(("STR", f"STR_{unit_len}bp", m.start()+1, m.end(), m.group()))
    return motifs

def apr_motifs(seq):
    motifs = []
    # APR (Bent DNA): at least 3 A-tracts (A{3,11}), ~10–11 nt spacing
    tract_positions = [m.start() for m in re.finditer(r"A{3,11}", seq)]
    for i in range(len(tract_positions) - 2):
        d1 = tract_positions[i+1] - tract_positions[i]
        d2 = tract_positions[i+2] - tract_positions[i+1]
        if 9 <= d1 <= 12 and 9 <= d2 <= 12:
            motifs.append(("Bent DNA", "APR/Bent_DNA", tract_positions[i]+1, tract_positions[i+2]+11, seq[tract_positions[i]:tract_positions[i+2]+11]))
    return motifs

def local_flex_motifs(seq):
    motifs = []
    # CA or TG dinucleotide ≥4 repeats
    for m in re.finditer(r"(CA){4,}", seq):
        motifs.append(("Local Flexible Region", "CA Flexible", m.start()+1, m.end(), m.group()))
    for m in re.finditer(r"(TG){4,}", seq):
        motifs.append(("Local Flexible Region", "TG Flexible", m.start()+1, m.end(), m.group()))
    return motifs

def local_curve_motifs(seq):
    motifs = []
    # A6,7 or T6,7
    for m in re.finditer(r"A{6,7}", seq):
        motifs.append(("Local Curved Motif", "A-Tract Curve", m.start()+1, m.end(), m.group()))
    for m in re.finditer(r"T{6,7}", seq):
        motifs.append(("Local Curved Motif", "T-Tract Curve", m.start()+1, m.end(), m.group()))
    return motifs

def hybrid_motifs(seq):
    motifs = []
    # G4 + triplex overlap (hybrid)
    for m in re.finditer(r"(G{3,}[ATGC]{1,12}){3}G{3,}[ATGC]{0,100}([AG]{10,}|[CT]{10,})", seq):
        motifs.append(("Quadruplex-Triplex Hybrid", "Quadruplex-Triplex_Hybrid", m.start()+1, m.end(), m.group()))
    # Cruciform-triplex
    for m in re.finditer(r"([ATGC]{6,})([ATGC]{0,100})([ATGC]{6,})([ATGC]{0,100})([AG]{10,}|[CT]{10,})", seq):
        motifs.append(("Cruciform-Triplex Junction", "Cruciform-Triplex_Junctions", m.start()+1, m.end(), m.group()))
    # G4+i-motif hybrid
    for m in re.finditer(r"(G{3,}[ATGC]{1,12}){3}G{3,}[ATGC]{0,100}(C{3,}[ATGC]{1,12}){3}C{3,}", seq):
        motifs.append(("G-Quadruplex_i-Motif_Hybrid", "G-Quadruplex_i-Motif_Hybrid", m.start()+1, m.end(), m.group()))
    return motifs

# --- Motif Score/Propensity (motif-specific logic) ---
def g4hunter_score(seq):
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
    return np.mean(vals) if vals else 0.0

def motif_propensity(motif_class, subtype, seq):
    if subtype == "G-Quadruplex":
        return f"{g4hunter_score(seq):.2f}"
    if subtype == "i-Motif":
        seq = seq.replace("G", "C")
        return f"{-g4hunter_score(seq):.2f}"
    if subtype == "G-Triplex":
        return f"{g4hunter_score(seq)*0.7:.2f}"
    if subtype == "Multimeric_G-Quadruplex":
        g4s = re.findall(r"(G{3,}[ATGC]{1,12}){3}G{3,}", seq)
        if g4s:
            return f"{np.mean([g4hunter_score(g) for g in g4s]):.2f}"
        return "NA"
    if motif_class == "Z-DNA":
        n = len(re.findall(r"(GC|CG|GT|TG|AC|CA)", seq))
        return f"{n}" if n > 0 else "NA"
    if subtype == "Sticky_DNA":
        n = len(re.findall(r"(GAA|TTC)", seq))
        return f"{n}" if n > 0 else "NA"
    if subtype == "Cruciform_DNA":
        ir = re.findall(r"([ATGC]{6,})", seq)
        return f"{max([len(i) for i in ir])}bp-arm" if ir else "NA"
    if subtype == "Slipped_DNA":
        dr = re.findall(r"([ATGC]{10,300})", seq)
        return f"{len(dr)}" if dr else "NA"
    if subtype == "Bent_DNA" or subtype == "APR/Bent_DNA":
        tracts = re.findall(r"A{3,11}", seq)
        return f"{max([len(t) for t in tracts])}bp-tract" if tracts else "NA"
    if motif_class == "STR":
        return f"{len(seq)}bp"
    return "NA"

# --- All Motif Search ---
def find_motifs(seq):
    seq = seq.upper()
    found = []
    for m in (
        gquadruplex_motifs(seq)
        + i_motif_motifs(seq)
        + z_dna_motifs(seq)
        + triplex_motifs(seq)
        + repeat_motifs(seq)
        + str_motifs(seq)
        + apr_motifs(seq)
        + local_flex_motifs(seq)
        + local_curve_motifs(seq)
        + hybrid_motifs(seq)
    ):
        motif_class, subtype, start, end, region = m
        found.append({
            "Motif Class": motif_class,
            "Subtype": subtype,
            "Start": start,
            "End": end,
            "Length": end - start + 1,
            "GC (%)": f"{gc_content(region):.1f}",
            "Propensity/Score": motif_propensity(motif_class, subtype, region),
            "Sequence": wrap(region, 60),
        })
    return found

# --- Example Sequence (multiline FASTA) ---
EXAMPLE_FASTA = """>Example
ATCGATCGATCGAAAATTTTATTTAAATTTAAATTTGGGTTAGGGTTAGGGTTAGGGCCCCCTCCCCCTCCCCCTCCCC
ATCGATCGCGCGCGCGATCGCACACACACAGCTGCTGCTGCTTGGGAAAGGGGAAGGGTTAGGGAAAGGGGTTT
GGGTTTAGGGGGGAGGGGCTGCTGCTGCATGCGGGAAGGGAGGGTAGAGGGTCCGGTAGGAACCCCTAACCCCTAA
GAAAGAAGAAGAAGAAGAAGAAAGGAAGGAAGGAGGAGGAGGAGGAGGAGGAGGAGGAGGAGGAGGAGGAGGG
"""

# --- Streamlit Multi-Page Layout ---
st.set_page_config(page_title="Non-B DNA Motif Finder", layout="wide")
pages = ["Home", "Upload & Analyze", "Results", "Visualization", "Download", "About"]
page = st.sidebar.radio("Navigation", pages)

if 'seq' not in st.session_state:
    st.session_state['seq'] = ""
if 'df' not in st.session_state:
    st.session_state['df'] = pd.DataFrame()

if page == "Home":
    st.title("Non-B DNA Motif Finder (FAST)")
    show_logo()
    st.markdown("**Rapid detection and visualization of non-canonical (non-B) DNA motifs.**")
    st.header("Motif Definitions")
    for name, expl in MOTIF_INFO:
        st.markdown(f"- **{name}**: {expl}")

elif page == "Upload & Analyze":
    st.header("Upload or Paste Sequence (FASTA or raw)")
    col1, col2 = st.columns([1, 1])
    with col1:
        fasta_file = st.file_uploader("Upload FASTA file", type=["fa", "fasta", "txt"])
        if fasta_file:
            try:
                seq = parse_fasta(fasta_file.read().decode("utf-8"))
                st.session_state['seq'] = seq
                st.success("FASTA file loaded!")
            except Exception:
                st.error("Could not parse file as FASTA.")
    with col2:
        if st.button("Use Example Sequence"):
            st.session_state['seq'] = parse_fasta(EXAMPLE_FASTA)
        seq_input = st.text_area("Paste sequence (FASTA or raw)", value=st.session_state.get('seq', ""), height=120)
        if seq_input:
            try:
                seq = parse_fasta(seq_input)
                st.session_state['seq'] = seq
            except Exception:
                st.error("Paste a valid FASTA or sequence.")

    if st.button("Run Analysis"):
        seq = st.session_state.get('seq', "")
        if not seq or not re.match("^[ATGC]+$", seq):
            st.error("Please upload or paste a valid DNA sequence (A/T/G/C only).")
        else:
            with st.spinner("Analyzing sequence..."):
                results = find_motifs(seq)
                st.session_state['df'] = pd.DataFrame(results)
            if st.session_state['df'].empty:
                st.warning("No non-B DNA motifs detected in this sequence.")
            else:
                st.success(f"Detected {len(st.session_state['df'])} motif regions in {len(seq):,} bp.")

elif page == "Results":
    st.header("Motif Detection Results")
    df = st.session_state.get('df', pd.DataFrame())
    if df.empty:
        st.info("No results yet. Go to 'Upload & Analyze' and run analysis.")
    else:
        st.markdown(f"**Sequence length:** {len(st.session_state['seq']):,} bp")
        st.dataframe(df, use_container_width=True, hide_index=True)
        with st.expander("Motif Class Summary"):
            motif_counts = df["Subtype"].value_counts().reset_index()
            motif_counts.columns = ["Motif Type", "Count"]
            st.dataframe(motif_counts, use_container_width=True, hide_index=True)

elif page == "Visualization":
    st.header("Motif Visualization")
    df = st.session_state.get('df', pd.DataFrame())
    seq = st.session_state.get('seq', "")
    if df.empty:
        st.info("No results to visualize. Run analysis first.")
    else:
        st.subheader("Motif Map (Full Sequence)")
        motif_types = sorted(df['Subtype'].unique())
        color_palette = sns.color_palette('husl', n_colors=len(motif_types))
        color_map = {typ: color_palette[i] for i, typ in enumerate(motif_types)}
        y_map = {typ: i+1 for i, typ in enumerate(motif_types)}

        fig, ax = plt.subplots(figsize=(10, len(motif_types)*0.7+2))
        for _, motif in df.iterrows():
            motif_type = motif['Subtype']
            y = y_map[motif_type]
            color = color_map[motif_type]
            ax.hlines(y, motif['Start'], motif['End'], color=color, linewidth=8)
        ax.set_yticks(list(y_map.values()))
        ax.set_yticklabels(list(y_map.keys()))
        ax.set_xlim(0, len(seq)+1)
        ax.set_xlabel('Position on Sequence (bp)')
        ax.set_title('Motif Map (Full Sequence)')
        sns.despine(left=False, bottom=False)
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Motif Type Distribution (Pie Chart)")
        counts = df['Subtype'].value_counts()
        fig2, ax2 = plt.subplots()
        ax2.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
        ax2.axis('equal')
        st.pyplot(fig2)

        st.subheader("Motif Counts (Bar Chart)")
        fig3, ax3 = plt.subplots()
        counts.plot.bar(ax=ax3)
        ax3.set_ylabel("Count")
        ax3.set_xlabel("Motif Type")
        plt.tight_layout()
        st.pyplot(fig3)

elif page == "Download":
    st.header("Download Motif Report")
    df = st.session_state.get('df', pd.DataFrame())
    if df.empty:
        st.info("No results to download. Run analysis first.")
    else:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name=f"motif_results_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv",
            mime="text/csv"
        )
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)
        st.download_button(
            label="Download Results as Excel",
            data=output.getvalue(),
            file_name=f"motif_results_{datetime.now().strftime('%Y%m%d-%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

elif page == "About":
    st.header("About")
    st.markdown("""
    **Non-B DNA Motif Finder** is a tool for rapid detection and visualization of non-canonical DNA structures ("non-B DNA motifs") in sequences.
    - Supports G-quadruplexes, triplexes, Z-DNA, cruciforms, bent DNA, and more.
    - Accepts FASTA files or direct sequence input.
    - Visualizes results and offers export options.
    - Created for research, education, and bioinformatics.
    """)
    st.markdown("**Developed by: Dr. V.R. Yella & A.S.C. Gummadi**")

