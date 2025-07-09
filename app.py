# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ---------- Sequence Parsing Utilities -----------
def parse_fasta(text: str) -> str:
    lines = text.strip().splitlines()
    seq = ''.join(line.strip() for line in lines if not line.startswith(">"))
    seq = seq.replace(" ", "").replace("U", "T").upper()
    return re.sub(r'[^ATGC]', '', seq)

EXAMPLE_FASTA = """>Example
ATCGATCGATCGAAAATTTTATTTAAATTTAAATTTGGGTTAGGGTTAGGGTTAGGGCCCCCTCCCCCTCCCCCTCCCC
ATCGATCGCGCGCGCGATCGCACACACACAGCTGCTGCTGCTTGGGAAAGGGGAAGGGTTAGGGAAAGGGGTTT
GGGTTTAGGGGGGAGGGGCTGCTGCTGCATGCGGGAAGGGAGGGTAGAGGGTCCGGTAGGAACCCCTAACCCCTAA
"""

# ---------- Motif Functions -----------
def find_gquadruplex(seq):
    # G4: (G{3,}[ATGC]{1,7}){3}G{3,}
    pattern = r'(G{3,}[ATGC]{1,7}){3}G{3,}'
    return [
        ("Quadruplex", "G-Quadruplex", m.start()+1, m.end(), seq[m.start():m.end()])
        for m in re.finditer(pattern, seq)
    ]

def find_imotif(seq):
    pattern = r'(C{3,}[ATGC]{1,7}){3}C{3,}'
    return [
        ("Quadruplex", "i-Motif", m.start()+1, m.end(), seq[m.start():m.end()])
        for m in re.finditer(pattern, seq)
    ]

def find_gtriplex(seq, g4_regions):
    # Only call G-triplex if not a G4 region
    pattern = r'(G{3,}[ATGC]{1,7}){2}G{3,}'
    motifs = []
    g4_spans = [(s-1, e) for _,_,s,e,_ in g4_regions]
    for m in re.finditer(pattern, seq):
        start, end = m.start(), m.end()
        # If overlaps with any G4, skip
        if any(max(start, gs) < min(end, ge) for gs, ge in g4_spans):
            continue
        motifs.append(("Quadruplex", "G-Triplex", start+1, end, seq[start:end]))
    return motifs

def find_bipartite_g4(seq):
    # Two G4s separated by ≤100nt
    pattern = r'((G{3,}[ATGC]{1,7}){3}G{3,})([ATGC]{0,100})((G{3,}[ATGC]{1,7}){3}G{3,})'
    return [
        ("Quadruplex", "Bipartite_G-Quadruplex", m.start()+1, m.end(), seq[m.start():m.end()])
        for m in re.finditer(pattern, seq)
    ]

def find_zdna(seq):
    pattern = r'((GC|CG|GT|TG|AC|CA){6,})'
    return [
        ("Z-DNA", "Z-DNA", m.start()+1, m.end(), seq[m.start():m.end()])
        for m in re.finditer(pattern, seq)
    ]

def find_direct_repeat(seq):
    pattern = r'([ATGC]{10,300})([ATGC]{0,100})\1'
    return [
        ("Direct Repeat", "Slipped_DNA", m.start()+1, m.end(), seq[m.start():m.end()])
        for m in re.finditer(pattern, seq)
    ]

def find_inverted_repeat(seq):
    pattern = r'([ATGC]{6,})([ATGC]{0,100})\1'
    return [
        ("Inverted Repeat", "Cruciform_DNA", m.start()+1, m.end(), seq[m.start():m.end()])
        for m in re.finditer(pattern, seq)
    ]

def find_bent_dna(seq):
    # Three A-tracts (A{3,11}) spaced by 3-11 nt
    pattern = r'(A{3,11}[ATGC]{3,11}){2}A{3,11}'
    return [
        ("Bent DNA", "APR/Bent_DNA", m.start()+1, m.end(), seq[m.start():m.end()])
        for m in re.finditer(pattern, seq)
    ]

def find_triplex(seq):
    pattern = r'([AG]{10,}|[CT]{10,})([ATGC]{0,8})([AG]{10,}|[CT]{10,})'
    return [
        ("Triplex", "H-DNA", m.start()+1, m.end(), seq[m.start():m.end()])
        for m in re.finditer(pattern, seq)
    ]

def find_cruciform_triplex_junction(seq):
    pattern = r'([ATGC]{10,})([ATGC]{0,100})([ATGC]{10,})([ATGC]{0,100})([AG]{10,}|[CT]{10,})'
    return [
        ("Cruciform-Triplex Junction", "Cruciform-Triplex_Junctions", m.start()+1, m.end(), seq[m.start():m.end()])
        for m in re.finditer(pattern, seq)
    ]

def find_g4_imotif_hybrid(seq):
    pattern = r'(G{3,}[ATGC]{1,7}){3}G{3,}[ATGC]{0,100}(C{3,}[ATGC]{1,7}){3}C{3,}'
    return [
        ("G4/i-Motif Hybrid", "G-Quadruplex_i-Motif_Hybrid", m.start()+1, m.end(), seq[m.start():m.end()])
        for m in re.finditer(pattern, seq)
    ]

def find_local_bends(seq):
    motifs = []
    for pattern, name in [(r'A{6,7}', "A-Tract Local Bend"), (r'T{6,7}', "T-Tract Local Bend"),
                          (r'(CA){4,}', "CA Dinucleotide Bend"), (r'(TG){4,}', "TG Dinucleotide Bend")]:
        for m in re.finditer(pattern, seq):
            motifs.append(("Local Bend", name, m.start()+1, m.end(), seq[m.start():m.end()]))
    return motifs

# ---------- Scoring Systems -----------
def gc_content(seq):
    return 100.0 * (seq.count("G") + seq.count("C")) / max(1, len(seq))

def g4hunter_score(seq):
    vals = []
    i = 0
    n = len(seq)
    while i < n:
        if seq[i] == 'G':
            run_len = 1
            while i+run_len < n and seq[i+run_len]=='G':
                run_len += 1
            score = min(run_len, 4)
            vals += [score]*run_len
            i += run_len
        elif seq[i] == 'C':
            run_len = 1
            while i+run_len < n and seq[i+run_len]=='C':
                run_len += 1
            score = -min(run_len, 4)
            vals += [score]*run_len
            i += run_len
        else:
            vals.append(0)
            i += 1
    return np.mean(vals) if vals else 0

def motif_score(motif, seq):
    name = motif[1]
    if name == "G-Quadruplex":
        return f"{g4hunter_score(seq):.2f}"
    if name == "i-Motif":
        return f"{-g4hunter_score(seq.replace('G','C')):.2f}"
    if name == "G-Triplex":
        return f"{g4hunter_score(seq)*0.7:.2f}"
    return "NA"

# ---------- Motif Aggregator -----------
def find_motifs(seq):
    g4 = find_gquadruplex(seq)
    imotif = find_imotif(seq)
    bipartite = find_bipartite_g4(seq)
    gtriplex = find_gtriplex(seq, g4)
    zdna = find_zdna(seq)
    directr = find_direct_repeat(seq)
    invertr = find_inverted_repeat(seq)
    bent = find_bent_dna(seq)
    triplex = find_triplex(seq)
    cruci_triplex = find_cruciform_triplex_junction(seq)
    g4imotif = find_g4_imotif_hybrid(seq)
    local_bends = find_local_bends(seq)
    motifs = (g4 + imotif + bipartite + gtriplex + zdna + directr + invertr +
              bent + triplex + cruci_triplex + g4imotif + local_bends)
    # Remove overlapping, keep the longest
    motifs = sorted(motifs, key=lambda x: (x[2], -len(x[4])))
    nonoverlap = []
    covered = set()
    for m in motifs:
        rng = set(range(m[2], m[3]+1))
        if covered.isdisjoint(rng):
            nonoverlap.append(m)
            covered.update(rng)
    return nonoverlap

def wrap(seq, width=60):
    return "\n".join(seq[i:i+width] for i in range(0, len(seq), width))

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Non-B DNA Motif Finder", layout="wide")

pages = ["Home", "Upload & Analyze", "Results", "Visualization", "Download Report", "About"]
page = st.sidebar.radio("Navigation", pages)

if page == "Home":
    st.title("Non-B DNA Motif Finder")
    # Try to display image if present
    if os.path.exists("nbd.PNG"):
        st.image("nbd.PNG", use_container_width=True)
    else:
        st.warning("Header image (nbd.PNG) not found in app folder.")
    st.markdown("""
    Detects and visualizes G-quadruplexes, G-triplex, Z-DNA, cruciforms, APR/Bent DNA, direct repeats, and more.
    *Upload or paste FASTA sequence, analyze motifs, visualize, and export results.*

    **Motifs included**: G-Quadruplex, G-Triplex, i-Motif, Bipartite-G4, Z-DNA, Slipped Direct Repeat, Cruciform, H-DNA (Triplex), Bent DNA/APR, Cruciform-Triplex Junctions, G4/i-Motif Hybrid, Local Bends.
    """)

elif page == "Upload & Analyze":
    st.header("Upload or Paste Sequence")
    col1, col2 = st.columns([1,1])
    with col1:
        fasta_file = st.file_uploader("Upload FASTA file", type=["fa", "fasta", "txt"])
        if fasta_file:
            seq = parse_fasta(fasta_file.read().decode("utf-8"))
            st.session_state['seq'] = seq
            st.success("FASTA file loaded.")
    with col2:
        if st.button("Use Example Sequence"):
            st.session_state['seq'] = parse_fasta(EXAMPLE_FASTA)
        seq_input = st.text_area("Paste sequence (FASTA or raw)", value=st.session_state.get('seq', ""), height=120)
        if seq_input:
            seq = parse_fasta(seq_input)
            st.session_state['seq'] = seq
    if st.button("Run Analysis"):
        seq = st.session_state.get('seq', "")
        if not seq or not re.match("^[ATGC]+$", seq):
            st.error("Please upload or paste a valid DNA sequence (A/T/G/C only).")
        else:
            with st.spinner("Analyzing..."):
                results = find_motifs(seq)
                df = pd.DataFrame([{
                    "Motif Class": m[0],
                    "Subtype": m[1],
                    "Start": m[2],
                    "End": m[3],
                    "Length": m[3]-m[2]+1,
                    "GC (%)": f"{gc_content(m[4]):.1f}",
                    "Propensity/Score": motif_score(m, m[4]),
                    "Sequence": wrap(m[4], 60)
                } for m in results])
                st.session_state['df'] = df
            st.success(f"Detected {len(df)} motifs.")

elif page == "Results":
    st.header("Motif Detection Results")
    df = st.session_state.get('df', pd.DataFrame())
    if df.empty:
        st.info("No results. Go to 'Upload & Analyze' and run analysis.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.markdown(f"**Sequence length:** {len(st.session_state['seq']):,} bp")
        with st.expander("Motif Type Summary"):
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

        fig, ax = plt.subplots(figsize=(12, len(motif_types)*0.5+3))
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

        st.subheader("Motif Type Distribution")
        counts = df['Subtype'].value_counts()
        fig2, ax2 = plt.subplots()
        ax2.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
        ax2.axis('equal')
        st.pyplot(fig2)

elif page == "Download Report":
    st.header("Download Motif Report")
    df = st.session_state.get('df', pd.DataFrame())
    if df.empty:
        st.info("No results. Run analysis first.")
    else:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name=f"motif_results_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv",
            mime="text/csv"
        )
        output = pd.ExcelWriter("motif_results.xlsx", engine='xlsxwriter')
        df.to_excel(output, index=False)
        output.close()
        with open("motif_results.xlsx", "rb") as f:
            st.download_button(
                label="Download Results as Excel",
                data=f.read(),
                file_name=f"motif_results_{datetime.now().strftime('%Y%m%d-%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        os.remove("motif_results.xlsx")

elif page == "About":
    st.header("About")
    st.markdown("""
    **Non-B DNA Motif Finder**  
    Fast detection of major non-B DNA structural motifs using regular expressions and canonical scoring rules.

    - G-quadruplex and derivatives: with G4Hunter-like scoring  
    - Bent DNA/APR using three A-tracts (as in nbst and literature)
    - Direct, inverted, triplex, and hybrid motifs  
    - No STRs (tandem repeats) included as per your request
    - Results exportable as CSV/Excel, and interactive visualization
    """)
