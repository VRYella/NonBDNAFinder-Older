import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import deque

# ----------------- FASTA/INPUT -----------------
EXAMPLE_FASTA = """>Example
AGGGTTTGGGATTTGGGTTTGGGATTTAGGGTTTGGGATTTAGGGA
CCCCCCCCCCCCCCCCCCCC
ATATATATATATATATATATATATATATATAT
TTTTTTTTTTTTTTTTTTT
AATTAATTAA
A"*30
T"*30
"""

def parse_fasta(text):
    lines = text.strip().splitlines()
    seq = [l.strip().upper() for l in lines if not l.startswith(">")]
    seq = ''.join(seq).replace(" ", "").replace("U", "T")
    return re.sub(r'[^ATGC]', '', seq)  # strict DNA

# ----------------- MOTIF SEARCH FUNCTIONS -----------------

def find_gquadruplexes(seq, offset=0):
    # GGG(N1-7)GGG(N1-7)GGG(N1-7)GGG (non-overlapping)
    pattern = re.compile(r'G{3,}(?:[ATGC]{1,7}G{3,}){3,}')
    motifs = []
    for m in pattern.finditer(seq):
        motifs.append(("Quadruplex", "G-Quadruplex", m.start()+1+offset, m.end()+offset, seq[m.start():m.end()]))
    return motifs

def find_imotif(seq, offset=0):
    pattern = re.compile(r'C{3,}(?:[ATGC]{1,7}C{3,}){3,}')
    motifs = []
    for m in pattern.finditer(seq):
        motifs.append(("Quadruplex", "i-Motif", m.start()+1+offset, m.end()+offset, seq[m.start():m.end()]))
    return motifs

def find_gtriplex(seq, g4_regions, offset=0):
    # G-triplex: GGG(N1-7)GGG(N1-7)GGG (not overlapping G4)
    pattern = re.compile(r'G{3,}(?:[ATGC]{1,7}G{3,}){2}')
    motifs = []
    for m in pattern.finditer(seq):
        start, end = m.start()+offset, m.end()+offset
        if not any(s < end and e > start for s, e in g4_regions):
            motifs.append(("Quadruplex", "G-Triplex", start+1, end, seq[m.start():m.end()]))
    return motifs

def find_bipartite_g4(seq, offset=0):
    # Two G4s with 1-100 nt linker
    pattern = re.compile(
        r'(G{3,}(?:[ATGC]{1,7}G{3,}){3,})[ATGC]{1,100}(G{3,}(?:[ATGC]{1,7}G{3,}){3,})'
    )
    motifs = []
    for m in pattern.finditer(seq):
        motifs.append(("Quadruplex", "Bipartite_G-Quadruplex", m.start()+1+offset, m.end()+offset, seq[m.start():m.end()]))
    return motifs

def find_zdna(seq, offset=0):
    pattern = re.compile(r'(?:GC|CG|GT|TG|AC|CA){6,}')
    motifs = []
    for m in pattern.finditer(seq):
        motifs.append(("Z-DNA", "Z-DNA", m.start()+1+offset, m.end()+offset, seq[m.start():m.end()]))
    return motifs

def find_direct_repeat(seq, offset=0):
    pattern = re.compile(r'([ATGC]{10,300})([ATGC]{0,10})\1')
    motifs = []
    for m in pattern.finditer(seq):
        motifs.append(("Direct Repeat", "Slipped_DNA", m.start()+1+offset, m.end()+offset, seq[m.start():m.end()]))
    return motifs

def find_apr(seq, offset=0):
    # A-tract periodic region: at least three A-tracts of length 3–11, spaced 7-15bp center-to-center
    pattern = re.compile(r'(A{3,11}(?:[ATGC]{7,15}A{3,11}){2,})')
    motifs = []
    for m in pattern.finditer(seq):
        motifs.append(("Bent DNA", "APR/Bent_DNA", m.start()+1+offset, m.end()+offset, seq[m.start():m.end()]))
    return motifs

def reverse_complement(seq):
    comp = str.maketrans('ATGC', 'TACG')
    return seq.translate(comp)[::-1]

def find_inverted_repeat(seq, offset=0, min_arm=6, max_loop=100):
    motifs = []
    n = len(seq)
    # Fast algorithm, sliding window with rolling hash for arms (up to 12bp arm)
    for arm_len in range(min_arm, 13):
        arm_dict = dict()
        for i in range(n - 2*arm_len - max_loop + 1):
            left_arm = seq[i:i+arm_len]
            for loop_len in range(0, max_loop+1):
                j = i + arm_len + loop_len
                if j + arm_len > n:
                    break
                right_arm = seq[j:j+arm_len]
                if right_arm == reverse_complement(left_arm):
                    motifs.append(("Inverted Repeat", "Cruciform_DNA",
                                   i+1+offset, j+arm_len+offset, seq[i:j+arm_len]))
    return motifs

def find_triplex(seq, offset=0):
    # Mirror repeat arms 10+, loop <=8, min 10% purine/pyrimidine
    pattern = re.compile(r'([AGCT]{10,})([ATGC]{0,8})\1')
    motifs = []
    for m in pattern.finditer(seq):
        left_arm = m.group(1)
        pur = sum(c in "AG" for c in left_arm)/len(left_arm)
        pyr = sum(c in "CT" for c in left_arm)/len(left_arm)
        if pur >= 0.1 or pyr >= 0.1:
            motifs.append(("Triplex", "H-DNA", m.start()+1+offset, m.end()+offset, seq[m.start():m.end()]))
    return motifs

# --------- WINDOWED PIPELINE FOR LARGE GENOMES ---------
def chunked_find_motifs(seq, window=10000, overlap=100):
    motifs = []
    n = len(seq)
    for start in range(0, n, window - overlap):
        chunk = seq[start:start+window]
        offset = start
        g4s = find_gquadruplexes(chunk, offset)
        g4_regions = [(m[2]-1, m[3]) for m in g4s]
        motifs.extend(g4s)
        motifs.extend(find_imotif(chunk, offset))
        motifs.extend(find_gtriplex(chunk, g4_regions, offset))
        motifs.extend(find_bipartite_g4(chunk, offset))
        motifs.extend(find_zdna(chunk, offset))
        motifs.extend(find_direct_repeat(chunk, offset))
        motifs.extend(find_apr(chunk, offset))
        motifs.extend(find_inverted_repeat(chunk, offset))
        motifs.extend(find_triplex(chunk, offset))
    # Remove perfect duplicates (region-based)
    seen = set()
    unique = []
    for m in motifs:
        key = (m[0], m[1], m[2], m[3])
        if key not in seen:
            unique.append(m)
            seen.add(key)
    return unique

def get_motif_df(seq):
    motifs = chunked_find_motifs(seq)
    data = []
    for m in motifs:
        gc = (m[4].count("G")+m[4].count("C"))/len(m[4])*100 if m[4] else 0
        data.append({
            "Motif Class": m[0],
            "Subtype": m[1],
            "Start": m[2],
            "End": m[3],
            "Length": m[3]-m[2]+1,
            "GC (%)": f"{gc:.1f}",
            "Sequence": m[4][:50]+("..." if len(m[4])>50 else ""),
        })
    return pd.DataFrame(data)

# -------------- STREAMLIT PAGES APP --------------------
st.set_page_config(page_title="Non-B DNA Motif Finder", layout="wide")
st.title("Non-B DNA Motif Finder (Fast, Genome-Ready)")

pages = ["Home", "Upload & Analyze", "Results", "Visualization", "Download"]
page = st.sidebar.radio("Navigation", pages)

if page == "Home":
    st.image("nbd.PNG", use_container_width=True)
    st.markdown("""
    **Detects all major non-B DNA motifs (G4, i-motif, G-triplex, Z-DNA, IR, DR, APR/Bent DNA, Triplex/H-DNA) in FASTA or pasted sequence.**
    - Windowed engine: Suitable for large (Mb) genomic regions
    - Paged interface: upload, results, viz, download
    - No STRs
    """)

elif page == "Upload & Analyze":
    st.header("Upload/Paste Sequence")
    col1, col2 = st.columns(2)
    with col1:
        fasta_file = st.file_uploader("Upload FASTA", type=["fa", "fasta", "txt"])
        if fasta_file:
            seq = parse_fasta(fasta_file.read().decode("utf-8"))
            st.session_state['seq'] = seq
            st.success(f"FASTA loaded ({len(seq):,} nt)")
    with col2:
        if st.button("Use Example Sequence"):
            st.session_state['seq'] = parse_fasta(EXAMPLE_FASTA)
        seq_input = st.text_area("Paste DNA sequence or FASTA", value=st.session_state.get('seq', ''), height=120)
        if seq_input:
            seq = parse_fasta(seq_input)
            st.session_state['seq'] = seq

    if st.button("Run Motif Analysis"):
        seq = st.session_state.get('seq', '')
        if not seq or not re.match("^[ATGC]+$", seq):
            st.error("Valid DNA (A/T/G/C) only.")
        else:
            with st.spinner("Finding motifs (genome-optimized)..."):
                df = get_motif_df(seq)
                st.session_state['df'] = df
            st.success(f"Motif search complete ({len(df)} regions in {len(seq):,} nt)")

elif page == "Results":
    st.header("Motif Results Table")
    df = st.session_state.get('df', pd.DataFrame())
    if df.empty:
        st.info("No results. Please upload/run analysis.")
    else:
        st.dataframe(df, use_container_width=True)
        st.markdown("**Motif summary:**")
        st.write(df['Subtype'].value_counts().to_frame('Count'))

elif page == "Visualization":
    st.header("Motif Visualization")
    df = st.session_state.get('df', pd.DataFrame())
    seq = st.session_state.get('seq', '')
    if df.empty:
        st.info("No results yet.")
    else:
        # Motif map
        import matplotlib.pyplot as plt
        import seaborn as sns
        motif_types = sorted(df['Subtype'].unique())
        ymap = {t: i+1 for i, t in enumerate(motif_types)}
        colormap = sns.color_palette('tab10', len(motif_types))
        color_map = {t: colormap[i] for i, t in enumerate(motif_types)}
        fig, ax = plt.subplots(figsize=(12, len(motif_types)*0.5+3))
        for _, row in df.iterrows():
            y = ymap[row['Subtype']]
            ax.hlines(y, row['Start'], row['End'], color=color_map[row['Subtype']], lw=6)
        ax.set_yticks(list(ymap.values()))
        ax.set_yticklabels(list(ymap.keys()))
        ax.set_xlim(1, len(seq))
        ax.set_xlabel("Position (bp)")
        ax.set_title("Motif Locations")
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown("*Pie/bar charts available on request*")

elif page == "Download":
    st.header("Download Results")
    df = st.session_state.get('df', pd.DataFrame())
    if df.empty:
        st.info("Nothing to download.")
    else:
        st.download_button("Download CSV", df.to_csv(index=False).encode('utf-8'),
            file_name="motif_results.csv", mime="text/csv")
        output = pd.ExcelWriter("results.xlsx", engine='xlsxwriter')
        df.to_excel(output, index=False)
        output.close()
        with open("results.xlsx", "rb") as f:
            st.download_button("Download Excel", f.read(),
                file_name="motif_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
