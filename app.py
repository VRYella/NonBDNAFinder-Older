import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re, io
from datetime import datetime
from motifs import (
    find_gquadruplex, find_imotif, find_gtriplex,
    find_bipartite_gquadruplex, find_multimeric_gquadruplex,
    find_zdna, find_hdna, find_sticky_dna,
    find_slipped_dna, find_cruciform, find_bent_dna,
    find_apr, find_mirror_repeat,
    find_quadruplex_triplex_hybrid,
    find_cruciform_triplex_junction,
    find_g4_imotif_hybrid
)
from utils import parse_fasta, wrap

EXAMPLE_FASTA = """>Example
ATCGATCGATCGAAAATTTTATTTAAATTTAAATTTGGGTTAGGGTTAGGGTTAGGGCCCCCTCCCCCTCCCCCTCCCC
ATCGATCGCGCGCGCGATCGCACACACACAGCTGCTGCTGCTTGGGAAAGGGGAAGGGTTAGGGAAAGGGGTTT
GGGTTTAGGGGGGAGGGGCTGCTGCTGCATGCGGGAAGGGAGGGTAGAGGGTCCGGTAGGAACCCCTAACCCCTAA
GAAAGAAGAAGAAGAAGAAGAAAGGAAGGAAGGAGGAGGAGGAGGAGGAGGAGGAGGAGGAGGAGGAGGG
"""

st.set_page_config(page_title="Non-B DNA Motif Finder", layout="wide")

if 'seq' not in st.session_state:
    st.session_state['seq'] = ""
if 'df' not in st.session_state:
    st.session_state['df'] = pd.DataFrame()
if 'motif_results' not in st.session_state:
    st.session_state['motif_results'] = []

PAGES = ["Home", "Upload & Analyze", "Results", "Visualization", "Download"]
page = st.sidebar.radio("Navigation", PAGES)

def collect_all_motifs(seq):
    results = []
    results += find_gquadruplex(seq)
    results += find_imotif(seq)
    results += find_gtriplex(seq)
    results += find_bipartite_gquadruplex(seq)
    results += find_multimeric_gquadruplex(seq)
    results += find_zdna(seq)
    results += find_hdna(seq)
    results += find_sticky_dna(seq)
    results += find_slipped_dna(seq)
    results += find_cruciform(seq)
    results += find_bent_dna(seq)
    results += find_apr(seq)
    results += find_mirror_repeat(seq)
    results += find_quadruplex_triplex_hybrid(seq)
    results += find_cruciform_triplex_junction(seq)
    results += find_g4_imotif_hybrid(seq)
    # Remove overlapping motifs (keep highest score if numeric)
    results = sorted(results, key=lambda x: (x['Start'], -float(x['Score']) if str(x['Score']).replace('.','',1).isdigit() else 0))
    nonoverlap = []
    covered = set()
    for r in results:
        covered_range = set(range(r['Start'], r['End'] + 1))
        if not covered_range & covered:
            nonoverlap.append(r)
            covered |= covered_range
    return nonoverlap

if page == "Home":
    st.title("Non-B DNA Motif Finder")
    try:
        st.image("nbd.PNG", use_container_width=True)
    except Exception:
        st.warning("Logo image (nbd.PNG) not found. Place it in the app folder.")

    st.markdown("""
    **Comprehensive, fast, and reference-grade non-B DNA motif finder.**
    - G-quadruplex, i-Motif, Bipartite G4, G-Triplex, Z-DNA (Z-Seeker), Cruciform, H-DNA, Sticky DNA, Direct/Mirror Repeats, STRs, local bends, flexible regions, and more.
    - Export to CSV/Excel, motif visualization included.
    """)

elif page == "Upload & Analyze":
    st.header("Input Sequence")
    col1, col2 = st.columns([1,1])
    with col1:
        fasta_file = st.file_uploader("Upload FASTA file", type=["fa", "fasta", "txt"])
        if fasta_file:
            try:
                seq = parse_fasta(fasta_file.read().decode("utf-8"))
                st.session_state['seq'] = seq
                st.success("FASTA file loaded!")
            except Exception:
                st.error("Could not parse file as UTF-8 or FASTA.")
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
            with st.spinner("Analyzing sequence ..."):
                results = collect_all_motifs(seq)
                st.session_state['motif_results'] = results
                st.session_state['df'] = pd.DataFrame(results)
            if not results:
                st.warning("No non-B DNA motifs detected in this sequence.")
            else:
                st.success(f"Detected {len(results)} motif region(s) in {len(seq):,} bp.")

elif page == "Results":
    st.header("Motif Detection Results")
    df = st.session_state.get('df', pd.DataFrame())
    if df.empty:
        st.info("No results yet. Go to 'Upload & Analyze' and run analysis.")
    else:
        st.markdown(f"**Sequence length:** {len(st.session_state['seq']):,} bp")
        st.dataframe(df[['Class', 'Subtype', 'Start', 'End', 'Length', 'Sequence', 'ScoreMethod', 'Score']],
            use_container_width=True, hide_index=True)
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

st.markdown("""
---
**Developed by [Your Name] & Team** | [GitHub](https://github.com/yourrepo)
""")
