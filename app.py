import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ----- UTILS -----
def wrap(seq, width=60):
    return "\n".join(seq[i:i+width] for i in range(0, len(seq), width))

def reverse_complement(seq):
    comp = str.maketrans("ATGC", "TACG")
    return seq.translate(comp)[::-1]

def parse_fasta(fasta_str):
    lines = fasta_str.strip().splitlines()
    seq = [line.strip() for line in lines if not line.startswith(">")]
    return "".join(seq).upper().replace(" ", "").replace("U", "T")

def gc_content(seq):
    seq = seq.upper()
    gc = seq.count("G") + seq.count("C")
    return 100.0 * gc / max(1, len(seq))

# ----- MOTIF FINDERS -----
def g4hunter_score(seq):
    seq = seq.upper()
    vals = []
    i = 0
    n = len(seq)
    while i < n:
        if seq[i] == 'G':
            run_len = 1
            while i + run_len < n and seq[i + run_len] == 'G':
                run_len += 1
            val = min(run_len, 4)
            vals.extend([val] * run_len)
            i += run_len
        elif seq[i] == 'C':
            run_len = 1
            while i + run_len < n and seq[i + run_len] == 'C':
                run_len += 1
            val = -min(run_len, 4)
            vals.extend([val] * run_len)
            i += run_len
        else:
            vals.append(0)
            i += 1
    return round(np.mean(vals), 2) if vals else 0

def find_gquadruplex(seq):
    # G-quadruplex: (G3+N1-7)3G3+
    pattern = r'(G{3,}[ATGC]{1,7}){3}G{3,}'
    matches = []
    for m in re.finditer(pattern, seq):
        region = seq[m.start():m.end()]
        matches.append(dict(
            Class="Quadruplex",
            Subtype="G-Quadruplex",
            Start=m.start()+1,
            End=m.end(),
            Length=len(region),
            Sequence=wrap(region),
            GC=f"{gc_content(region):.1f}",
            Score=g4hunter_score(region),
            ScoreMethod="G4Hunter"
        ))
    return matches

def find_zdna(seq):
    # Z-DNA: (GC|CG|GT|TG|AC|CA) >= 12bp (6 dinuc repeats)
    pattern = r'((?:GC|CG|GT|TG|AC|CA){6,})'
    matches = []
    for m in re.finditer(pattern, seq):
        region = seq[m.start():m.end()]
        score = len(re.findall(r'GC|CG|GT|TG|AC|CA', region))
        matches.append(dict(
            Class="Z-DNA",
            Subtype="Z-DNA",
            Start=m.start()+1,
            End=m.end(),
            Length=len(region),
            Sequence=wrap(region),
            GC=f"{gc_content(region):.1f}",
            Score=score,
            ScoreMethod="Z-Seeker"
        ))
    return matches

def find_cruciform(seq):
    # Inverted repeats, arms ≥6bp, loop ≤100bp, non-overlapping only
    matches = []
    n = len(seq)
    for arm in range(6, 21):  # arms 6-20 bp
        for loop in range(0, 101):
            pattern = rf"([ATGC]{{{arm}}})([ATGC]{{0,{loop}}})([ATGC]{{{arm}}})"
            for m in re.finditer(pattern, seq):
                left = m.group(1)
                right = m.group(3)
                if reverse_complement(left) == right:
                    region = seq[m.start():m.end()]
                    matches.append(dict(
                        Class="Inverted Repeat",
                        Subtype="Cruciform_DNA",
                        Start=m.start()+1,
                        End=m.end(),
                        Length=len(region),
                        Sequence=wrap(region),
                        GC=f"{gc_content(region):.1f}",
                        Score=arm,
                        ScoreMethod="Arm length"
                    ))
    return matches

def nonoverlapping_motifs(motifs):
    # Sort motifs by start, longest first; keep only non-overlapping
    mask = set()
    nonoverlap = []
    for m in sorted(motifs, key=lambda x: (x['Start'], -x['Length'])):
        if not any(i in mask for i in range(m['Start'], m['End']+1)):
            nonoverlap.append(m)
            mask.update(range(m['Start'], m['End']+1))
    return nonoverlap

def find_all_motifs(seq):
    motifs = []
    motifs += find_gquadruplex(seq)
    motifs += find_zdna(seq)
    motifs += find_cruciform(seq)
    return nonoverlapping_motifs(motifs)

# ----------- STREAMLIT APP -----------
EXAMPLE_FASTA = """>Example
ATCGATCGATCGAAAATTTTATTTAAATTTAAATTTGGGTTAGGGTTAGGGTTAGGGCCCCCTCCCCCTCCCCCTCCCC
ATCGATCGCGCGCGCGATCGCACACACACAGCTGCTGCTGCTTGGGAAAGGGGAAGGGTTAGGGAAAGGGGTTT
GGGTTTAGGGGGGAGGGGCTGCTGCTGCATGCGGGAAGGGAGGGTAGAGGGTCCGGTAGGAACCCCTAACCCCTAA
GAAAGAAGAAGAAGAAGAAGAAAGGAAGGAAGGAGGAGGAGGAGGAGGAGGAGGAGGAGGAGGAGGAGGG"""

st.set_page_config(page_title="Non-B DNA Motif Finder", layout="wide")

PAGES = ["Home", "Upload & Analyze", "Results", "Visualization", "Download"]
page = st.sidebar.radio("Navigation", PAGES)

if page == "Home":
    st.title("Non-B DNA Motif Finder")
    st.image("nbd.PNG", use_container_width=True)
    st.markdown("""
    Ultra-fast, reference-quality, non-B DNA motif finder.  
    Detects **G-quadruplex, Z-DNA, and Cruciform DNA** motifs using published, structure-specific scoring systems.
    """)
    st.markdown("""
    **Supported Motifs:**  
    - **G-Quadruplex:** (G3+N1-7)3G3+, scored by G4Hunter  
    - **Z-DNA:** Alternating pyrimidine-purine dinucleotide, scored by Z-Seeker logic  
    - **Cruciform DNA:** Inverted repeats with arms ≥6bp and ≤100bp loop, scored by arm length  
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
                results = find_all_motifs(seq)
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
        st.dataframe(df[['Class', 'Subtype', 'Start', 'End', 'Length', 'GC', 'ScoreMethod', 'Score', 'Sequence']],
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

# ---- End of file ----
