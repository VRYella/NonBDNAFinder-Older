import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# --- Utility functions ---
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

# --- Motif scoring functions ---
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

def z_seeker_score(seq):
    # Number of alternating purine/pyrimidine dinucleotides
    return len(re.findall(r'GC|CG|GT|TG|AC|CA', seq))

def arm_score(arm):
    return len(arm)

# --- Motif finders (non-overlapping, longest-first) ---
def find_gquadruplex(seq):
    # G-quadruplex: (G3+N1-7)3G3+
    pattern = r'(G{3,}[ATGC]{1,7}){3}G{3,}'
    results = []
    for m in re.finditer(pattern, seq):
        region = seq[m.start():m.end()]
        results.append(dict(
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
    return results

def find_imotif(seq):
    # i-Motif: (C3+N1-7)3C3+
    pattern = r'(C{3,}[ATGC]{1,7}){3}C{3,}'
    results = []
    for m in re.finditer(pattern, seq):
        region = seq[m.start():m.end()]
        results.append(dict(
            Class="Quadruplex",
            Subtype="i-Motif",
            Start=m.start()+1,
            End=m.end(),
            Length=len(region),
            Sequence=wrap(region),
            GC=f"{gc_content(region):.1f}",
            Score=-g4hunter_score(region.replace("G", "C").replace("C", "G")), # flip for C
            ScoreMethod="G4Hunter"
        ))
    return results

def find_bipartite_g4(seq):
    # Two G4s separated by <=100bp
    pattern = r'(G{3,}[ATGC]{1,7}){3}G{3,}[ATGC]{0,100}(G{3,}[ATGC]{1,7}){3}G{3,}'
    results = []
    for m in re.finditer(pattern, seq):
        region = seq[m.start():m.end()]
        results.append(dict(
            Class="Quadruplex",
            Subtype="Bipartite_G-Quadruplex",
            Start=m.start()+1,
            End=m.end(),
            Length=len(region),
            Sequence=wrap(region),
            GC=f"{gc_content(region):.1f}",
            Score=g4hunter_score(region),
            ScoreMethod="G4Hunter"
        ))
    return results

def find_gtriplex(seq, g4_spans):
    # Only report G-triplex if not overlapped with G4s
    pattern = r'(G{3,}[ATGC]{1,7}){2}G{3,}'
    results = []
    for m in re.finditer(pattern, seq):
        region = seq[m.start():m.end()]
        # Overlap filter
        s, e = m.start(), m.end()
        overlaps_g4 = any((s < g4e and e > g4s) for g4s, g4e in g4_spans)
        if not overlaps_g4:
            results.append(dict(
                Class="Quadruplex",
                Subtype="G-Triplex",
                Start=s+1,
                End=e,
                Length=len(region),
                Sequence=wrap(region),
                GC=f"{gc_content(region):.1f}",
                Score=g4hunter_score(region) * 0.75,
                ScoreMethod="G4Hunter scaled"
            ))
    return results

def find_zdna(seq):
    pattern = r'((?:GC|CG|GT|TG|AC|CA){6,})'
    results = []
    for m in re.finditer(pattern, seq):
        region = seq[m.start():m.end()]
        score = z_seeker_score(region)
        results.append(dict(
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
    return results

def find_cruciform(seq):
    results = []
    n = len(seq)
    for arm in range(6, 21):  # arms 6-20 bp for speed
        for loop in range(0, 101):
            pattern = rf"([ATGC]{{{arm}}})([ATGC]{{0,{loop}}})([ATGC]{{{arm}}})"
            for m in re.finditer(pattern, seq):
                left = m.group(1)
                right = m.group(3)
                if reverse_complement(left) == right:
                    region = seq[m.start():m.end()]
                    results.append(dict(
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
    return results

def find_hdna(seq):
    # Mirror repeats: ([AG]{10,}|[CT]{10,})([ATGC]{0,8})([AG]{10,}|[CT]{10,})
    pattern = r'([AG]{10,}|[CT]{10,})([ATGC]{0,8})([AG]{10,}|[CT]{10,})'
    results = []
    for m in re.finditer(pattern, seq):
        left, spacer, right = m.group(1), m.group(2), m.group(3)
        if left == right[::-1] and len(left) >= 10 and len(right) >= 10:
            region = seq[m.start():m.end()]
            results.append(dict(
                Class="Triplex",
                Subtype="H-DNA",
                Start=m.start()+1,
                End=m.end(),
                Length=len(region),
                Sequence=wrap(region),
                GC=f"{gc_content(region):.1f}",
                Score="NA",
                ScoreMethod="Mirror repeat"
            ))
    return results

def find_sticky_dna(seq):
    # (GAA)5+ or (TTC)5+
    pattern = r'(?:GAA){5,}|(?:TTC){5,}'
    results = []
    for m in re.finditer(pattern, seq):
        region = seq[m.start():m.end()]
        results.append(dict(
            Class="Triplex",
            Subtype="Sticky_DNA",
            Start=m.start()+1,
            End=m.end(),
            Length=len(region),
            Sequence=wrap(region),
            GC=f"{gc_content(region):.1f}",
            Score=region.count("GAA")+region.count("TTC"),
            ScoreMethod="Repeat count"
        ))
    return results

def find_direct_repeats(seq):
    # Unit length 10–300bp, spacer ≤100bp, no overlap
    results = []
    for unit_len in range(10, 31):  # 10–30 for speed; can increase to 300
        pattern = rf'([ATGC]{{{unit_len}}})([ATGC]{{0,100}})\1'
        for m in re.finditer(pattern, seq):
            region = seq[m.start():m.end()]
            results.append(dict(
                Class="Direct Repeat",
                Subtype="Slipped_DNA",
                Start=m.start()+1,
                End=m.end(),
                Length=len(region),
                Sequence=wrap(region),
                GC=f"{gc_content(region):.1f}",
                Score=unit_len,
                ScoreMethod="Unit length"
            ))
    return results

def find_mirror_repeats(seq):
    # Arms 10bp+, loop ≤100bp
    results = []
    for arm in range(10, 21):
        pattern = rf'([ATGC]{{{arm}}})([ATGC]{{0,100}})\1'
        for m in re.finditer(pattern, seq):
            left, loop, right = m.group(1), m.group(2), m.group(1)
            if left == right[::-1]:
                region = seq[m.start():m.end()]
                results.append(dict(
                    Class="Mirror Repeat",
                    Subtype="Mirror_Repeat",
                    Start=m.start()+1,
                    End=m.end(),
                    Length=len(region),
                    Sequence=wrap(region),
                    GC=f"{gc_content(region):.1f}",
                    Score=arm,
                    ScoreMethod="Arm length"
                ))
    return results

def find_local_bends(seq):
    # A-tracts (6–7), T-tracts (6–7)
    pattern = r'A{6,7}|T{6,7}'
    results = []
    for m in re.finditer(pattern, seq):
        region = seq[m.start():m.end()]
        results.append(dict(
            Class="Local Bend",
            Subtype="A/T-tract",
            Start=m.start()+1,
            End=m.end(),
            Length=len(region),
            Sequence=wrap(region),
            GC=f"{gc_content(region):.1f}",
            Score="NA",
            ScoreMethod="A/T-tract"
        ))
    return results

def find_local_flexible(seq):
    # CA or TG dinucleotide bends, 4+ repeats
    pattern = r'(?:CA){4,}|(?:TG){4,}'
    results = []
    for m in re.finditer(pattern, seq):
        region = seq[m.start():m.end()]
        results.append(dict(
            Class="Local Flexibility",
            Subtype="CA/TG_dinucleotide",
            Start=m.start()+1,
            End=m.end(),
            Length=len(region),
            Sequence=wrap(region),
            GC=f"{gc_content(region):.1f}",
            Score="NA",
            ScoreMethod="Dinucleotide"
        ))
    return results

def find_str(seq):
    # STR: unit 1-9bp, total len ≥10bp, at least 2 repeats
    results = []
    n = len(seq)
    for unit_len in range(1, 10):
        i = 0
        while i < n - unit_len:
            unit = seq[i:i+unit_len]
            reps = 1
            j = i + unit_len
            while j + unit_len <= n and seq[j:j+unit_len] == unit:
                reps += 1
                j += unit_len
            total_len = reps * unit_len
            if reps >= 2 and total_len >= 10:
                region = seq[i:j]
                results.append(dict(
                    Class="STR",
                    Subtype=f"STR_{unit_len}bp",
                    Start=i+1,
                    End=j,
                    Length=len(region),
                    Sequence=wrap(region),
                    GC=f"{gc_content(region):.1f}",
                    Score=reps,
                    ScoreMethod="Repeat count"
                ))
                i = j  # Skip overlapping
            else:
                i += 1
    return results

def collect_all_motifs(seq):
    # Run all finders
    g4 = find_gquadruplex(seq)
    g4_spans = [(m['Start']-1, m['End']) for m in g4]
    motifs = (
        g4
        + find_imotif(seq)
        + find_bipartite_g4(seq)
        + find_gtriplex(seq, g4_spans)
        + find_zdna(seq)
        + find_cruciform(seq)
        + find_hdna(seq)
        + find_sticky_dna(seq)
        + find_direct_repeats(seq)
        + find_mirror_repeats(seq)
        + find_local_bends(seq)
        + find_local_flexible(seq)
        + find_str(seq)
    )
    # Non-overlapping, longest first
    motifs.sort(key=lambda x: (x['Start'], -x['Length']))
    mask = set()
    nonoverlap = []
    for m in motifs:
        s, e = m['Start'], m['End']
        if not any(i in mask for i in range(s, e+1)):
            nonoverlap.append(m)
            mask.update(range(s, e+1))
    return nonoverlap

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
    Comprehensive, fast, and reference-grade non-B DNA motif finder.  
    **Motifs:** G-quadruplex, i-Motif, Bipartite G4, G-Triplex, Z-DNA (Z-Seeker), Cruciform, H-DNA, Sticky DNA, Direct/Mirror Repeats, STRs, local bends, flexible regions, and more.
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
