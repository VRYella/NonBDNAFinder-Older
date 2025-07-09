import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import Counter

# ------------- MOTIF CLASSES ----------------

def parse_fasta(fasta_str: str) -> str:
    lines = fasta_str.strip().splitlines()
    seq = [line.strip() for line in lines if not line.startswith(">")]
    return "".join(seq).upper().replace(" ", "").replace("U", "T")

def gc_content(seq: str) -> float:
    seq = seq.upper()
    gc = seq.count("G") + seq.count("C")
    return 100.0 * gc / max(1, len(seq))

def wrap(seq: str, width=60) -> str:
    return "\n".join([seq[i:i+width] for i in range(0, len(seq), width)])

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
    return np.mean(np.array(vals)) if vals else 0.0

# --------- MOTIF SCANNERS ---------

def find_apr_bent(seq):
    """APR/Bent DNA: At least 3 A-tracts (or T-tracts) of 3–11bp, spaced 7–15bp apart."""
    res = []
    for nt in ('A','T'):
        tract = f'{nt}{{3,11}}'
        for m in re.finditer(tract, seq):
            res.append((nt, m.start(), m.end()))
    # Group tracts by "clusters" spaced 7-15bp apart (center-to-center)
    apr_hits = []
    for nt in ('A','T'):
        tracts = sorted([ (s+e)//2 for n,s,e in res if n==nt ])
        if len(tracts) < 3:
            continue
        for i in range(len(tracts)-2):
            d1 = tracts[i+1]-tracts[i]
            d2 = tracts[i+2]-tracts[i+1]
            if 7 <= d1 <= 15 and 7 <= d2 <= 15:
                # Merge the three tracts
                s = tracts[i] - 5 if tracts[i]-5 >= 0 else tracts[i]
                e = tracts[i+2] + 6 if tracts[i+2]+6 < len(seq) else tracts[i+2]
                region = seq[s:e]
                apr_hits.append({
                    "Motif Class": "Bent DNA/APR",
                    "Subtype": f"{nt}-tract APR",
                    "Start": s+1, "End": e,
                    "Length": len(region),
                    "GC (%)": f"{gc_content(region):.1f}",
                    "Propensity/Score": "Bend",
                    "Sequence": wrap(region),
                })
    return apr_hits

def find_direct_repeats(seq):
    """Direct repeats: units 10-300bp, spacer up to 100bp."""
    results = []
    for unit_len in range(10, 51, 5): # Up to 50bp, for speed (change to 301 for full)
        pattern = re.compile(f'([ATGC]{{{unit_len}}})([ATGC]{{0,100}})\\1')
        for m in pattern.finditer(seq):
            region = seq[m.start():m.end()]
            results.append({
                "Motif Class": "Direct Repeat",
                "Subtype": f"DR{unit_len}bp",
                "Start": m.start()+1, "End": m.end(),
                "Length": len(region),
                "GC (%)": f"{gc_content(region):.1f}",
                "Propensity/Score": f"{unit_len}bp",
                "Sequence": wrap(region)
            })
    return results

def find_inverted_repeats(seq):
    """Inverted repeats: arms ≥6bp, loop ≤100bp."""
    results = []
    min_arm = 6
    max_loop = 100
    for arm in range(min_arm, 21): # Up to 20bp for performance (increase if needed)
        pat = re.compile(f'([ATGC]{{{arm}}})([ATGC]{{0,{max_loop}}})\\1'[::-1])
        for m in pat.finditer(seq):
            region = seq[m.start():m.end()]
            results.append({
                "Motif Class": "Inverted Repeat",
                "Subtype": f"IR{arm}bp",
                "Start": m.start()+1, "End": m.end(),
                "Length": len(region),
                "GC (%)": f"{gc_content(region):.1f}",
                "Propensity/Score": f"{arm}bp",
                "Sequence": wrap(region)
            })
    return results

def find_mirror_repeats(seq):
    """Mirror repeats: arms ≥10bp, loop ≤100bp, arm == arm (not complement)."""
    results = []
    min_arm = 10
    max_loop = 100
    for arm in range(min_arm, 21, 2):
        for m in re.finditer(f'([ATGC]{{{arm}}})([ATGC]{{0,{max_loop}}})\\1', seq):
            region = seq[m.start():m.end()]
            # Mirror: arm == arm (not complement, just repeated)
            results.append({
                "Motif Class": "Mirror Repeat",
                "Subtype": f"MR{arm}bp",
                "Start": m.start()+1, "End": m.end(),
                "Length": len(region),
                "GC (%)": f"{gc_content(region):.1f}",
                "Propensity/Score": f"{arm}bp",
                "Sequence": wrap(region)
            })
    return results

def find_strs(seq):
    """STRs: unit 1-9bp, total length ≥10bp."""
    results = []
    for unit in range(1, 10):
        pat = re.compile(f'((?:[ATGC]{{{unit}}}){{2,}})')
        for m in pat.finditer(seq):
            region = m.group(0)
            if len(region) >= 10:
                results.append({
                    "Motif Class": "STR",
                    "Subtype": f"{unit}bp",
                    "Start": m.start()+1, "End": m.end(),
                    "Length": len(region),
                    "GC (%)": f"{gc_content(region):.1f}",
                    "Propensity/Score": f"{unit}bp",
                    "Sequence": wrap(region)
                })
    return results

def find_zdna(seq):
    """Z-DNA: alternating purine-pyrimidine, >10bp."""
    pattern = re.compile(r'((?:GC|CG|GT|TG|AC|CA){5,})')
    results = []
    for m in pattern.finditer(seq):
        region = seq[m.start():m.end()]
        if len(region) > 10:
            results.append({
                "Motif Class": "Z-DNA",
                "Subtype": "Classic",
                "Start": m.start()+1, "End": m.end(),
                "Length": len(region),
                "GC (%)": f"{gc_content(region):.1f}",
                "Propensity/Score": "Z",
                "Sequence": wrap(region)
            })
    return results

def find_g4(seq):
    """G-quadruplex, i-motif, G-triplex, multimeric G4 (with scoring)."""
    motifs = []
    # G4
    for m in re.finditer(r'(G{3,}[ATGC]{1,12}){3}G{3,}', seq):
        region = seq[m.start():m.end()]
        motifs.append({
            "Motif Class": "G4/i-Motif",
            "Subtype": "G-Quadruplex",
            "Start": m.start()+1, "End": m.end(),
            "Length": len(region),
            "GC (%)": f"{gc_content(region):.1f}",
            "Propensity/Score": f"{g4hunter_score(region):.2f}",
            "Sequence": wrap(region)
        })
    # i-Motif
    for m in re.finditer(r'(C{3,}[ATGC]{1,12}){3}C{3,}', seq):
        region = seq[m.start():m.end()]
        motifs.append({
            "Motif Class": "G4/i-Motif",
            "Subtype": "i-Motif",
            "Start": m.start()+1, "End": m.end(),
            "Length": len(region),
            "GC (%)": f"{gc_content(region):.1f}",
            "Propensity/Score": f"{-g4hunter_score(region):.2f}",
            "Sequence": wrap(region)
        })
    # G-Triplex
    for m in re.finditer(r'(G{3,}[ATGC]{1,12}){2}G{3,}', seq):
        region = seq[m.start():m.end()]
        motifs.append({
            "Motif Class": "G4/i-Motif",
            "Subtype": "G-Triplex",
            "Start": m.start()+1, "End": m.end(),
            "Length": len(region),
            "GC (%)": f"{gc_content(region):.1f}",
            "Propensity/Score": f"{g4hunter_score(region)*0.7:.2f}",
            "Sequence": wrap(region)
        })
    return motifs

def find_sticky_dna(seq):
    """Sticky DNA: extended GAA/TTC repeats."""
    results = []
    for pat in ['(GAA){5,}', '(TTC){5,}']:
        for m in re.finditer(pat, seq):
            region = seq[m.start():m.end()]
            results.append({
                "Motif Class": "Triplex",
                "Subtype": "Sticky_DNA",
                "Start": m.start()+1, "End": m.end(),
                "Length": len(region),
                "GC (%)": f"{gc_content(region):.1f}",
                "Propensity/Score": f"{len(region)//3} repeats",
                "Sequence": wrap(region)
            })
    return results

def find_triplex(seq):
    """Triplex: Mirror repeat with arms ≥10bp, purine/pyrimidine-rich, loop ≤8bp."""
    results = []
    pat = re.compile(r'([AGCT]{10,})([ATGC]{0,8})\1')
    for m in pat.finditer(seq):
        left, spacer, right = m.group(1), m.group(2), m.group(1)
        # Check purine/pyrimidine content
        purines = left.count('A') + left.count('G')
        pyrimidines = left.count('C') + left.count('T')
        pct = max(purines, pyrimidines) / len(left)
        if pct >= 0.10:
            region = seq[m.start():m.end()]
            results.append({
                "Motif Class": "Triplex",
                "Subtype": "Mirror Triplex",
                "Start": m.start()+1, "End": m.end(),
                "Length": len(region),
                "GC (%)": f"{gc_content(region):.1f}",
                "Propensity/Score": f"{pct:.2f}",
                "Sequence": wrap(region)
            })
    return results

# -------- MAIN MOTIF SCANNER --------
def find_all_motifs(seq):
    return (
        find_apr_bent(seq) +
        find_direct_repeats(seq) +
        find_inverted_repeats(seq) +
        find_mirror_repeats(seq) +
        find_strs(seq) +
        find_zdna(seq) +
        find_g4(seq) +
        find_sticky_dna(seq) +
        find_triplex(seq)
    )

# ------------- STREAMLIT APP ------------

EXAMPLE_FASTA = """>Example
ATCGATCGATCGAAAATTTTATTTAAATTTAAATTTGGGTTAGGGTTAGGGTTAGGGCCCCCTCCCCCTCCCCCTCCCC
ATCGATCGCGCGCGCGATCGCACACACACAGCTGCTGCTGCTTGGGAAAGGGGAAGGGTTAGGGAAAGGGGTTT
GGGTTTAGGGGGGAGGGGCTGCTGCTGCATGCGGGAAGGGAGGGTAGAGGGTCCGGTAGGAACCCCTAACCCCTAA
GAAAGAAGAAGAAGAAGAAGAAAGGAAGGAAGGAGGAGGAGGAGGAGGAGGAGGAGGAGGAGGAGGAGGG
"""

st.set_page_config(page_title="Non-B DNA Motif Finder", layout="wide")
st.image("nbd.PNG", use_column_width=True)

st.title("Non-B DNA Motif Finder (FAST Modular Version)")
st.markdown("Upload, paste or use the example FASTA. Detects: G4, i-Motif, G-Triplex, Sticky, APR/Bent, DR, IR, MR, STRs, Triplex, Z-DNA.")

fasta_file = st.file_uploader("Upload FASTA file", type=["fa", "fasta", "txt"])
if st.button("Use Example Sequence"):
    st.session_state['seq'] = parse_fasta(EXAMPLE_FASTA)
else:
    st.session_state['seq'] = ""
seq_input = st.text_area("Paste sequence (FASTA or raw)", value=st.session_state.get('seq', ""), height=120)
if fasta_file:
    seq = parse_fasta(fasta_file.read().decode("utf-8"))
    st.session_state['seq'] = seq
elif seq_input:
    seq = parse_fasta(seq_input)
    st.session_state['seq'] = seq
else:
    seq = st.session_state.get('seq', "")

if st.button("Run Analysis"):
    if not seq or not re.match("^[ATGC]+$", seq):
        st.error("Provide a valid DNA sequence (A/T/G/C only, after FASTA parsing).")
    else:
        with st.spinner("Scanning for non-B DNA motifs..."):
            motif_results = find_all_motifs(seq)
            st.session_state['motif_results'] = motif_results
            st.session_state['df'] = pd.DataFrame(motif_results)
        if not motif_results:
            st.warning("No non-B DNA motifs detected in this sequence.")
        else:
            st.success(f"Detected {len(motif_results)} motif regions in {len(seq):,} bp.")

if 'df' not in st.session_state:
    st.session_state['df'] = pd.DataFrame()
if 'motif_results' not in st.session_state:
    st.session_state['motif_results'] = []

st.header("Motif Results")
df = st.session_state.get('df', pd.DataFrame())
if not df.empty:
    st.dataframe(df, use_container_width=True, hide_index=True)
    with st.expander("Motif Class Summary"):
        motif_counts = df["Motif Class"].value_counts().reset_index()
        motif_counts.columns = ["Motif Class", "Count"]
        st.dataframe(motif_counts, use_container_width=True, hide_index=True)

    st.subheader("Visualization")
    motif_types = sorted(df['Motif Class'].unique())
    color_palette = sns.color_palette('husl', n_colors=len(motif_types))
    color_map = {typ: color_palette[i] for i, typ in enumerate(motif_types)}
    y_map = {typ: i+1 for i, typ in enumerate(motif_types)}
    fig, ax = plt.subplots(figsize=(10, len(motif_types)*0.7+2))
    for _, motif in df.iterrows():
        motif_type = motif['Motif Class']
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
    counts = df['Motif Class'].value_counts()
    fig2, ax2 = plt.subplots()
    ax2.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
    ax2.axis('equal')
    st.pyplot(fig2)

    st.subheader("Motif Counts (Bar Chart)")
    fig3, ax3 = plt.subplots()
    counts.plot.bar(ax=ax3)
    ax3.set_ylabel("Count")
    ax3.set_xlabel("Motif Class")
    plt.tight_layout()
    st.pyplot(fig3)

    st.header("Download")
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
else:
    st.info("No results to show. Upload/paste sequence and run analysis.")
