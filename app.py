import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.graph_objects as go

# --- MOTIFS, CLASSES, and EXPLANATIONS ---
MOTIF_INFO = [
    ("G-Quadruplex", "Guanine-rich, four-stranded DNA structure, four runs of ≥3 Gs, loops of 1-12 nt."),
    ("i-Motif", "Cytosine-rich quadruplex, forms at low pH."),
    ("Bipartite_G-Quadruplex", "Two G4s separated by ≤100 nt."),
    ("G-Triplex", "Three guanine runs, possible triple-stranded structure."),
    ("Multimeric_G-Quadruplex", "Several tandem G4s, separated by ≤50 nt."),
    ("Z-DNA", "Left-handed helix, alternating purine-pyrimidine (GC/CG/GT/TG/AC/CA) repeats."),
    ("H-DNA", "Triplex-forming mirror repeats (polypurine/polypyrimidine)."),
    ("Sticky_DNA", "Extended GAA or TTC repeats (≥5 units)."),
    ("Slipped_DNA", "Direct repeats that can slip during replication."),
    ("Cruciform_DNA", "Inverted repeats, may form cruciform/hairpin."),
    ("Bent_DNA", "A-tract/T-tract periodicity causing DNA bending."),
    ("Quadruplex-Triplex_Hybrid", "Region with G4 and triplex potential."),
    ("Cruciform-Triplex_Junctions", "Junctions with both cruciform and triplex potential."),
    ("G-Quadruplex_i-Motif_Hybrid", "Close proximity of G4 and i-motif sequences."),
]

MOTIFS = [
    ("Quadruplex", r"(G{3,}[ATGC]{1,12}){3}G{3,}", "G-Quadruplex"),
    ("Quadruplex", r"(C{3,}[ATGC]{1,12}){3}C{3,}", "i-Motif"),
    ("Quadruplex", r"(G{3,}[ATGC]{1,12}){3}G{3,}[ATGC]{0,100}(G{3,}[ATGC]{1,12}){3}G{3,}", "Bipartite_G-Quadruplex"),
    ("Quadruplex", r"(G{3,}[ATGC]{1,12}){2}G{3,}", "G-Triplex"),
    ("Quadruplex", r"((G{3,}[ATGC]{1,12}){3}G{3,}([ATGC]{1,50}(G{3,}[ATGC]{1,12}){3}G{3,})+)", "Multimeric_G-Quadruplex"),
    ("Z-DNA", r"((GC|CG|GT|TG|AC|CA){6,})", "Z-DNA"),
    ("Triplex", r"([AG]{10,}|[CT]{10,})([ATGC]{0,100})([AG]{10,}|[CT]{10,})", "H-DNA"),
    ("Triplex", r"(GAA){5,}|(TTC){5,}", "Sticky_DNA"),
    ("Direct Repeat", r"([ATGC]{10,25})([ATGC]{0,10})\1", "Slipped_DNA"),
    ("Inverted Repeat", r"([ATGC]{10,})([ATGC]{0,100})\1", "Cruciform_DNA"),
    ("Bent DNA", r"(A{4,6}|T{4,6})([ATGC]{7,11})(A{4,6}|T{4,6})([ATGC]{7,11})(A{4,6}|T{4,6})", "Bent_DNA"),
    ("Quadruplex-Triplex Hybrid", r"(G{3,}[ATGC]{1,12}){3}G{3,}[ATGC]{0,100}([AG]{10,}|[CT]{10,})", "Quadruplex-Triplex_Hybrid"),
    ("Cruciform-Triplex Junction", r"([ATGC]{10,})([ATGC]{0,100})([ATGC]{10,})([ATGC]{0,100})([AG]{10,}|[CT]{10,})", "Cruciform-Triplex_Junctions"),
    ("G-Quadruplex_i-Motif_Hybrid", r"(G{3,}[ATGC]{1,12}){3}G{3,}[ATGC]{0,100}(C{3,}[ATGC]{1,12}){3}C{3,}", "G-Quadruplex_i-Motif_Hybrid"),
]

EXAMPLE_FASTA = """>Example
ATCGATCGATCGAAAATTTTATTTAAATTTAAATTTGGGTTAGGGTTAGGGTTAGGGCCCCCTCCCCCTCCCCCTCCCC
ATCGATCGCGCGCGCGATCGCACACACACAGCTGCTGCTGCTTGGGAAAGGGGAAGGGTTAGGGAAAGGGGTTT
GGGTTTAGGGGGGAGGGGCTGCTGCTGCATGCGGGAAGGGAGGGTAGAGGGTCCGGTAGGAACCCCTAACCCCTAA
GAAAGAAGAAGAAGAAGAAGAAAGGAAGGAAGGAGGAGGAGGAGGAGGAGGAGGAGGAGGAGGAGGAGGG
"""

# --------- UTILITIES ---------
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
        score = g4hunter_score(seq)
        return f"{score * 0.7:.2f}"
    if name == "Multimeric_G-Quadruplex":
        g4s = re.findall(r"(G{3,}[ATGC]{1,12}){3}G{3,}", seq)
        if g4s:
            return f"{np.mean([g4hunter_score(g) for g in g4s]):.2f}"
        return "NA"
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
    if name == "Slipped_DNA":
        dr = re.findall(r"([ATGC]{10,25})", seq)
        return f"{len(dr)}" if dr else "NA"
    if name == "Cruciform_DNA":
        ir = re.findall(r"([ATGC]{10,})", seq)
        return f"{max([len(i) for i in ir])}bp-arm" if ir else "NA"
    if name == "Bent_DNA":
        tracts = re.findall(r"A{4,6}|T{4,6}", seq)
        return f"{max([len(t) for t in tracts])}bp-tract" if tracts else "NA"
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
                "Sequence": wrap(region.replace("_", " "), 60),  # Replace _ with space
            })
    return results

def find_multiconformational(results: list, max_gap=10) -> list:
    if not results:
        return []
    df = pd.DataFrame(results)
    df = df.sort_values("Start")
    mcrs = []
    for i in range(len(df)-1):
        curr, nxt = df.iloc[i], df.iloc[i+1]
        if nxt["Start"] - curr["End"] <= max_gap and curr["Subtype"] != nxt["Subtype"]:
            mcr_seq = curr["Sequence"].replace("\n", "").replace("_", " ") + nxt["Sequence"].replace("\n", "").replace("_", " ")
            mcrs.append({
                "Motif Class": "Multi-Conformational",
                "Subtype": f"{curr['Subtype']}/{nxt['Subtype']}",
                "Start": curr["Start"],
                "End": nxt["End"],
                "Length": nxt["End"] - curr["Start"] + 1,
                "GC (%)": f"{gc_content(mcr_seq):.1f}",
                "Propensity/Score": "NA",
                "Sequence": wrap(mcr_seq.replace("_", " "), 60)
            })
    return mcrs

def csv_download_button(df, label="Download CSV"):
    nowstr = datetime.now().strftime("%Y%m%d-%H%M%S")
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode('utf-8'),
        file_name=f"motifs_{nowstr}.csv",
        mime="text/csv"
    )

def excel_download_button(df, label="Download Excel"):
    nowstr = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = io.BytesIO()
    try:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
    except ImportError:
        st.error("openpyxl is required for Excel export. Please install it.")
        return
    output.seek(0)
    st.download_button(
        label=label,
        data=output.getvalue(),
        file_name=f"motifs_{nowstr}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# ----------- APP LAYOUT -----------
st.set_page_config(page_title="Non-B DNA Motif Finder", layout="wide")

# --- Custom CSS for modern look ---
st.markdown("""
    <style>
    body { background: #f8f9fc !important; }
    .stApp { font-family: 'Segoe UI',sans-serif; }
    .motif-bar { background: #e9ecef; border-radius: 8px; padding: 8px; }
    .motif-label { font-weight: 600; color: #22223b; }
    </style>
""", unsafe_allow_html=True)

# --- Responsive header image ---
st.markdown(
    """
    <div style="width:100%; display:flex; justify-content:center;">
        <img src="https://raw.githubusercontent.com/VRYella/NonBDNAFinder/main/nbd.PNG" 
             alt="Non-B DNA Finder Banner" 
             style="width:100%; max-width:1400px; height:auto; display:block; margin-bottom:-20px;"/>
    </div>
    """,
    unsafe_allow_html=True
)

st.title("Non-B DNA Motif Finder")
st.caption("Detects and visualizes non-B DNA motifs, including G-quadruplexes, triplexes, Z-DNA, and more.")

with st.expander("ℹ️ Motif Explanations"):
    for name, expl in MOTIF_INFO:
        st.markdown(f"**{name}**: {expl}")

st.write(
    "Upload a FASTA file or paste a DNA sequence below (A/T/G/C only). "
    "Click **Run** to begin."
)

# --------- Input Area ---------
col1, col2 = st.columns(2)
with col1:
    fasta_file = st.file_uploader("Upload FASTA file", type=["fa", "fasta", "txt"])
with col2:
    if st.button("Use Example Sequence"):
        st.session_state["input_seq"] = EXAMPLE_FASTA
    input_seq = st.text_area("Paste sequence in FASTA format", st.session_state.get("input_seq", EXAMPLE_FASTA), height=120)

# --------- Run & Stop Buttons (Horizontal) ---------
run_col, stop_col, status_col = st.columns([1,1,3])
with run_col:
    run = st.button("▶️ Run", key="run", use_container_width=True)
with stop_col:
    stop = st.button("🛑 Stop", key="stop", use_container_width=True)
with status_col:
    if stop:
        st.warning("Processing stopped by user.")
        st.stop()
    if run:
        st.info("Running analysis...", icon="🧬")

seq = None
if "input_seq" in st.session_state and st.session_state["input_seq"] == EXAMPLE_FASTA:
    seq = parse_fasta(EXAMPLE_FASTA)
elif fasta_file is not None:
    try:
        seq = parse_fasta(fasta_file.read().decode("utf-8"))
    except Exception:
        st.error("File could not be decoded as UTF-8. Please upload a valid text-based FASTA file.")
        st.stop()
elif input_seq.strip():
    seq = parse_fasta(input_seq.strip())

if not run:
    st.info("Load or paste a sequence, then click 'Run'.")
    st.stop()

if not seq or not re.match("^[ATGC]+$", seq):
    st.error("No valid DNA sequence detected. Please upload or paste a valid FASTA (A/T/G/C only).")
    st.stop()

st.markdown(f"<div class='motif-label'>Sequence length: <b>{len(seq):,} bp</b></div>", unsafe_allow_html=True)

with st.spinner('Analyzing sequence...'):
    results = find_motifs(seq)
    multi_conf = find_multiconformational(results)
    all_results = results + multi_conf

if not results:
    st.warning("No non-B DNA motifs detected in this sequence.")
    st.stop()

df = pd.DataFrame(all_results)

# ----------- Results + Visualization ---------
col_tbl, col_vis = st.columns([1, 1.1])

# ---- Table and download ----
with col_tbl:
    st.markdown("### 🧬 Predicted Non-B DNA Motifs")
    st.dataframe(df.style.background_gradient(cmap="rainbow"), use_container_width=True, hide_index=True)
    with st.expander("Motif Class Summary", expanded=True):
        motif_counts = df["Subtype"].value_counts().reset_index()
        motif_counts.columns = ["Motif Type", "Count"]
        st.dataframe(
            motif_counts.style.background_gradient(cmap="cool"), 
            use_container_width=True, 
            hide_index=True
        )
    col_csv, col_excel = st.columns(2)
    with col_csv:
        csv_download_button(df, "Download Results as CSV")
    with col_excel:
        excel_download_button(df, "Download Results as Excel")

# ---- Linear full-sequence motif visualization ----
with col_vis:
    st.markdown("### 📊 Motif Map (Full Sequence)")
    # Plotly for interactive & colorful bar visualization
    motif_types = sorted(set(r['Subtype'] for r in results))
    color_map = {typ: f"hsl({i*360//max(1,len(motif_types))},70%,55%)" for i, typ in enumerate(motif_types)}

    fig = go.Figure()

    # Draw full sequence as faint bar
    fig.add_trace(go.Bar(
        x=[len(seq)],
        y=["Sequence"],
        orientation='h',
        marker=dict(color='#e0e0e0'),
        width=0.4,
        showlegend=False,
        hoverinfo='skip',
    ))

    # Add motifs as colored overlays
    for r in results:
        start = r['Start']-1
        end = r['End']
        motif_type = r['Subtype']
        color = color_map[motif_type]
        fig.add_trace(go.Bar(
            x=[end-start],
            y=["Sequence"],
            orientation='h',
            base=[start],
            marker=dict(color=color, line=dict(color='black', width=0.8)),
            width=0.4,
            name=motif_type,
            hovertemplate=f"<b>{motif_type}</b><br>Pos: {start+1}-{end}<br>Len: {end-start} bp<br>Score: {r['Propensity/Score']}<extra></extra>",
            showlegend=False,
        ))

    # Legend (separate color boxes)
    for motif_type in color_map:
        fig.add_trace(go.Bar(
            x=[0],
            y=[motif_type],
            marker=dict(color=color_map[motif_type]),
            showlegend=True,
            name=motif_type,
            base=[0],
            orientation='h',
            hoverinfo='skip',
            width=0.3
        ))

    fig.update_layout(
        barmode='overlay',
        height=400,
        margin=dict(t=40, b=30, l=40, r=40),
        xaxis=dict(title='Position (bp)', range=[0, len(seq)], showgrid=False),
        yaxis=dict(showticklabels=True, tickvals=["Sequence"], title=''),
        showlegend=True,
        legend_title="Motif Type",
        plot_bgcolor="#f9f9fc",
        paper_bgcolor="#f9f9fc",
        font=dict(size=15)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Optionally, show first 100-200bp as text with motif highlights
    st.markdown("#### Sequence (first 200bp, motifs highlighted)")
    motif_regions = []
    for r in results:
        motif_regions.append((r['Start']-1, r['End'], color_map[r['Subtype']]))
    motif_regions.sort()
    out_html = ""
    i = 0
    shown = 0
    max_len = min(200, len(seq))
    while i < max_len:
        match = None
        for s, e, c in motif_regions:
            if s <= i < e:
                match = c
                break
        char = seq[i] if seq[i] != "_" else " "
        if match:
            out_html += f"<span style='background:{match};color:#fff;border-radius:2px;padding:0 2px'>{char}</span>"
        else:
            out_html += char
        i += 1
        shown += 1
        if shown % 60 == 0:
            out_html += "<br>"
    st.markdown(f"<div style='font-family:monospace;font-size:1.0em;'>{out_html}</div>", unsafe_allow_html=True)
