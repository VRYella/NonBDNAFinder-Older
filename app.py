import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import io
from datetime import datetime

# ------------- MOTIF DEFINITIONS AND SCORING REFERENCES --------------

MOTIF_INFO = [
    # (Class, Subtype, Description, Score Method, Reference)
    ("Quadruplex", "G-Quadruplex", "Four runs of ≥3 Gs separated by loops of 1–7 nt.", "G4Hunter Score", "Bedrat et al., 2016 (NAR)"),
    ("Quadruplex", "G-Triplex", "Three runs of ≥3 Gs separated by loops of 1–7 nt (not G4).", "Scaled G4Hunter", "Chen et al., 2018 (NAR)"),
    ("Quadruplex", "Bipartite_G-Quadruplex", "Two canonical G4s separated by ≤100 nt.", "Avg. G4Hunter", "Matsugami et al., 2001; Siddiqui-Jain et al., 2002"),
    ("Quadruplex", "i-Motif", "C-quadruplex: Four runs of ≥3 Cs separated by 1–7 nt.", "iM Hunter (neg. G4H)", "Abou Assi et al., 2018 (NAR)"),
    ("Z-DNA", "Z-DNA", "≥12 bp alternating purine-pyrimidine (GC, CG, GT, TG, AC, CA)", "Z-Hunt Z-score", "Ho et al., 1999 (Annu Rev Biophys)"),
    ("Triplex", "H-DNA", "Mirror repeat: ≥10 bp purine/pyrimidine, short spacer.", "NA", "Buske et al., 2011 (NAR)"),
    ("Triplex", "Sticky_DNA", "≥5 GAA/TTC repeats.", "Repeat count", "Sakamoto et al., 1999 (Science)"),
    ("Direct Repeat", "Slipped_DNA", "Direct repeats, 10–300 bp, ≤10 bp spacer.", "Repeat count", "Wells et al., 1988 (Science)"),
    ("Inverted Repeat", "Cruciform_DNA", "Inverted repeats, arms ≥6 bp, loop ≤100 bp.", "Arm length", "Pearson et al., 1996 (NAR)"),
    ("Bent DNA", "APR", "A-tract periodicity: 3+ A-tracts (3–11 bp) spaced 10–11 bp apart.", "NA", "Marini et al., 1982 (Nature)"),
    ("Bent DNA", "Bent_DNA", "A-tract/T-tract periodicity.", "Tract length", "Marini et al., 1982 (Nature)"),
    ("Quadruplex-Triplex Hybrid", "Quadruplex-Triplex_Hybrid", "G4 adjacent to triplex motif.", "G4Hunter, NA", "Siddiqui-Jain et al., 2002 (Nature)"),
    ("Cruciform-Triplex Junction", "Cruciform-Triplex_Junctions", "Junction of cruciform and triplex.", "NA", "Cer et al., 2011 (NAR)"),
    ("G-Quadruplex_i-Motif_Hybrid", "G-Quadruplex_i-Motif_Hybrid", "G4 and i-motif nearby.", "Min G4/iM", "Abou Assi et al., 2018 (NAR)"),
    ("Local Bend", "A-Tract Local Bend", "A6-7/T6-7 stretches.", "NA", "Goodsell & Dickerson, 1994 (NAR)"),
    ("Local Bend", "CA Dinucleotide Bend", "CA repeats (≥4 units).", "NA", "Haranczyk et al., 2010 (J Struct Biol)"),
    ("Local Bend", "TG Dinucleotide Bend", "TG repeats (≥4 units).", "NA", "Haranczyk et al., 2010 (J Struct Biol)"),
]

EXAMPLE_FASTA = """>Example
ATCGATCGATCGAAAATTTTATTTAAATTTAAATTTGGGTTAGGGTTAGGGTTAGGGCCCCCTCCCCCTCCCCCTCCCC
ATCGATCGCGCGCGCGATCGCACACACACAGCTGCTGCTGCTTGGGAAAGGGGAAGGGTTAGGGAAAGGGGTTT
GGGTTTAGGGGGGAGGGGCTGCTGCTGCATGCGGGAAGGGAGGGTAGAGGGTCCGGTAGGAACCCCTAACCCCTAA
GAAAGAAGAAGAAGAAGAAGAAAGGAAGGAAGGAGGAGGAGGAGGAGGAGGAGGAGGAGGAGGAGGAGGAGGG
"""

# --------- FASTA & SEQUENCE HANDLING ----------
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

# ------------ MOTIF REGEX FUNCTIONS -----------

def find_g_quadruplex(seq):
    # G4: Four runs of ≥3 Gs, 1-7 nt loops
    pattern = r"(G{3,})([ATGC]{1,7})(G{3,})([ATGC]{1,7})(G{3,})([ATGC]{1,7})(G{3,})"
    matches = []
    for m in re.finditer(pattern, seq):
        region = seq[m.start():m.end()]
        matches.append(dict(
            Class="Quadruplex",
            Subtype="G-Quadruplex",
            Start=m.start()+1, End=m.end(), Length=len(region),
            Sequence=wrap(region),
            ScoreMethod="G4Hunter Score",
            Reference="Bedrat et al., 2016",
            Score=g4hunter_score(region)
        ))
    return matches

def find_g_triplex(seq, g4_locs):
    # G-triplex: Three runs of ≥3 Gs, 1-7 nt loops, but NOT overlapping a G4
    pattern = r"(G{3,})([ATGC]{1,7})(G{3,})([ATGC]{1,7})(G{3,})"
    matches = []
    n = len(seq)
    for m in re.finditer(pattern, seq):
        overlap_g4 = any((m.start() < g4['End'] and m.end() > g4['Start']) for g4 in g4_locs)
        if overlap_g4:
            continue
        region = seq[m.start():m.end()]
        matches.append(dict(
            Class="Quadruplex",
            Subtype="G-Triplex",
            Start=m.start()+1, End=m.end(), Length=len(region),
            Sequence=wrap(region),
            ScoreMethod="Scaled G4Hunter",
            Reference="Chen et al., 2018",
            Score=round(g4hunter_score(region)*0.7, 2)
        ))
    return matches

def find_bipartite_g4(seq):
    # Bipartite G4: Two canonical G4s (as above), separated by ≤100 nt
    g4_pattern = r"(G{3,})([ATGC]{1,7})(G{3,})([ATGC]{1,7})(G{3,})([ATGC]{1,7})(G{3,})"
    matches = []
    g4_locs = [m for m in re.finditer(g4_pattern, seq)]
    for i, m1 in enumerate(g4_locs):
        for m2 in g4_locs[i+1:]:
            if 0 < (m2.start() - m1.end()) <= 100:
                region = seq[m1.start():m2.end()]
                matches.append(dict(
                    Class="Quadruplex",
                    Subtype="Bipartite_G-Quadruplex",
                    Start=m1.start()+1, End=m2.end(), Length=len(region),
                    Sequence=wrap(region),
                    ScoreMethod="Avg. G4Hunter",
                    Reference="Matsugami et al., 2001; Siddiqui-Jain et al., 2002",
                    Score=round((g4hunter_score(seq[m1.start():m1.end()]) + g4hunter_score(seq[m2.start():m2.end()]))/2, 2)
                ))
    return matches

def find_i_motif(seq):
    # i-Motif: Four runs of ≥3 Cs, 1-7 nt loops
    pattern = r"(C{3,})([ATGC]{1,7})(C{3,})([ATGC]{1,7})(C{3,})([ATGC]{1,7})(C{3,})"
    matches = []
    for m in re.finditer(pattern, seq):
        region = seq[m.start():m.end()]
        matches.append(dict(
            Class="Quadruplex",
            Subtype="i-Motif",
            Start=m.start()+1, End=m.end(), Length=len(region),
            Sequence=wrap(region),
            ScoreMethod="iM Hunter (neg. G4H)",
            Reference="Abou Assi et al., 2018",
            Score=-g4hunter_score(region.replace("C","G").replace("G","C"))
        ))
    return matches

def find_zdna(seq):
    # Z-DNA: sliding 12-bp window, score by Z-Hunt table, threshold >6 is Z-DNA
    z_table = {"CG": 2.0, "GC": 1.8, "GT": 0.6, "TG": 0.6, "CA": 0.5, "AC": 0.5,
               "TA": 0.0, "AT": 0.0, "AA": -1.0, "TT": -1.0, "CC": -1.0, "GG": -1.0,
               "AG": -1.0, "GA": -1.0, "CT": -1.0, "TC": -1.0}
    matches = []
    window = 12
    seq = seq.upper()
    for i in range(len(seq)-window+1):
        win = seq[i:i+window]
        score = sum(z_table.get(win[j:j+2], 0.0) for j in range(window-1))
        if score > 6:
            region = seq[i:i+window]
            matches.append(dict(
                Class="Z-DNA",
                Subtype="Z-DNA",
                Start=i+1, End=i+window, Length=window,
                Sequence=wrap(region),
                ScoreMethod="Z-Hunt Z-score",
                Reference="Ho et al., 1999",
                Score=round(score,2)
            ))
    return matches

def find_hdna(seq):
    # H-DNA: mirror repeats ≥10bp, arms of A/G or C/T, max loop 8bp (see Buske)
    pattern = r"(([AG]{10,}|[CT]{10,}))([ATGC]{0,8})\1"
    matches = []
    for m in re.finditer(pattern, seq):
        region = seq[m.start():m.end()]
        matches.append(dict(
            Class="Triplex",
            Subtype="H-DNA",
            Start=m.start()+1, End=m.end(), Length=len(region),
            Sequence=wrap(region),
            ScoreMethod="NA",
            Reference="Buske et al., 2011",
            Score="NA"
        ))
    return matches

def find_sticky(seq):
    # Sticky DNA: (GAA)≥5 or (TTC)≥5
    pattern = r"(GAA){5,}|(TTC){5,}"
    matches = []
    for m in re.finditer(pattern, seq):
        region = seq[m.start():m.end()]
        matches.append(dict(
            Class="Triplex",
            Subtype="Sticky_DNA",
            Start=m.start()+1, End=m.end(), Length=len(region),
            Sequence=wrap(region),
            ScoreMethod="Repeat count",
            Reference="Sakamoto et al., 1999",
            Score=region.count("GAA") + region.count("TTC")
        ))
    return matches

def find_slipped(seq):
    # Direct repeats, unit 10–300bp, spacer ≤10bp
    matches = []
    for unit_len in range(10, 301):
        for m in re.finditer(rf"([ATGC]{{{unit_len}}})([ATGC]{{0,10}})\1", seq):
            region = seq[m.start():m.end()]
            matches.append(dict(
                Class="Direct Repeat",
                Subtype="Slipped_DNA",
                Start=m.start()+1, End=m.end(), Length=len(region),
                Sequence=wrap(region),
                ScoreMethod="Repeat count",
                Reference="Wells et al., 1988",
                Score=2
            ))
    return matches

def reverse_complement(seq):
    comp = str.maketrans("ATGC", "TACG")
    return seq.translate(comp)[::-1]

def find_cruciform(seq):
    # Inverted repeats, arms ≥6bp, loop ≤100bp
    matches = []
    n = len(seq)
    for arm in range(6, 21):  # arms 6-20 bp for speed
        for loop in range(0, 101):
            pattern = rf"([ATGC]{{{arm}}})([ATGC]{{0,{loop}}})([ATGC]{{{arm}}})"
            for m in re.finditer(pattern, seq):
                left = m.group(1)
                right = m.group(3)
                # Is right arm the reverse-complement of left?
                if reverse_complement(left) == right:
                    region = seq[m.start():m.end()]
                    matches.append(dict(
                        Class="Inverted Repeat",
                        Subtype="Cruciform_DNA",
                        Start=m.start()+1, End=m.end(), Length=len(region),
                        Sequence=wrap(region),
                        ScoreMethod="Arm length",
                        Reference="Pearson et al., 1996",
                        Score=arm
                    ))
    return matches

def find_apr(seq):
    # APR: 3+ A-tracts of length 3-11bp, centers 10-11bp apart
    matches = []
    pattern = r"(A{3,11})[ATGC]{7,9}(A{3,11})[ATGC]{7,9}(A{3,11})"
    for m in re.finditer(pattern, seq):
        region = seq[m.start():m.end()]
        matches.append(dict(
            Class="Bent DNA",
            Subtype="APR",
            Start=m.start()+1, End=m.end(), Length=len(region),
            Sequence=wrap(region),
            ScoreMethod="NA",
            Reference="Marini et al., 1982",
            Score="NA"
        ))
    return matches

def find_bent(seq):
    # Bent DNA: A-tract/T-tract periodicity (e.g., AAAAxxxxAAAA)
    matches = []
    pattern = r"(A{4,6}|T{4,6})([ATGC]{7,11})(A{4,6}|T{4,6})"
    for m in re.finditer(pattern, seq):
        region = seq[m.start():m.end()]
        matches.append(dict(
            Class="Bent DNA",
            Subtype="Bent_DNA",
            Start=m.start()+1, End=m.end(), Length=len(region),
            Sequence=wrap(region),
            ScoreMethod="Tract length",
            Reference="Marini et al., 1982",
            Score=max(len(x) for x in re.findall(r"A{4,6}|T{4,6}", region))
        ))
    return matches

def find_local_bends(seq):
    # A-Tract: A{6,7}
    pattern_a = r"A{6,7}"
    pattern_ca = r"(CA){4,}"
    pattern_tg = r"(TG){4,}"
    matches = []
    for m in re.finditer(pattern_a, seq):
        region = seq[m.start():m.end()]
        matches.append(dict(
            Class="Local Bend",
            Subtype="A-Tract Local Bend",
            Start=m.start()+1, End=m.end(), Length=len(region),
            Sequence=wrap(region),
            ScoreMethod="NA",
            Reference="Goodsell & Dickerson, 1994",
            Score="NA"
        ))
    for m in re.finditer(pattern_ca, seq):
        region = seq[m.start():m.end()]
        matches.append(dict(
            Class="Local Bend",
            Subtype="CA Dinucleotide Bend",
            Start=m.start()+1, End=m.end(), Length=len(region),
            Sequence=wrap(region),
            ScoreMethod="NA",
            Reference="Haranczyk et al., 2010",
            Score="NA"
        ))
    for m in re.finditer(pattern_tg, seq):
        region = seq[m.start():m.end()]
        matches.append(dict(
            Class="Local Bend",
            Subtype="TG Dinucleotide Bend",
            Start=m.start()+1, End=m.end(), Length=len(region),
            Sequence=wrap(region),
            ScoreMethod="NA",
            Reference="Haranczyk et al., 2010",
            Score="NA"
        ))
    return matches

def find_g4_triplex_hybrid(seq):
    # Hybrid: G4 adjacent (within 10bp) to triplex
    g4s = find_g_quadruplex(seq)
    hdnas = find_hdna(seq)
    matches = []
    for g4 in g4s:
        for h in hdnas:
            if abs(g4['End'] - h['Start']) <= 10 or abs(h['End'] - g4['Start']) <= 10:
                region = seq[min(g4['Start'], h['Start'])-1:max(g4['End'], h['End'])]
                matches.append(dict(
                    Class="Quadruplex-Triplex Hybrid",
                    Subtype="Quadruplex-Triplex_Hybrid",
                    Start=min(g4['Start'], h['Start']), End=max(g4['End'], h['End']),
                    Length=max(g4['End'], h['End'])-min(g4['Start'], h['Start'])+1,
                    Sequence=wrap(region),
                    ScoreMethod=f"{g4['ScoreMethod']}, {h['ScoreMethod']}",
                    Reference=f"{g4['Reference']}; {h['Reference']}",
                    Score=f"{g4['Score']}, {h['Score']}"
                ))
    return matches

def find_cruciform_triplex_junction(seq):
    cruciforms = find_cruciform(seq)
    hdnas = find_hdna(seq)
    matches = []
    for c in cruciforms:
        for h in hdnas:
            if abs(c['End'] - h['Start']) <= 10 or abs(h['End'] - c['Start']) <= 10:
                region = seq[min(c['Start'], h['Start'])-1:max(c['End'], h['End'])]
                matches.append(dict(
                    Class="Cruciform-Triplex Junction",
                    Subtype="Cruciform-Triplex_Junctions",
                    Start=min(c['Start'], h['Start']), End=max(c['End'], h['End']),
                    Length=max(c['End'], h['End'])-min(c['Start'], h['Start'])+1,
                    Sequence=wrap(region),
                    ScoreMethod=f"{c['ScoreMethod']}, {h['ScoreMethod']}",
                    Reference=f"{c['Reference']}; {h['Reference']}",
                    Score=f"{c['Score']}, {h['Score']}"
                ))
    return matches

def find_g4_imotif_hybrid(seq):
    g4s = find_g_quadruplex(seq)
    ims = find_i_motif(seq)
    matches = []
    for g4 in g4s:
        for im in ims:
            if abs(g4['End'] - im['Start']) <= 10 or abs(im['End'] - g4['Start']) <= 10:
                region = seq[min(g4['Start'], im['Start'])-1:max(g4['End'], im['End'])]
                matches.append(dict(
                    Class="G-Quadruplex_i-Motif_Hybrid",
                    Subtype="G-Quadruplex_i-Motif_Hybrid",
                    Start=min(g4['Start'], im['Start']), End=max(g4['End'], im['End']),
                    Length=max(g4['End'], im['End'])-min(g4['Start'], im['Start'])+1,
                    Sequence=wrap(region),
                    ScoreMethod=f"{g4['ScoreMethod']}, {im['ScoreMethod']}",
                    Reference=f"{g4['Reference']}; {im['Reference']}",
                    Score=f"{g4['Score']}, {im['Score']}"
                ))
    return matches

# ---------- MOTIF SCORING ---------------

def g4hunter_score(seq):
    # G4Hunter: score per Bedrat et al., 2016
    seq = seq.upper()
    vals = []
    n = len(seq)
    i = 0
    while i < n:
        if seq[i] == 'G':
            run = 1
            while i+run < n and seq[i+run] == 'G':
                run += 1
            score = min(run, 4)
            vals += [score]*run
            i += run
        elif seq[i] == 'C':
            run = 1
            while i+run < n and seq[i+run] == 'C':
                run += 1
            score = -min(run, 4)
            vals += [score]*run
            i += run
        else:
            vals.append(0)
            i += 1
    return round(np.mean(vals),2) if vals else 0.0

# ------------- MAIN STREAMLIT APP ---------------

st.set_page_config(page_title="Non-B DNA Motif Finder", layout="wide")
pages = ["Home", "Upload & Analyze", "Results", "Visualization", "Download Report", "About", "Contact"]
page = st.sidebar.radio("Navigation", pages)

if page == "Home":
    st.title("Non-B DNA Motif Finder (Research-Ready)")
    st.image("nbd.PNG", use_container_width=True)
    st.markdown("**Rapid, motif-by-motif, scientifically referenced non-B DNA motif discovery for research and genomics.**")
    st.subheader("Motif Classes, Scoring, and References")
    for cl, subtype, expl, score, ref in MOTIF_INFO:
        st.markdown(f"- **{cl} / {subtype}**: {expl}<br> Scoring: `{score}`<br> Reference: {ref}", unsafe_allow_html=True)

elif page == "Upload & Analyze":
    st.header("Upload or Paste Sequence")
    col1, col2 = st.columns(2)
    with col1:
        fasta_file = st.file_uploader("Upload FASTA file", type=["fa", "fasta", "txt"])
        if fasta_file:
            try:
                seq = parse_fasta(fasta_file.read().decode("utf-8"))
                st.session_state['seq'] = seq
                st.success("FASTA loaded!")
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
                st.error("Paste a valid sequence.")
    if st.button("Run Analysis"):
        seq = st.session_state.get('seq', "")
        if not seq or not re.match("^[ATGC]+$", seq):
            st.error("Please upload or paste a valid DNA sequence (A/T/G/C only).")
        else:
            with st.spinner("Analyzing..."):
                g4s = find_g_quadruplex(seq)
                gtriplex = find_g_triplex(seq, g4s)
                bipartite = find_bipartite_g4(seq)
                imotifs = find_i_motif(seq)
                zdna = find_zdna(seq)
                hdna = find_hdna(seq)
                sticky = find_sticky(seq)
                slipped = find_slipped(seq)
                cruciform = find_cruciform(seq)
                apr = find_apr(seq)
                bent = find_bent(seq)
                localbend = find_local_bends(seq)
                g4tri = find_g4_triplex_hybrid(seq)
                cruci_tri = find_cruciform_triplex_junction(seq)
                g4imotif = find_g4_imotif_hybrid(seq)
                results = (g4s + gtriplex + bipartite + imotifs + zdna + hdna +
                           sticky + slipped + cruciform + apr + bent + localbend +
                           g4tri + cruci_tri + g4imotif)
                df = pd.DataFrame(results)
                if not df.empty:
                    df['GC (%)'] = df['Sequence'].apply(lambda x: gc_content(x.replace("\n", "")))
                    df['Score'] = df['Score'].astype(str)
                    df['ScoreMethod'] = df['ScoreMethod'].astype(str)
                st.session_state['motif_df'] = df
            if df.empty:
                st.warning("No motifs detected in this sequence.")
            else:
                st.success(f"Detected {len(df)} motif region(s). See 'Results' for details.")

elif page == "Results":
    st.header("Motif Detection Results")
    df = st.session_state.get('motif_df', pd.DataFrame())
    if df.empty:
        st.info("No results yet. Go to 'Upload & Analyze' and run analysis.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)
        with st.expander("Motif Class Summary"):
            motif_counts = df["Subtype"].value_counts().reset_index()
            motif_counts.columns = ["Motif Type", "Count"]
            st.dataframe(motif_counts, use_container_width=True, hide_index=True)
        with st.expander("Scoring System References Table"):
            score_refs = pd.DataFrame([
                {"Class": cl, "Subtype": subtype, "Scoring": score, "Reference": ref}
                for cl, subtype, _, score, ref in MOTIF_INFO
            ])
            st.dataframe(score_refs, use_container_width=True, hide_index=True)

elif page == "Visualization":
    st.header("Motif Visualization")
    df = st.session_state.get('motif_df', pd.DataFrame())
    seq = st.session_state.get('seq', "")
    if df.empty:
        st.info("No results to visualize. Run analysis first.")
    else:
        st.subheader("Motif Map")
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
        ax.set_title('Motif Map')
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

elif page == "Download Report":
    st.header("Download Motif Report")
    df = st.session_state.get('motif_df', pd.DataFrame())
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
    **Non-B DNA Motif Finder**: Research-ready, scientifically referenced, and rapid detection/visualization of non-canonical DNA structures (non-B DNA motifs).
    - Supports G-quadruplex, triplex, Z-DNA, cruciforms, APR/bent DNA, and more.
    - Implements **published scoring systems** (G4Hunter, Z-Hunt, etc) where available.
    - Accepts FASTA files or direct sequence input.
    - Visualizes results and offers export options.
    - Created for genomics research and bioinformatics.
    """)

elif page == "Contact":
    st.header("Contact")
    st.markdown("""
    For questions, bug reports, or feedback:

    - Email: [your_email@example.com](mailto:your_email@example.com)
    - GitHub: [Non-B DNA Finder](https://github.com/VRYella/NonBDNAFinder)
    - Developed by: Dr. Venkata Rajesh Yella & Aruna Sesha Chandrika Gummadi

    _Thank you for using Non-B DNA Motif Finder!_
    """)
