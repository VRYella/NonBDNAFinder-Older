# ======================
# utils.py
# ======================

import re
import numpy as np

def parse_fasta(fasta_str: str) -> str:
    lines = fasta_str.strip().splitlines()
    seq = [line.strip() for line in lines if not line.startswith(">")]
    return "".join(seq).upper().replace(" ", "").replace("U", "T")

def wrap(seq: str, width=60) -> str:
    return "\n".join([seq[i:i+width] for i in range(0, len(seq), width)])

def gc_content(seq: str) -> float:
    seq = seq.upper()
    return 100.0 * (seq.count("G") + seq.count("C")) / max(1, len(seq))

def reverse_complement(seq: str) -> str:
    comp = str.maketrans("ATGC", "TACG")
    return seq.translate(comp)[::-1]

def g4hunter_score(seq: str) -> float:
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

def zseeker_score(seq: str) -> float:
    dinucs = re.findall(r"(GC|CG|GT|TG|AC|CA)", seq)
    return len(dinucs) / (len(seq)/2) if len(seq) >= 2 else 0.0

# ======================
# motifs.py
# ======================
import re

def non_overlapping_finditer(pattern, seq):
    """Find non-overlapping matches"""
    regex = re.compile(pattern)
    pos = 0
    while pos < len(seq):
        match = regex.search(seq, pos)
        if not match:
            break
        yield match
        pos = match.end()  # Move past the entire match for non-overlapping

def create_motif_dict(cls, subtype, match, seq, score_method="None", score="0", group=0):
    """Helper to create standardized motif dictionary"""
    sequence = match.group(group)
    return {
        "Class": cls, "Subtype": subtype, "Start": match.start() + 1,
        "End": match.start() + len(sequence), "Length": len(sequence),
        "Sequence": wrap(sequence), "ScoreMethod": score_method, "Score": score
    }

def find_motif(seq, pattern, cls, subtype, score_method="None", score_func=None, group=0):
    """Generic motif finder with non-overlapping logic"""
    results = []
    for m in non_overlapping_finditer(pattern, seq):
        if score_func:
            score = f"{score_func(m.group(group)):.2f}"
        else:
            score = "0"
        results.append(create_motif_dict(cls, subtype, m, seq, score_method, score, group))
    return results

# G-Quadruplex variants
def find_gquadruplex(seq):
    return find_motif(seq, r"(?=(G{3,}([ATGC]{1,7}G{3,}){3}))", 
                     "Quadruplex", "Canonical_G-Quadruplex", "G4Hunter", g4hunter_score)

def find_relaxed_gquadruplex(seq):
    return find_motif(seq, r"(?=(G{3,}(?:[ATGC]{0,12}G{3,}){3}))", 
                     "Quadruplex", "Relaxed_G-Quadruplex", "G4Hunter", g4hunter_score)

def find_bulged_gquadruplex(seq):
    return find_motif(seq, r"(?=(G{3,}[ATGC]{0,3}G{3,}[ATGC]{0,3}G{3,}[ATGC]{0,3}G{3,}))", 
                     "Quadruplex", "Bulged_G-Quadruplex", "G4Hunter (bulge)", g4hunter_score, 1)

def find_bipartite_gquadruplex(seq):
    return find_motif(seq, r"(?=(G{3,}(?:[ATGC]{0,30}G{3,}){3}))", 
                     "Quadruplex", "Bipartite_G-Quadruplex", "G4Hunter", g4hunter_score)

def find_multimeric_gquadruplex(seq):
    return find_motif(seq, r"(?=((G{3,}(?:[ATGC]{0,12}G{3,}){4,})))", 
                     "Quadruplex", "Multimeric_G-Quadruplex", "G4Hunter", g4hunter_score, 1)

def find_imotif(seq):
    return find_motif(seq, r"(?=(C{3,}([ATGC]{1,7}C{3,}){3}))", 
                     "Quadruplex", "i-Motif", "G4Hunter", 
                     lambda x: -g4hunter_score(x.replace('C','G')))

def find_gtriplex(seq):
    return find_motif(seq, r"(?=(G{3,}([ATGC]{1,7}G{3,}){2}))", 
                     "Triplex", "G-Triplex", "G4Hunter", g4hunter_score)

def find_zdna(seq):
    return find_motif(seq, r"(?=((?:CG){6,}))", 
                     "Z-DNA", "CG_Repeat", "ZSeeker", zseeker_score, 1)

# Simple motifs (no scoring)
SIMPLE_MOTIFS = [
    (r"(?=(T{3,}[ATGC]{1,7}A{3,}))", "H-DNA", "T-A"),
    (r"(?=(CTGCTGCTGCTG))", "Sticky_DNA", "CTG"),
    (r"(?=((?:AT){6,}))", "Slipped_DNA", "AT_Slippage"),
    (r"(?=(A{4,}TTTT))", "Cruciform", "A-T"),
    (r"(?=(A{6,7}|T{6,7}))", "Bent_DNA", "Poly-A/T"),
    (r"(?=(?:AAATT){2,})", "A-Phased_Repeat", "APR"),
    (r"(?=(ATCGCGAT))", "Mirror_Repeat", "ATCGCGAT"),
    (r"(?=(G{6,}))", "Direct_Repeat", "Poly-G"),
]

def find_simple_motifs(seq):
    """Find all simple motifs that don't require scoring"""
    results = []
    for pattern, cls, subtype in SIMPLE_MOTIFS:
        group = 1 if pattern.count('(') > 1 else 0
        results.extend(find_motif(seq, pattern, cls, subtype, group=group))
    return results

def find_local_bent(seq):
    return find_motif(seq, r"(?=(A{6,7}|T{6,7}))", "Bent_DNA", "Poly-A/T", group=1)

def find_overlap_hybrid(seq, pattern1, pattern2, cls, subtype):
    """Generic function to find overlapping motifs (non-overlapping within each pattern)"""
    hits = []
    hits1 = [(m.start(), m.end()) for m in non_overlapping_finditer(pattern1, seq)]
    for m in non_overlapping_finditer(pattern2, seq):
        for start1, end1 in hits1:
            if m.start() < end1 and m.end() > start1:
                hits.append(create_motif_dict(cls, subtype, m, seq))
                break
    return hits

def find_quadruplex_triplex_hybrid(seq):
    return find_overlap_hybrid(seq, r"(?=(G{3,}([ATGC]{1,7}G{3,}){3}))", 
                              r"(?=(G{3,}([ATGC]{1,7}G{3,}){2}))", 
                              "Hybrid", "G4-Triplex")

def find_cruciform_triplex_junction(seq):
    return find_overlap_hybrid(seq, r"(?=(A{4,}TTTT))", 
                              r"(?=(G{3,}([ATGC]{1,7}G{3,}){2}))", 
                              "Junction", "Cruciform-Triplex")

def find_g4_imotif_hybrid(seq):
    return find_overlap_hybrid(seq, r"(?=(G{3,}([ATGC]{1,7}G{3,}){3}))", 
                              r"(?=(C{3,}([ATGC]{1,7}C{3,}){3}))", 
                              "Hybrid", "G4-i-Motif")

def find_polyG(seq):
    return find_motif(seq, r"(?=(G{6,}))", "Direct_Repeat", "Poly-G", group=1)

def all_motifs(seq):
    """Find all motifs in sequence"""
    motif_funcs = [
        find_gquadruplex, find_relaxed_gquadruplex, find_bulged_gquadruplex,
        find_imotif, find_gtriplex, find_bipartite_gquadruplex, find_multimeric_gquadruplex,
        find_zdna, find_simple_motifs, find_quadruplex_triplex_hybrid, 
        find_cruciform_triplex_junction, find_g4_imotif_hybrid, find_polyG, find_local_bent
    ]
    return [hit for func in motif_funcs for hit in func(seq)]

def find_hotspots(seq, motif_hits, window=100, min_count=3):
    """Find regions with high motif density"""
    motif_positions = [(hit["Start"], hit["End"]) for hit in motif_hits]
    hotspots = {}
    
    for i in range(1, len(seq) - window + 2):
        region_end = i + window - 1
        count = sum(start <= region_end and end >= i for start, end in motif_positions)
        if count >= min_count:
            hotspots[(i, region_end)] = {"RegionStart": i, "RegionEnd": region_end, "MotifCount": count}
    
    return list(hotspots.values())

# ======================
# app.py
# ======================
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re, io
from datetime import datetime
    all_motifs,
    find_gquadruplex, find_relaxed_gquadruplex, find_bulged_gquadruplex, find_gtriplex,
    find_bipartite_gquadruplex, find_multimeric_gquadruplex,
    find_imotif, find_hotspots
)

EXAMPLE_FASTA = ">Example\nATCGATCGATCGAAAATTTTATTTAAATTTAAATTTGGGTTAGGGTTAGGGTTAGGGCCCCCTCCCCCTCCCCCTCCCC\nATCGATCGCGCGCGCGATCGCACACACACAGCTGCTGCTGCTTGGGAAAGGGGAAGGGTTAGGGAAAGGGGTTT\nGGGTTTAGGGGGGAGGGGCTGCTGCTGCATGCGGGAAGGGAGGGTAGAGGGTCCGGTAGGAACCCCTAACCCCTAA\nGAAAGAAGAAGAAGAAGAAGAAAGGAAGGAAGGAGGAGGAGGAGGAGGAGGAGGAGGAGGAGGAGGAGGAGGG"

st.set_page_config(page_title="Non-B DNA Motif Finder (Non-overlapping)", layout="wide")

# Initialize session state
if 'seq' not in st.session_state:
    st.session_state['seq'] = ""
if 'df' not in st.session_state:
    st.session_state['df'] = pd.DataFrame()
if 'motif_results' not in st.session_state:
    st.session_state['motif_results'] = []
if 'analysis_status' not in st.session_state:
    st.session_state['analysis_status'] = ""
if 'stop_analysis' not in st.session_state:
    st.session_state['stop_analysis'] = False

PAGES = [
    "Home",
    "Upload & Analyze",
    "Results",
    "Visualization",
    "Download",
    "Additional Information",
    "Motif Definitions Glossary"
]

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", PAGES)

st.title("Non-B DNA Motif Finder (Non-overlapping Detection)")

def collect_all_motifs(seq, status_callback=None, stop_flag=None):
    """Collect all motifs using non-overlapping detection"""
    if status_callback:
        status_callback("Scanning for non-B DNA motifs using non-overlapping regex patterns...")
    if stop_flag and stop_flag():
        return []
    return all_motifs(seq)

if page == "Home":
    st.markdown("""
    <style>
    .home-header {
        font-size: 2.5em;
        font-weight: bold;
        color: #1A5276;
        text-align: center;
        margin-bottom: 20px;
    }
    .non-overlap-info {
        background-color: #E8F4FD;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #3498db;
        margin: 20px 0;
    }
    </style>
    <div class='home-header'>Welcome to Non-B DNA Motif Finder</div>
    """, unsafe_allow_html=True)
    
    try:
        st.image("nbd.PNG", use_container_width=True)
    except Exception:
        st.warning("Logo image (nbd.PNG) not found. Place it in the app folder.")
    
    st.markdown("""
    <div class='non-overlap-info'>
    <h3>🔍 Non-overlapping Detection Mode</h3>
    <p>This application uses <strong>non-overlapping</strong> motif detection, which means:</p>
    <ul>
        <li>Each nucleotide position is assigned to at most one motif of the same type</li>
        <li>More biologically realistic representation of motif distribution</li>
        <li>Prevents redundant detection of the same DNA region</li>
        <li>May result in fewer total motifs compared to overlapping detection</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    This application identifies and visualizes various non-B DNA motifs including:
    - **Canonical and non-canonical G-quadruplexes**
    - **i-Motifs, Z-DNA, Cruciforms**
    - **Triplex DNA, Bent DNA, Sticky DNA**
    - **Hybrid structures and hotspot regions**
    
    Upload a sequence or use the example to begin analysis.
    """)

elif page == "Upload & Analyze":
    st.markdown("<h2 style='color:#1A5276;'>Upload & Analyze</h2>", unsafe_allow_html=True)
    
    # File upload
    fasta_file = st.file_uploader("Upload FASTA file", type=["fa", "fasta", "txt"])
    if fasta_file:
        try:
            seq = parse_fasta(fasta_file.read().decode("utf-8"))
            st.session_state['seq'] = seq
            st.success(f"FASTA loaded successfully! Sequence length: {len(seq)} nucleotides")
        except Exception as e:
            st.error(f"Error loading FASTA file: {str(e)}")

    # Example sequence button
    if st.button("Use Example Sequence"):
        st.session_state['seq'] = parse_fasta(EXAMPLE_FASTA)
        st.success(f"Example sequence loaded! Length: {len(st.session_state['seq'])} nucleotides")

    # Text area for sequence input
    seq_input = st.text_area("Paste Sequence (FASTA or Raw)", 
                            value=st.session_state.get('seq', ''), 
                            height=150,
                            help="Enter DNA sequence in FASTA format or as raw nucleotides (A, T, G, C)")
    
    if seq_input:
        try:
            processed_seq = parse_fasta(seq_input)
            st.session_state['seq'] = processed_seq
            st.info(f"Sequence processed. Length: {len(processed_seq)} nucleotides")
        except Exception as e:
            st.error(f"Invalid sequence format: {str(e)}")

    # Analysis parameters
    st.subheader("Analysis Parameters")
    col1, col2 = st.columns(2)
    with col1:
        hotspot_window = st.number_input("Hotspot window size (bp)", min_value=50, max_value=500, value=100)
    with col2:
        min_motif_count = st.number_input("Minimum motifs for hotspot", min_value=2, max_value=10, value=3)

    # Run analysis
    if st.button("🔍 Run Non-overlapping Motif Analysis", type="primary"):
        seq = st.session_state.get('seq', '')
        if not seq:
            st.error("Please input a DNA sequence first.")
        elif not re.match("^[ATGC]+$", seq.upper()):
            st.error("Please input a valid DNA sequence containing only A, T, G, C nucleotides.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("Running non-overlapping motif analysis..."):
                try:
                    # Update progress
                    progress_bar.progress(20)
                    status_text.text("Initializing motif detection...")
                    
                    # Run motif analysis
                    results = collect_all_motifs(seq.upper())
                    progress_bar.progress(80)
                    status_text.text("Processing results...")
                    
                    # Store results
                    st.session_state['motif_results'] = results
                    st.session_state['df'] = pd.DataFrame(results)
                    st.session_state['hotspot_params'] = {
                        'window': hotspot_window,
                        'min_count': min_motif_count
                    }
                    
                    progress_bar.progress(100)
                    status_text.text("Analysis complete!")
                    
                    if results:
                        st.success(f"✅ Found {len(results)} non-overlapping motifs in sequence of {len(seq)} nucleotides")
                        
                        # Quick summary
                        df = pd.DataFrame(results)
                        motif_counts = df['Class'].value_counts()
                        st.subheader("Quick Summary")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Motifs", len(results))
                        with col2:
                            st.metric("Motif Types", len(df['Subtype'].unique()))
                        with col3:
                            st.metric("Sequence Coverage", f"{(df['Length'].sum()/len(seq)*100):.1f}%")
                        
                        st.write("**Motif Class Distribution:**")
                        for motif_class, count in motif_counts.items():
                            st.write(f"- {motif_class}: {count}")
                            
                    else:
                        st.warning("⚠️ No motifs found in the provided sequence.")
                        
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                finally:
                    progress_bar.empty()
                    status_text.empty()

elif page == "Results":
    st.markdown("<h2 style='color:#1A5276;'>Detected Motifs (Non-overlapping)</h2>", unsafe_allow_html=True)
    df = st.session_state.get('df', pd.DataFrame())
    
    if df.empty:
        st.info("No results available. Please run analysis first.")
    else:
        # Display summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Motifs", len(df))
        with col2:
            st.metric("Unique Types", len(df['Subtype'].unique()))
        with col3:
            st.metric("Avg Length", f"{df['Length'].mean():.1f} bp")
        with col4:
            seq_len = len(st.session_state.get('seq', ''))
            coverage = (df['Length'].sum() / seq_len * 100) if seq_len > 0 else 0
            st.metric("Coverage", f"{coverage:.1f}%")

        # Filter options
        st.subheader("Filter Results")
        col1, col2 = st.columns(2)
        with col1:
            selected_classes = st.multiselect("Filter by Class", 
                                            options=sorted(df['Class'].unique()),
                                            default=sorted(df['Class'].unique()))
        with col2:
            selected_subtypes = st.multiselect("Filter by Subtype",
                                             options=sorted(df['Subtype'].unique()),
                                             default=sorted(df['Subtype'].unique()))

        # Apply filters
        filtered_df = df[
            (df['Class'].isin(selected_classes)) & 
            (df['Subtype'].isin(selected_subtypes))
        ]

        # Display filtered results
        st.subheader(f"Motif Results ({len(filtered_df)} of {len(df)} motifs)")
        st.dataframe(
            filtered_df[['Class', 'Subtype', 'Start', 'End', 'Length', 'Sequence', 'ScoreMethod', 'Score']], 
            use_container_width=True
        )

        # Motif distribution chart
        st.subheader("Motif Type Distribution")
        counts = filtered_df['Subtype'].value_counts()
        
        if len(counts) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            counts.plot(kind='bar', ax=ax, color='skyblue')
            ax.set_xlabel("Motif Type")
            ax.set_ylabel("Count")
            ax.set_title("Distribution of Non-overlapping Motifs")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No motifs to display with current filters.")

        # Hotspot analysis
        st.subheader("Hotspot Regions")
        params = st.session_state.get('hotspot_params', {'window': 100, 'min_count': 3})
        
        if st.session_state.get('seq'):
            hotspots = find_hotspots(
                st.session_state['seq'], 
                st.session_state['motif_results'], 
                window=params['window'], 
                min_count=params['min_count']
            )
            
            if hotspots:
                st.success(f"Found {len(hotspots)} hotspot regions")
                hotspot_df = pd.DataFrame(hotspots)
                st.dataframe(hotspot_df, use_container_width=True)
            else:
                st.info(f"No hotspot regions found with ≥{params['min_count']} motifs in {params['window']} bp windows.")

elif page == "Visualization":
    st.markdown("<h2 style='color:#1A5276;'>Motif Visualization</h2>", unsafe_allow_html=True)
    df = st.session_state.get('df', pd.DataFrame())
    seq = st.session_state.get('seq', '')
    
    if df.empty:
        st.info("No motifs to visualize. Please run analysis first.")
    else:
        # Visualization options
        st.subheader("Visualization Options")
        col1, col2 = st.columns(2)
        with col1:
            viz_classes = st.multiselect("Select classes to visualize",
                                       options=sorted(df['Class'].unique()),
                                       default=sorted(df['Class'].unique()))
        with col2:
            show_sequence_ruler = st.checkbox("Show sequence ruler", value=True)

        # Filter data for visualization
        viz_df = df[df['Class'].isin(viz_classes)] if viz_classes else df

        if not viz_df.empty:
            # Create motif map
            motif_types = sorted(viz_df['Subtype'].unique())
            color_palette = sns.color_palette('husl', n_colors=len(motif_types))
            color_map = {typ: color_palette[i] for i, typ in enumerate(motif_types)}
            y_map = {typ: i+1 for i, typ in enumerate(motif_types)}

            fig, ax = plt.subplots(figsize=(12, max(len(motif_types)*0.8, 4)))
            
            # Plot motifs
            for _, motif in viz_df.iterrows():
                motif_type = motif['Subtype']
                y = y_map[motif_type]
                color = color_map[motif_type]
                ax.barh(y, motif['End'] - motif['Start'] + 1, 
                       left=motif['Start'], height=0.6, 
                       color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
                
                # Add motif label if bar is wide enough
                if motif['Length'] > len(seq) * 0.05:  # Only label if >5% of sequence
                    ax.text(motif['Start'] + motif['Length']/2, y, 
                           f"{motif['Length']}bp", 
                           ha='center', va='center', fontsize=8, fontweight='bold')

            # Formatting
            ax.set_yticks(list(y_map.values()))
            ax.set_yticklabels(list(y_map.keys()))
            ax.set_xlim(0, len(seq))
            ax.set_xlabel('Position on Sequence (bp)')
            ax.set_title(f'Non-overlapping Motif Map (Sequence length: {len(seq)} bp)')
            
            # Add sequence ruler if requested
            if show_sequence_ruler:
                ax2 = ax.twiny()
                ax2.set_xlim(0, len(seq))
                ax2.set_xlabel('Position (bp)')
                ruler_ticks = range(0, len(seq)+1, max(1, len(seq)//10))
                ax2.set_xticks(ruler_ticks)
            
            plt.tight_layout()
            st.pyplot(fig)

            # Summary statistics
            st.subheader("Visualization Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Motifs Shown", len(viz_df))
            with col2:
                total_coverage = viz_df['Length'].sum()
                st.metric("Total Coverage", f"{total_coverage} bp")
            with col3:
                coverage_percent = (total_coverage / len(seq) * 100) if len(seq) > 0 else 0
                st.metric("Coverage %", f"{coverage_percent:.1f}%")

        else:
            st.info("No motifs selected for visualization.")

elif page == "Download":
    st.markdown("<h2 style='color:#1A5276;'>Download Results</h2>", unsafe_allow_html=True)
    df = st.session_state.get('df', pd.DataFrame())
    
    if df.empty:
        st.info("No results to download. Please run analysis first.")
    else:
        st.write(f"**Available data:** {len(df)} motifs detected using non-overlapping algorithm")
        
        # Prepare download data
        download_df = df.copy()
        download_df['Analysis_Method'] = 'Non-overlapping'
        download_df['Analysis_Date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # CSV download
        csv = download_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Download CSV",
            data=csv,
            file_name=f"non_overlapping_motifs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

        # Excel download
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            download_df.to_excel(writer, sheet_name='Motifs', index=False)
            
            # Add summary sheet
            summary_data = {
                'Metric': ['Total Motifs', 'Unique Types', 'Average Length', 'Total Coverage'],
                'Value': [
                    len(df),
                    len(df['Subtype'].unique()),
                    f"{df['Length'].mean():.1f} bp",
                    f"{df['Length'].sum()} bp"
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
        st.download_button(
            label="📊 Download Excel",
            data=excel_buffer.getvalue(),
            file_name=f"non_overlapping_motifs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

elif page == "Additional Information":
    st.markdown("<h2 style='color:#1A5276;'>Additional Information</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    ### About Non-overlapping Detection
    
    This application implements **non-overlapping motif detection**, which provides several advantages:
    
    - **Biological Realism**: Prevents the same DNA region from being counted multiple times
    - **Cleaner Results**: Reduces redundancy in motif identification
    - **Better Visualization**: Motif maps are easier to interpret without overlapping features
    - **Accurate Statistics**: Coverage calculations reflect actual sequence usage
    
    ### Algorithm Details
    
    The non-overlapping algorithm works by:
    1. Scanning the sequence from 5' to 3' direction
    2. When a motif is found, the search continues after the end of that motif
    3. This ensures no nucleotide is part of multiple motifs of the same type
    4. Different motif types can still overlap (e.g., G4 and triplex hybrid detection)
    
    ### Reference Information
    
    This app detects a comprehensive array of non-B DNA motifs using validated algorithms and patterns.
    
    **Development Team:**
    - Dr. Venkata Rajesh Yella
    - Chandrika Gummadi
    
    **Source Code:** Available on [GitHub](https://github.com/VRYella/Non-B-DNA-Finder)
    
    **Citation:** If you use this tool in your research, please cite our work appropriately.
    """)

elif page == "Motif Definitions Glossary":
    st.markdown("<h2 style='color:#1A5276;'>Motif Definitions Glossary</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    ### Primary Motif Types
    
    **G-Quadruplex (G4):** Four runs of guanines (G≥3) forming a square planar structure stabilized by Hoogsteen hydrogen bonding. Variants include:
    - *Canonical*: Standard G4 with loops 0-7 nucleotides
    - *Relaxed*: Extended loops up to 12 nucleotides  
    - *Bulged*: Allows interruptions in G-runs
    - *Bipartite*: Long loops up to 30 nucleotides
    - *Multimeric*: Multiple G4 units in sequence
    
    **i-Motif:** Cytosine-rich regions (C≥3) that form intercalated four-stranded structures, particularly stable at acidic pH.
    
    **Z-DNA:** Left-handed DNA helix formed by alternating purine-pyrimidine sequences, especially CG repeats.
    
    **Triplex DNA:** Three-stranded DNA structures formed by homopurine or homopyrimidine sequences.
    
    ### Secondary Structures
    
    **Cruciform:** Hairpin-like structures formed at palindromic sequences, creating a cross-shaped configuration.
    
    **Sticky DNA:** Repetitive purine/pyrimidine tracts that can form unusual structures through self-complementarity.
    
    **Bent DNA:** Poly-A or Poly-T tracts (≥6 nucleotides) that introduce local bending in the DNA helix.
    
    **Slipped DNA:** AT-rich repeats that can form slipped-strand structures during replication.
    
    ### Specialized Motifs
    
    **A-Phased Repeats (APR):** AAATT repeats associated with nucleosome positioning and chromatin structure.
    
    **Mirror Repeats:** Symmetric sequences that can form unusual structures within the same DNA strand.
    
    **H-DNA:** Structures formed by homopurine/homopyrimidine sequences under specific conditions.
    
    ### Hybrid Structures
    
    **G4-Triplex Hybrid:** Co-localized G-quadruplex and triplex-forming sequences.
    
    **G4-i-Motif Hybrid:** Complementary G4 and i-motif sequences that may interact.
    
    **Cruciform-Triplex Junction:** Regions where cruciform and triplex structures may coexist.
    
    ### Scoring Methods
    
    - **G4Hunter**: Quantitative scoring for G-quadruplex formation potential
    - **ZSeeker**: Scoring algorithm for Z-DNA formation likelihood
    - **Non-scored**: Motifs identified by sequence pattern matching only
    
    ### Important Notes for Non-overlapping Detection
    
    - Each nucleotide position contributes to at most one motif of each type
    - Hybrid detection still allows different motif types to overlap
    - Results may show fewer total motifs compared to overlapping methods
    - More accurately represents biological motif distribution
    """)
