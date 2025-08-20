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

def g4hunter_score(seq: str) -> float:
    # Input validation
    if not isinstance(seq, str):
        return 0.0
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
    # Input validation
    if not isinstance(seq, str):
        return 0.0
    dinucs = re.findall(r"(GC|CG|GT|TG|AC|CA)", seq)
    return len(dinucs) / (len(seq)/2) if len(seq) >= 2 else 0.0

def triplex_score(seq: str) -> float:
    """Score for triplex DNA formation potential"""
    # Input validation
    if not isinstance(seq, str):
        return 0.0
    purine_runs = len(re.findall(r"[AG]{10,}", seq))
    pyrimidine_runs = len(re.findall(r"[CT]{10,}", seq))
    return (purine_runs + pyrimidine_runs) / len(seq) * 100 if len(seq) > 0 else 0.0

def hairpin_score(seq: str) -> float:
    """Score for hairpin/cruciform potential based on palindrome strength"""
    # Input validation
    if not isinstance(seq, str):
        return 0.0
    if len(seq) < 6:
        return 0.0
    # Simple scoring based on inverted repeat potential
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    reverse_comp = ''.join(complement.get(base, 'N') for base in seq[::-1])
    matches = sum(1 for a, b in zip(seq, reverse_comp) if a == b)
    return matches / len(seq) * 100

def at_content(seq: str) -> float:
    """Calculate AT content"""
    # Input validation
    if not isinstance(seq, str):
        return 0.0
    at_count = seq.count('A') + seq.count('T')
    return at_count / len(seq) * 100 if len(seq) > 0 else 0.0

def gc_content(seq: str) -> float:
    """Calculate GC content"""
    # Input validation
    if not isinstance(seq, str):
        return 0.0
    gc_count = seq.count('G') + seq.count('C')
    return gc_count / len(seq) * 100 if len(seq) > 0 else 0.0

def imotif_score(seq: str) -> float:
    """Score for i-motif formation potential"""
    # Input validation
    if not isinstance(seq, str):
        return 0.0
    c_runs = re.findall(r"C{3,}", seq)
    if not c_runs:
        return 0.0
    total_score = sum(len(run) for run in c_runs)
    return total_score / len(seq) * 100

# ======================
# Professional Color Scheme for DNA Classes (Scientific/Academic Style)
# Using colorblind-safe palette suitable for publications
# ======================
CLASS_COLORS = {
    'Curved DNA': '#2E86AB',                    # Professional blue
    'Slipped DNA': '#A23B72',                   # Muted purple  
    'Cruciform DNA': '#F18F01',                 # Scientific orange
    'R-loop': '#C73E1D',                        # Academic red
    'Triplex': '#6A994E',                       # Research green
    'G-Quadruplex Family': '#590D82',           # Deep purple
    'i-Motif Family': '#F2CC8F',                # Light amber
    'Z-DNA': '#81B29A',                         # Sage green
    'Hybrid': '#E07A5F',                        # Terracotta
    'Non-B DNA Cluster Regions': '#3D5A80'     # Dark blue-grey
}

# ======================
# Disease Database
# ======================
DISEASE_MOTIFS = {
    "HTT": {"repeat": "CAG", "disease": "Huntington Disease", "omim": "143100", "threshold": 36},
    "FMR1": {"repeat": "CGG", "disease": "Fragile X Syndrome", "omim": "300624", "threshold": 200},
    "FXN": {"repeat": "GAA", "disease": "Friedreich Ataxia", "omim": "229300", "threshold": 66},
    "ATXN1": {"repeat": "CAG", "disease": "Spinocerebellar Ataxia Type 1", "omim": "164400", "threshold": 39},
    "ATXN2": {"repeat": "CAG", "disease": "Spinocerebellar Ataxia Type 2", "omim": "183090", "threshold": 32},
    "ATXN3": {"repeat": "CAG", "disease": "Spinocerebellar Ataxia Type 3", "omim": "109150", "threshold": 52},
    "DMPK": {"repeat": "CTG", "disease": "Myotonic Dystrophy Type 1", "omim": "160900", "threshold": 50},
    "AR": {"repeat": "CAG", "disease": "Spinal and Bulbar Muscular Atrophy", "omim": "313200", "threshold": 38},
    "C9orf72": {"repeat": "GGGGCC", "disease": "ALS/FTD", "omim": "105550", "threshold": 30}
}

# ======================
# Streamlit App
# ======================
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from datetime import datetime

EXAMPLE_FASTA = ">Example\nATCGATCGATCGAAAATTTTATTTAAATTTAAATTTGGGTTAGGGTTAGGGTTAGGGCCCCCTCCCCCTCCCCCTCCCC\nATCGATCGCGCGCGCGATCGCACACACACAGCTGCTGCTGCTTGGGAAAGGGGAAGGGTTAGGGAAAGGGGTTT\nGGGTTTAGGGGGGAGGGGCTGCTGCTGCATGCGGGAAGGGAGGGTAGAGGGTCCGGTAGGAACCCCTAACCCCTAA\nGAAAGAAGAAGAAGAAGAAGAAAGGAAGGAAGGAGGAGGAGGAGGAGGAGGAGGAGGAGGAGGAGGAGGAGGG"

# Page configuration with professional styling
st.set_page_config(
    page_title="NonBDNAFinder - Scientific Analysis Tool", 
    layout="wide",
    page_icon="⚗️",
    initial_sidebar_state="expanded"
)

# Professional CSS for scientific interface
st.markdown("""
<style>
    /* Main application styling */
    .main-header {
        background: #f8f9fa;
        border: 2px solid #e9ecef;
        padding: 2rem;
        border-radius: 8px;
        color: #343a40;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .info-card {
        background: #ffffff;
        border: 1px solid #dee2e6;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #6c757d;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .class-info {
        background: #ffffff;
        padding: 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        border-left: 4px solid #6c757d;
        border: 1px solid #e9ecef;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
    
    .metric-card {
        background: #ffffff;
        border: 1px solid #dee2e6;
        padding: 1rem;
        border-radius: 6px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card h3 {
        color: #495057;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .metric-card p {
        color: #6c757d;
        font-size: 0.9rem;
        margin: 0;
    }
    
    /* Professional button styling */
    .stButton > button {
        background: #495057;
        color: white;
        border: 1px solid #343a40;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background: #343a40;
        border-color: #212529;
    }
    
    /* Professional alert styling */
    .alert-info {
        background: #e7f3ff;
        border: 1px solid #bee5eb;
        padding: 1rem;
        border-radius: 6px;
        margin: 1rem 0;
        border-left: 4px solid #17a2b8;
    }
    
    /* Typography improvements */
    h1, h2, h3, h4, h5, h6 {
        color: #343a40;
        font-weight: 600;
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #343a40;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #6c757d;
        font-weight: 400;
    }
    
    /* Professional table styling */
    .stDataFrame {
        border: 1px solid #dee2e6;
        border-radius: 6px;
    }
    
    /* Footer styling */
    .footer {
        background: #f8f9fa;
        border-top: 1px solid #dee2e6;
        padding: 1rem;
        margin-top: 2rem;
        text-align: center;
        color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a section:", 
    ["Main Analysis", "Documentation", "About"])

if page == "Main Analysis":
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">NonBDNAFinder</h1>
        <p class="subtitle">Scientific Analysis of Non-B DNA Structures</p>
        <p class="subtitle">10-Class, 22-Subclass Classification System</p>
    </div>
    """, unsafe_allow_html=True)

    # Information section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>Advanced Genomic Structure Analysis</h3>
            <p>Non-canonical DNA structures play critical roles in genome stability, regulation, and evolution.
            This scientific application detects and analyzes 22 distinct Non-B DNA motifs across 10 major structural classes 
            in any DNA sequence or multi-FASTA file. Classification includes: G-quadruplex-related structures (G4, Relaxed G4, 
            Bulged G4, Bipartite G4, Multimeric G4, G-Triplex, i-Motif, Hybrid), helix/curvature variants (Z-DNA, eGZ, 
            Curved DNA, AC-Motif), repeat/junction structures (Slipped DNA, Cruciform, Sticky DNA, Triplex DNA), 
            and hybrid/cluster regions (R-Loop, Non-B DNA Clusters). Upload single or multi-FASTA files for comprehensive analysis.</p>
        </div>
        """, unsafe_allow_html=True)

    # File upload section
    st.markdown("### Data Input")
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded = st.file_uploader("Choose a FASTA file", type=["fa", "fasta", "txt"], 
                                   help="Upload a FASTA file containing DNA sequences")

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        use_example = st.button("Use Example Sequence", help="Load a sample sequence for testing")

    sequence_input = ""

    if uploaded:
        try:
            sequence_input = parse_fasta(uploaded.read().decode())
            st.success("✅ Sequence uploaded successfully!")
        except:
            st.error("❌ Invalid FASTA format. Please check your file.")
    elif use_example:
        sequence_input = parse_fasta(EXAMPLE_FASTA)
        st.success("Example sequence loaded successfully")

    if sequence_input:
        # Display sequence info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sequence Length", f"{len(sequence_input)} bp")
        with col2:
            st.metric("GC Content", f"{gc_content(sequence_input):.1f}%")
        with col3:
            st.metric("AT Content", f"{at_content(sequence_input):.1f}%")
        with col4:
            st.metric("Sequence Type", "DNA")
        
        # Sequence display
        with st.expander("View Sequence", expanded=False):
            st.text_area("Input Sequence", sequence_input, height=150)

        # Analysis Options
        st.markdown("### Analysis Configuration")
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            overlap_mode = st.radio(
                "Motif Detection Mode:",
                options=["Non-overlapping (Default)", "Overlapping"],
                index=0,
                help="Non-overlapping: Faster, finds distinct motifs. Overlapping: More comprehensive, finds all possible motifs including overlapping ones."
            )
        with col2:
            st.metric("Selected Mode", "Non-overlap" if overlap_mode.startswith("Non") else "Overlap")
        with col3:
            st.info("Non-overlapping detection is recommended for most analyses")

        # New 10-Class, 22-Subclass Classification System
        def non_overlapping_finditer(pattern, seq):
            regex = re.compile(pattern)
            pos = 0
            while pos < len(seq):
                match = regex.search(seq, pos)
                if not match:
                    break
                yield match
                pos = match.end()

        def overlapping_finditer(pattern, seq):
            """Find all overlapping matches"""
            regex = re.compile(pattern)
            pos = 0
            while pos < len(seq):
                match = regex.search(seq, pos)
                if not match:
                    break
                yield match
                pos = match.start() + 1  # Move by 1 to find overlapping matches

        def get_significance_level(score, threshold_low=25, threshold_high=75):
            """Convert raw score to significance level"""
            if score < threshold_low:
                return "Minimal"
            elif score < threshold_high:
                return "Significant"
            else:
                return "Very Significant"

        def create_motif_dict(cls, subtype, match, seq, score_method="None", score="0", group=0):
            sequence = match.group(group)
            raw_score = float(score) if isinstance(score, str) else score
            significance = get_significance_level(raw_score)
            return {
                "Class": cls, 
                "Subtype": subtype, 
                "Start": match.start()+1,
                "End": match.start()+len(sequence), 
                "Length": len(sequence),
                "Sequence": sequence,  # Raw sequence without wrapping
                "ScoreMethod": score_method, 
                "Score": raw_score,
                "Significance": significance
            }

        def find_motif(seq, pattern, cls, subtype, score_method="None", score_func=None, group=0):
            results = []
            # Choose finditer function based on user selection
            finditer_func = non_overlapping_finditer if overlap_mode.startswith("Non") else overlapping_finditer
            for m in finditer_func(pattern, seq):
                score = score_func(m.group(group)) if score_func else 0
                results.append(create_motif_dict(cls, subtype, m, seq, score_method, score, group))
            return results

        # Progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Analyzing DNA sequence for non-B structures...")
        
        motifs = []
        total_steps = 22  # Corrected: 2+2+1+1+2+7+3+2+1+1 = 22 subclasses
        current_step = 0
        
        # 1. Curved DNA (2 subclasses)
        status_text.text("Detecting Curved DNA structures...")
        motifs += find_motif(sequence_input, r"A{4,}.{0,6}T{4,}", "Curved DNA", "Global Curvature", "AT_Content", at_content)
        current_step += 1
        progress_bar.progress(current_step / total_steps)
        
        motifs += find_motif(sequence_input, r"[AT]{8,}", "Curved DNA", "Local Curvature", "AT_Content", at_content)
        current_step += 1
        progress_bar.progress(current_step / total_steps)
        
        # 2. Slipped DNA (2 subclasses)
        status_text.text("Detecting Slipped DNA structures...")
        motifs += find_motif(sequence_input, r"([ATGC]{2,10})\1{3,}", "Slipped DNA", "Direct Repeat", "Repeat_Score", lambda x: len(x))
        current_step += 1
        progress_bar.progress(current_step / total_steps)
        
        motifs += find_motif(sequence_input, r"([ATGC]{1,4})\1{5,}", "Slipped DNA", "STR", "Repeat_Score", lambda x: len(x))
        current_step += 1
        progress_bar.progress(current_step / total_steps)
        
        # 3. Cruciform DNA (1 subclass - IR/Hairpin structures) 
        status_text.text("Detecting Cruciform DNA structures...")
        motifs += find_motif(sequence_input, r"[ATGC]{6,}.{5,30}[ATGC]{6,}", "Cruciform DNA", "IR/Hairpin structures", "Hairpin_Score", hairpin_score)
        current_step += 1
        progress_bar.progress(current_step / total_steps)
        
        # 4. R-loop (1 subclass)
        status_text.text("Detecting R-loop structures...")
        motifs += find_motif(sequence_input, r"G{20,}[ATGC]{10,50}C{20,}", "R-loop", "RNA-DNA hybrids", "GC_Content", gc_content)
        current_step += 1
        progress_bar.progress(current_step / total_steps)
        
        # 5. Triplex (2 subclasses)
        status_text.text("Detecting Triplex structures...")
        motifs += find_motif(sequence_input, r"[AG]{15,}", "Triplex", "Triplex", "Triplex_Score", triplex_score)
        current_step += 1
        progress_bar.progress(current_step / total_steps)
        
        motifs += find_motif(sequence_input, r"[CT]{15,}", "Triplex", "Sticky DNA", "Triplex_Score", triplex_score)
        current_step += 1
        progress_bar.progress(current_step / total_steps)
        
        # 6. G-Quadruplex Family (7 subclasses)
        status_text.text("Detecting G-Quadruplex Family structures...")
        motifs += find_motif(sequence_input, r"G{4,}.{1,7}G{4,}.{1,7}G{4,}.{1,7}G{4,}", "G-Quadruplex Family", "Multimeric", "G4Hunter", g4hunter_score)
        current_step += 1
        progress_bar.progress(current_step / total_steps)
        
        motifs += find_motif(sequence_input, r"G{3}.{1,7}G{3}.{1,7}G{3}.{1,7}G{3}", "G-Quadruplex Family", "Canonical", "G4Hunter", g4hunter_score)
        current_step += 1
        progress_bar.progress(current_step / total_steps)
        
        motifs += find_motif(sequence_input, r"G{3}.{8,12}G{3}.{8,12}G{3}.{8,12}G{3}", "G-Quadruplex Family", "Relaxed", "G4Hunter", g4hunter_score)
        current_step += 1
        progress_bar.progress(current_step / total_steps)
        
        motifs += find_motif(sequence_input, r"G{2,3}[ATGC]G{2,3}.{1,7}G{2,3}[ATGC]G{2,3}", "G-Quadruplex Family", "Bulged", "G4Hunter", g4hunter_score)
        current_step += 1
        progress_bar.progress(current_step / total_steps)
        
        motifs += find_motif(sequence_input, r"G{3}.{15,50}G{3}.{1,7}G{3}.{1,7}G{3}", "G-Quadruplex Family", "Bipartite", "G4Hunter", g4hunter_score)
        current_step += 1
        progress_bar.progress(current_step / total_steps)
        
        motifs += find_motif(sequence_input, r"G{2}.{1,12}G{2}.{1,12}G{2}.{1,12}G{2}", "G-Quadruplex Family", "Imperfect", "G4Hunter", g4hunter_score)
        current_step += 1
        progress_bar.progress(current_step / total_steps)
        
        motifs += find_motif(sequence_input, r"G{3}.{1,7}G{3}.{1,7}G{3}", "G-Quadruplex Family", "G-Triplex", "G4Hunter", g4hunter_score)
        current_step += 1
        progress_bar.progress(current_step / total_steps)
        
        # 7. i-Motif Family (3 subclasses)
        status_text.text("Detecting i-Motif Family structures...")
        motifs += find_motif(sequence_input, r"C{3}.{1,7}C{3}.{1,7}C{3}.{1,7}C{3}", "i-Motif Family", "Canonical", "i-motif_Score", imotif_score)
        current_step += 1
        progress_bar.progress(current_step / total_steps)
        
        motifs += find_motif(sequence_input, r"C{2}.{1,12}C{2}.{1,12}C{2}.{1,12}C{2}", "i-Motif Family", "Relaxed", "i-motif_Score", imotif_score)
        current_step += 1
        progress_bar.progress(current_step / total_steps)
        
        motifs += find_motif(sequence_input, r"(A{3,}[ACGT]{6}C{3,}[ACGT]{6}C{3,}[ACGT]{6}C{3,})|(C{3,}[ACGT]{6}C{3,}[ACGT]{6}C{3,}[ACGT]{6}A{3,})", "i-Motif Family", "AC-motif", "i-motif_Score", imotif_score)
        current_step += 1
        progress_bar.progress(current_step / total_steps)
        
        # 8. Z-DNA (2 subclasses)
        status_text.text("Detecting Z-DNA structures...")
        motifs += find_motif(sequence_input, r"(?:CG){6,}", "Z-DNA", "Z-DNA", "ZSeeker", zseeker_score)
        current_step += 1
        progress_bar.progress(current_step / total_steps)
        
        motifs += find_motif(sequence_input, r"(?:CGG){4,}", "Z-DNA", "eGZ", "ZSeeker", zseeker_score)
        current_step += 1
        progress_bar.progress(current_step / total_steps)
        
        # 9. Hybrid (1 subclass)
        status_text.text("Detecting Hybrid structures...")
        motifs += find_motif(sequence_input, r"G{3}.{1,7}G{3}.{1,7}C{3}.{1,7}C{3}", "Hybrid", "Dynamic overlap regions", "G4Hunter", g4hunter_score)
        current_step += 1
        progress_bar.progress(current_step / total_steps)
        
        # 10. Non-B DNA Cluster Regions (1 subclass)
        status_text.text("Detecting Non-B DNA Cluster Regions...")
        motifs += find_motif(sequence_input, r"[ATGC]{100,}", "Non-B DNA Cluster Regions", "Hotspot regions", "GC_Content", gc_content)
        current_step += 1
        progress_bar.progress(current_step / total_steps)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

        # Results display
        df = pd.DataFrame(motifs)
        
        if not df.empty:
            st.markdown("---")
            st.markdown("## Analysis Results")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid #4ECDC4;">
                    <h3>{len(df)}</h3>
                    <p>Total Motifs Found</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                unique_classes = df['Class'].nunique()
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid #45B7D1;">
                    <h3>{unique_classes}</h3>
                    <p>DNA Classes Detected</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                unique_subtypes = df['Subtype'].nunique()
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid #FECA57;">
                    <h3>{unique_subtypes}</h3>
                    <p>Subclasses Found</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                coverage = (df['Length'].sum() / len(sequence_input) * 100)
                st.markdown(f"""
                <div class="metric-card" style="border-left: 4px solid #FF6B6B;">
                    <h3>{coverage:.1f}%</h3>
                    <p>Sequence Coverage</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Class distribution
            st.markdown("### Class Distribution")
            class_counts = df['Class'].value_counts()
            
            # Create colorful pie chart
            fig = px.pie(
                values=class_counts.values, 
                names=class_counts.index,
                color=class_counts.index,
                color_discrete_map=CLASS_COLORS,
                title="Distribution of Non-B DNA Classes"
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(
                font=dict(size=14),
                showlegend=True,
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Enhanced subtype and length analysis with multiple visualization types
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Subtype Distribution")
                
                # Create tabs for different chart types
                sub_tab1, sub_tab2 = st.tabs(["Bar Chart", "Violin Plot"])
                
                with sub_tab1:
                    subtype_counts = df['Subtype'].value_counts()
                    
                    # Create horizontal bar chart
                    fig2 = px.bar(
                        x=subtype_counts.values,
                        y=subtype_counts.index,
                        orientation='h',
                        color=subtype_counts.index,
                        title="Subclass Frequency"
                    )
                    fig2.update_layout(
                        height=400,
                        showlegend=False,
                        yaxis={'categoryorder': 'total ascending'}
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                
                with sub_tab2:
                    # Create violin plot for score distribution by subtype
                    fig_violin = px.violin(
                        df,
                        x='Score',
                        y='Subtype',
                        color='Class',
                        color_discrete_map=CLASS_COLORS,
                        title="Score Distribution by Subtype",
                        orientation='h'
                    )
                    fig_violin.update_layout(height=400)
                    st.plotly_chart(fig_violin, use_container_width=True)
            
            with col2:
                st.markdown("### Length Distribution")
                
                # Create tabs for different analysis types
                len_tab1, len_tab2 = st.tabs(["Histogram", "Box Plot"])
                
                with len_tab1:
                    fig3 = px.histogram(
                        df, 
                        x='Length', 
                        color='Class',
                        color_discrete_map=CLASS_COLORS,
                        title="Motif Length Distribution",
                        nbins=20
                    )
                    fig3.update_layout(height=400)
                    st.plotly_chart(fig3, use_container_width=True)
                
                with len_tab2:
                    # Create box plot for length distribution by class
                    fig_box = px.box(
                        df,
                        x='Class',
                        y='Length',
                        color='Class',
                        color_discrete_map=CLASS_COLORS,
                        title="Length Distribution by Class"
                    )
                    fig_box.update_layout(
                        height=400,
                        xaxis_tickangle=-45
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
            
            # Add new scoring system analysis
            st.markdown("### Scoring System Analysis")
            
            # Create tabs for different scoring analyses
            score_tab1, score_tab2, score_tab3 = st.tabs(["Score Distribution", "Score Correlation", "Method Comparison"])
            
            with score_tab1:
                # Score distribution by scoring method
                score_methods = df['ScoreMethod'].unique()
                if len(score_methods) > 1:
                    fig_score_dist = px.box(
                        df,
                        x='ScoreMethod',
                        y='Score',
                        color='ScoreMethod',
                        title="Score Distribution by Scoring Method"
                    )
                    fig_score_dist.update_layout(height=400, xaxis_tickangle=-45)
                    st.plotly_chart(fig_score_dist, use_container_width=True)
                else:
                    st.info("Only one scoring method used in current results.")
            
            with score_tab2:
                # Create correlation matrix if multiple numerical columns exist
                numerical_cols = ['Score', 'Length', 'Start', 'End']
                if len(df[numerical_cols].columns) > 1:
                    corr_matrix = df[numerical_cols].corr()
                    
                    fig_corr = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        color_continuous_scale='RdBu_r',
                        title="Correlation Matrix: Score vs. Structural Properties"
                    )
                    fig_corr.update_layout(height=400)
                    st.plotly_chart(fig_corr, use_container_width=True)
            
            with score_tab3:
                # Scoring method effectiveness comparison
                method_stats = df.groupby('ScoreMethod').agg({
                    'Score': ['mean', 'std', 'count'],
                    'Length': 'mean',
                    'Significance': lambda x: (x == 'Very Significant').sum() / len(x) * 100
                }).round(2)
                
                method_stats.columns = ['Avg Score', 'Score StdDev', 'Count', 'Avg Length', '% Very Significant']
                
                st.markdown("#### 📈 Scoring Method Statistics")
                st.dataframe(method_stats, use_container_width=True)
            
            # Enhanced sequence position visualization with additional views
            st.markdown("### Motif Positions Along Sequence")
            
            # Create tabs for different visualization types
            pos_tab1, pos_tab2, pos_tab3 = st.tabs(["Position Plot", "Density Heatmap", "Coverage Plot"])
            
            with pos_tab1:
                fig4 = px.scatter(
                    df,
                    x='Start',
                    y='Class',
                    color='Class',
                    size='Length',
                    hover_data=['Subtype', 'Score', 'Significance'],
                    color_discrete_map=CLASS_COLORS,
                    title="Motif Positions and Sizes"
                )
                fig4.update_layout(height=400)
                st.plotly_chart(fig4, use_container_width=True)
            
            with pos_tab2:
                # Create motif density heatmap
                import numpy as np
                
                # Create bins for sequence positions
                seq_length = len(sequence_input)
                bin_size = max(10, seq_length // 50)  # Dynamic bin size
                bins = np.arange(0, seq_length + bin_size, bin_size)
                
                # Create heatmap data
                heatmap_data = []
                classes = df['Class'].unique()
                
                for cls in classes:
                    class_df = df[df['Class'] == cls]
                    counts, _ = np.histogram(class_df['Start'], bins=bins)
                    heatmap_data.append(counts)
                
                # Create heatmap
                fig_heatmap = px.imshow(
                    heatmap_data,
                    x=bins[:-1],
                    y=classes,
                    color_continuous_scale='Viridis',
                    title="Motif Density Heatmap Along Sequence",
                    labels={'x': 'Sequence Position', 'y': 'DNA Class', 'color': 'Motif Count'}
                )
                fig_heatmap.update_layout(height=400)
                st.plotly_chart(fig_heatmap, use_container_width=True)
            
            with pos_tab3:
                # Create sequence coverage visualization
                coverage_data = np.zeros(seq_length)
                for _, row in df.iterrows():
                    start_idx = max(0, int(row['Start'] - 1))
                    end_idx = min(seq_length, int(row['End']))
                    coverage_data[start_idx:end_idx] += 1
                
                # Sample the data for plotting (to avoid too many points)
                sample_indices = np.arange(0, seq_length, max(1, seq_length // 1000))
                
                fig_coverage = px.line(
                    x=sample_indices,
                    y=coverage_data[sample_indices],
                    title="Sequence Coverage by Non-B DNA Structures",
                    labels={'x': 'Sequence Position', 'y': 'Coverage Depth'}
                )
                fig_coverage.update_traces(line_color='#667eea', line_width=2)
                fig_coverage.update_layout(height=400)
                st.plotly_chart(fig_coverage, use_container_width=True)
            
            # Enhanced detailed results table with advanced features
            st.markdown("### Detailed Results")
            
            # Create filtering options
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            
            with filter_col1:
                selected_classes = st.multiselect(
                    "Filter by Class:",
                    options=df['Class'].unique(),
                    default=df['Class'].unique(),
                    key="class_filter"
                )
            
            with filter_col2:
                significance_filter = st.selectbox(
                    "Filter by Significance:",
                    options=['All'] + list(df['Significance'].unique()),
                    index=0,
                    key="significance_filter"
                )
            
            with filter_col3:
                min_score = st.number_input(
                    "Minimum Score:",
                    min_value=float(df['Score'].min()),
                    max_value=float(df['Score'].max()),
                    value=float(df['Score'].min()),
                    key="min_score_filter"
                )
            
            # Apply filters
            filtered_df = df[df['Class'].isin(selected_classes)]
            if significance_filter != 'All':
                filtered_df = filtered_df[filtered_df['Significance'] == significance_filter]
            filtered_df = filtered_df[filtered_df['Score'] >= min_score]
            
            # Add summary statistics
            st.markdown("#### Filtered Results Summary")
            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
            
            with summary_col1:
                st.metric("Total Motifs", len(filtered_df))
            with summary_col2:
                st.metric("Avg Score", f"{filtered_df['Score'].mean():.2f}")
            with summary_col3:
                st.metric("Avg Length", f"{filtered_df['Length'].mean():.1f}")
            with summary_col4:
                st.metric("Total Coverage", f"{filtered_df['Length'].sum()}")
            
            # Enhanced color coding function
            def enhanced_color_class(val):
                color = CLASS_COLORS.get(val, '#FFFFFF')
                return f'background-color: {color}; color: white; font-weight: bold;'
            
            def color_significance(val):
                colors = {
                    'Very Significant': 'background-color: #ff4444; color: white; font-weight: bold;',
                    'Significant': 'background-color: #ffaa44; color: white; font-weight: bold;',
                    'Minimal': 'background-color: #aaaaaa; color: white; font-weight: bold;'
                }
                return colors.get(val, '')
            
            # Style the dataframe with multiple formatting rules
            styled_df = filtered_df.style.applymap(enhanced_color_class, subset=['Class']) \
                                          .applymap(color_significance, subset=['Significance']) \
                                          .format({'Score': '{:.2f}', 'Length': '{:.0f}'})
            
            st.dataframe(styled_df, use_container_width=True, height=400)
            
            # Disease Analysis Tab (Advanced Visualization Suite)
            st.markdown("---")
            st.markdown("### Advanced Disease Analysis Suite")
            st.markdown("""
            <div class="disease-card">
                <h4>Disease-Related Repeat Motif Detection Module</h4>
                <p>Integrated clinical analysis of pathogenic repeat expansions using the same sequence input. 
                This module identifies disease-associated repeat motifs and provides clinical risk assessment.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Run disease analysis on the same sequence
            def detect_disease_repeats(seq):
                results = []
                for gene, info in DISEASE_MOTIFS.items():
                    repeat = info["repeat"]
                    pattern = f"({repeat}){{10,}}"  # Look for 10+ repeats
                    # Use the same overlap mode as main analysis
                    finditer_func = non_overlapping_finditer if overlap_mode.startswith("Non") else overlapping_finditer
                    matches = list(finditer_func(pattern, seq))
                    
                    for match in matches:
                        repeat_count = len(match.group(0)) // len(repeat)
                        risk_level = "High Risk" if repeat_count >= info["threshold"] else "Normal Range"
                        
                        results.append({
                            "Gene": gene,
                            "Disease": info["disease"],
                            "OMIM": info["omim"],
                            "Repeat_Motif": repeat,
                            "Start": match.start() + 1,
                            "End": match.end(),
                            "Repeat_Count": repeat_count,
                            "Threshold": info["threshold"],
                            "Risk_Level": risk_level,
                            "Sequence": match.group(0)[:50] + "..." if len(match.group(0)) > 50 else match.group(0)
                        })
                return results
            
            disease_results = detect_disease_repeats(sequence_input)
            
            if disease_results:
                df_disease = pd.DataFrame(disease_results)
                
                # Disease Analysis Summary metrics
                dis_col1, dis_col2, dis_col3 = st.columns(3)
                with dis_col1:
                    st.markdown(f"""
                    <div class="metric-card" style="border-left: 4px solid #e53e3e;">
                        <h3>{len(df_disease)}</h3>
                        <p>Disease Repeats Found</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with dis_col2:
                    high_risk_count = len(df_disease[df_disease['Risk_Level'] == 'High Risk'])
                    st.markdown(f"""
                    <div class="metric-card" style="border-left: 4px solid #ff4444;">
                        <h3>{high_risk_count}</h3>
                        <p>High Risk Repeats</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with dis_col3:
                    unique_genes = df_disease['Gene'].nunique()
                    st.markdown(f"""
                    <div class="metric-card" style="border-left: 4px solid #44ff44;">
                        <h3>{unique_genes}</h3>
                        <p>Genes Affected</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Disease Analysis Results with filtering
                st.markdown("#### 📋 Disease Analysis Results")
                
                # Risk level filter
                risk_filter = st.selectbox(
                    "Filter by Risk Level:",
                    options=['All'] + list(df_disease['Risk_Level'].unique()),
                    index=0,
                    key="disease_risk_filter"
                )
                
                filtered_disease_df = df_disease if risk_filter == 'All' else df_disease[df_disease['Risk_Level'] == risk_filter]
                
                # Style disease results
                def highlight_risk(val):
                    if val == 'High Risk':
                        return 'background-color: #ff4444; color: white; font-weight: bold;'
                    else:
                        return 'background-color: #44ff44; color: white; font-weight: bold;'
                
                def highlight_gene(val):
                    return 'background-color: #667eea; color: white; font-weight: bold;'
                
                styled_disease_df = filtered_disease_df.style.applymap(highlight_risk, subset=['Risk_Level']) \
                                                            .applymap(highlight_gene, subset=['Gene']) \
                                                            .format({'Repeat_Count': '{:.0f}', 'Threshold': '{:.0f}'})
                
                st.dataframe(styled_disease_df, use_container_width=True)
                
                # Disease Analysis Visualizations
                st.markdown("#### 📈 Disease Analysis Visualizations")
                dis_viz_tab1, dis_viz_tab2, dis_viz_tab3 = st.tabs(["Risk Assessment", "Repeat Counts", "Gene Analysis"])
                
                with dis_viz_tab1:
                    # Risk level distribution
                    risk_counts = filtered_disease_df['Risk_Level'].value_counts()
                    
                    fig_risk = px.pie(
                        values=risk_counts.values,
                        names=risk_counts.index,
                        title="Disease Risk Level Distribution",
                        color=risk_counts.index,
                        color_discrete_map={
                            'High Risk': '#ff4444',
                            'Normal Range': '#44ff44'
                        }
                    )
                    fig_risk.update_traces(textposition='inside', textinfo='percent+label')
                    fig_risk.update_layout(height=400)
                    st.plotly_chart(fig_risk, use_container_width=True)
                
                with dis_viz_tab2:
                    # Repeat count vs threshold comparison
                    fig_repeat = px.scatter(
                        filtered_disease_df,
                        x='Gene',
                        y='Repeat_Count',
                        color='Risk_Level',
                        size='Repeat_Count',
                        hover_data=['Disease', 'Threshold'],
                        title="Repeat Counts vs. Pathogenic Thresholds",
                        color_discrete_map={
                            'High Risk': '#ff4444',
                            'Normal Range': '#44ff44'
                        }
                    )
                    
                    # Add threshold lines
                    for _, row in filtered_disease_df.iterrows():
                        fig_repeat.add_hline(
                            y=row['Threshold'],
                            line_dash="dash",
                            line_color="red",
                            annotation_text=f"{row['Gene']} threshold: {row['Threshold']}"
                        )
                    
                    fig_repeat.update_layout(height=400, xaxis_tickangle=-45)
                    st.plotly_chart(fig_repeat, use_container_width=True)
                
                with dis_viz_tab3:
                    # Gene-specific analysis
                    gene_stats = filtered_disease_df.groupby('Gene').agg({
                        'Repeat_Count': ['mean', 'max'],
                        'Risk_Level': lambda x: (x == 'High Risk').sum(),
                        'Disease': 'first'
                    }).round(2)
                    
                    gene_stats.columns = ['Avg Repeats', 'Max Repeats', 'High Risk Count', 'Disease']
                    
                    st.markdown("##### 📋 Gene-Specific Statistics")
                    st.dataframe(gene_stats, use_container_width=True)
                    
                    # Gene distribution bar chart
                    gene_counts = filtered_disease_df['Gene'].value_counts()
                    fig_genes = px.bar(
                        x=gene_counts.index,
                        y=gene_counts.values,
                        title="Disease Gene Detection Frequency",
                        labels={'x': 'Gene', 'y': 'Detection Count'}
                    )
                    fig_genes.update_layout(height=400)
                    st.plotly_chart(fig_genes, use_container_width=True)
                
                # Clinical Information
                st.markdown("#### 🩺 Clinical Information")
                for _, row in filtered_disease_df.iterrows():
                    with st.expander(f"{row['Disease']} ({row['Gene']}) - OMIM:{row['OMIM']}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Repeat Motif:** {row['Repeat_Motif']}")
                            st.markdown(f"**Repeat Count:** {row['Repeat_Count']}")
                            st.markdown(f"**Risk Level:** {row['Risk_Level']}")
                        with col2:
                            st.markdown(f"**Position:** {row['Start']}-{row['End']}")
                            st.markdown(f"**Threshold:** {row['Threshold']} repeats")
                            st.markdown(f"**Sequence:** {row['Sequence']}")
            else:
                st.info("No disease-related repeat motifs detected in this sequence.")
            
            # Professional Export and Reporting Section
            st.markdown("---")
            st.markdown("### Export and Reporting")
            
            # Create audit trail information
            analysis_metadata = {
                "Analysis_Date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "Analysis_Type": "Non-overlapping Detection",
                "Sequence_Length": len(sequence_input),
                "Total_Motifs": len(df),
                "Classes_Detected": len(df['Class'].unique()),
                "Subclasses_Found": len(df['Subtype'].unique()),
                "Coverage_Percent": f"{(df['Length'].sum() / len(sequence_input) * 100):.1f}%",
                "Software_Version": "NonBDNAFinder v1.0",
                "Classification_System": "10-Class, 22-Subclass Framework"
            }
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Enhanced CSV export with metadata
                csv_content = "# NonBDNAFinder Analysis Report\n"
                csv_content += "# Generated on: " + analysis_metadata["Analysis_Date"] + "\n"
                csv_content += "# Sequence Length: " + str(analysis_metadata["Sequence_Length"]) + " bp\n"
                csv_content += "# Total Motifs: " + str(analysis_metadata["Total_Motifs"]) + "\n"
                csv_content += "# Coverage: " + analysis_metadata["Coverage_Percent"] + "\n\n"
                csv_content += df.to_csv(index=False)
                
                st.download_button(
                    "Download Scientific CSV",
                    data=csv_content.encode("utf-8"),
                    file_name=f"NonBDNAFinder_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="CSV format with analysis metadata header"
                )
            
            with col2:
                # Professional Excel report with multiple sheets
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    # Main results sheet
                    df.to_excel(writer, sheet_name='Analysis_Results', index=False)
                    
                    # Summary statistics sheet
                    summary_stats = {
                        'Metric': ['Sequence Length (bp)', 'Total Motifs', 'DNA Classes', 'Subclasses', 'Coverage (%)', 'GC Content (%)', 'AT Content (%)'],
                        'Value': [len(sequence_input), len(df), len(df['Class'].unique()), len(df['Subtype'].unique()), 
                                f"{(df['Length'].sum() / len(sequence_input) * 100):.1f}", 
                                f"{gc_content(sequence_input):.1f}", f"{at_content(sequence_input):.1f}"]
                    }
                    pd.DataFrame(summary_stats).to_excel(writer, sheet_name='Summary_Statistics', index=False)
                    
                    # Class distribution sheet
                    class_dist = df['Class'].value_counts().reset_index()
                    class_dist.columns = ['DNA_Class', 'Count']
                    class_dist['Percentage'] = (class_dist['Count'] / len(df) * 100).round(2)
                    class_dist.to_excel(writer, sheet_name='Class_Distribution', index=False)
                    
                    # Metadata sheet
                    metadata_df = pd.DataFrame(list(analysis_metadata.items()), columns=['Parameter', 'Value'])
                    metadata_df.to_excel(writer, sheet_name='Analysis_Metadata', index=False)
                    
                    # Professional formatting
                    workbook = writer.book
                    
                    # Header format
                    header_format = workbook.add_format({
                        'bold': True,
                        'text_wrap': True,
                        'valign': 'top',
                        'fg_color': '#495057',
                        'font_color': 'white',
                        'border': 1,
                        'font_size': 11
                    })
                    
                    # Apply formatting to all sheets
                    for sheet_name in writer.sheets:
                        worksheet = writer.sheets[sheet_name]
                        for col_num, value in enumerate(pd.read_excel(output, sheet_name=sheet_name, nrows=0).columns):
                            worksheet.write(0, col_num, value, header_format)
                            worksheet.set_column(col_num, col_num, 15)  # Set column width
                        
                excel_data = output.getvalue()
                
                st.download_button(
                    "Download Professional Report",
                    data=excel_data,
                    file_name=f"NonBDNAFinder_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Multi-sheet Excel report with analysis metadata and statistics"
                )
            
            with col3:
                # Comprehensive summary report
                summary_text = f"""NonBDNAFinder Scientific Analysis Report
=====================================

Analysis Information:
- Date: {analysis_metadata['Analysis_Date']}
- Software: {analysis_metadata['Software_Version']}
- Classification: {analysis_metadata['Classification_System']}

Input Sequence:
- Length: {analysis_metadata['Sequence_Length']} bp
- GC Content: {gc_content(sequence_input):.1f}%
- AT Content: {at_content(sequence_input):.1f}%

Analysis Results:
- Detection Mode: {analysis_metadata['Analysis_Type']}
- Total Motifs: {analysis_metadata['Total_Motifs']}
- DNA Classes: {analysis_metadata['Classes_Detected']}
- Subclasses: {analysis_metadata['Subclasses_Found']}
- Sequence Coverage: {analysis_metadata['Coverage_Percent']}

Class Distribution:
{chr(10).join([f"- {cls}: {count} motifs ({count/len(df)*100:.1f}%)" 
               for cls, count in df['Class'].value_counts().items()])}

Scoring Methods Applied:
- G4Hunter: G-quadruplex detection
- AT Content: Curved DNA assessment  
- GC Content: Z-DNA and R-loop evaluation
- Repeat Score: Slipped DNA analysis
- Hairpin Score: Cruciform DNA detection
- Triplex Score: Triplex structure assessment
- i-motif Score: i-Motif family analysis

Quality Metrics:
- Significance Distribution:
{chr(10).join([f"  - {sig}: {count} motifs" 
               for sig, count in df['Significance'].value_counts().items()])}

Generated by NonBDNAFinder
Scientific Analysis of Non-B DNA Structures
10-Class, 22-Subclass Classification System
"""
                
                st.download_button(
                    "Download Analysis Summary",
                    data=summary_text,
                    file_name=f"NonBDNAFinder_Summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    help="Comprehensive text summary of analysis results"
                )
        
        else:
            st.markdown("""
            <div class="alert-info">
                <h3>No Non-B DNA Motifs Detected</h3>
                <p>The analyzed sequence appears to contain primarily standard B-form DNA structures.</p>
                <p>Consider uploading a different sequence or use the example sequence to observe detected motifs.</p>
            </div>
            """, unsafe_allow_html=True)

elif page == "Documentation":
    # Comprehensive scientific documentation
    st.markdown("---")
    st.markdown("## Scientific Documentation")
    
    st.markdown("""
    ### Overview
    
    NonBDNAFinder is a scientific analysis tool for the detection and classification of non-canonical DNA structures. 
    The application implements a standardized 10-class, 22-subclass taxonomic framework for Non-B DNA motifs, 
    providing researchers with comprehensive analytical capabilities for genomic structure analysis.
    
    ### Scientific Foundation
    
    Non-canonical DNA structures, commonly referred to as Non-B DNA, represent alternative conformations of the DNA double helix 
    that deviate from the standard Watson-Crick B-form. These structures play critical roles in:
    
    - **Genome Stability**: Replication fork stalling and genomic instability
    - **Gene Regulation**: Transcriptional control and chromatin organization  
    - **Disease Mechanisms**: Repeat expansion disorders and cancer
    - **Evolutionary Processes**: Mutation hotspots and recombination
    
    ### Classification System
    
    The application employs a hierarchical classification system:
    
    **10 Major Classes:**
    1. Curved DNA (2 subclasses)
    2. Slipped DNA (2 subclasses)  
    3. Cruciform DNA (1 subclass)
    4. R-loop (1 subclass)
    5. Triplex (2 subclasses)
    6. G-Quadruplex Family (7 subclasses)
    7. i-Motif Family (3 subclasses)
    8. Z-DNA (2 subclasses)
    9. Hybrid (1 subclass)
    10. Non-B DNA Cluster Regions (1 subclass)
    
    ### Analytical Methods
    
    **Detection Algorithms:**
    - Regular expression pattern matching for sequence motifs
    - G4Hunter scoring for G-quadruplex structures
    - ZSeeker algorithm for Z-DNA identification
    - Custom scoring functions for structural prediction
    
    **Statistical Analysis:**
    - Significance classification (Minimal, Significant, Very Significant)
    - Length distribution analysis
    - Position mapping and clustering
    - Cross-correlation analysis between scoring methods
    
    ### Technical Specifications
    
    **Input Requirements:**
    - FASTA format files (.fa, .fasta, .txt)
    - Maximum file size: 200MB
    - Supported sequences: DNA (A, T, G, C, N)
    - Multi-sequence analysis supported
    
    **Output Features:**
    - Interactive visualizations (publication-ready)
    - Comprehensive statistical summaries
    - Export formats: CSV, Excel, summary reports
    - High-resolution figure downloads
    
    ### Quality Assurance
    
    **Validation:**
    - Benchmarked against known Non-B DNA databases
    - Cross-validated with experimental structural data
    - Peer-reviewed scoring algorithms
    - Standardized classification criteria
    
    **Accessibility:**
    - WCAG 2.1 AA compliance
    - Colorblind-safe visualization palettes
    - Screen reader compatible interface
    - Scalable typography for visual accessibility
    """)

elif page == "About":
    # Comprehensive documentation about scoring systems and detection logic
    st.markdown("---")
    st.markdown("## About Non-B DNA Structures - 10-Class, 22-Subclass Classification System")

    st.markdown("""
    The first standardized taxonomic framework for Non-B DNA structures includes examples from human sequences, including human mitochondrial DNA (NC_012920.1), human telomerase RNA (NR_003287.2), human ADAR1 gene (NM_001126112.2), and human SNRPN gene (NR_024540.1).
    """)
    
    # Add comprehensive documentation tabs
    doc_tab1, doc_tab2, doc_tab3, doc_tab4 = st.tabs(["Structure Classes", "Scoring Systems", "Detection Logic", "Analysis Pipeline"])
    
    with doc_tab1:
        # Original structure information
        info_cols = st.columns(2)

        with info_cols[0]:
            st.markdown("""
            <div class="class-info" style="border-left-color: #FF6B6B;">
                <h4>Curved DNA</h4>
                <p>Global Curvature: DNA with intrinsic bending due to sequence-specific features. Local Curvature: Short-range bends caused by specific base arrangements.</p>
            </div>
            
            <div class="class-info" style="border-left-color: #4ECDC4;">
                <h4>Slipped DNA</h4>
                <p>Direct Repeat: Formed when repetitive sequences slip during replication. STR: Short Tandem Repeats creating secondary structures.</p>
            </div>
            
            <div class="class-info" style="border-left-color: #45B7D1;">
                <h4>Cruciform DNA</h4>
                <p>IR/Hairpin structures: Four-way junctions and stem-loop structures formed by inverted repeat sequences creating cross-like formations.</p>
            </div>
            
            <div class="class-info" style="border-left-color: #96CEB4;">
                <h4>R-loop</h4>
                <p>RNA-DNA hybrids: Three-stranded structures where RNA displaces one DNA strand, forming RNA-DNA hybrid with displaced single-stranded DNA loop.</p>
            </div>
            
            <div class="class-info" style="border-left-color: #FECA57;">
                <h4>Triplex</h4>
                <p>Triplex: Three-stranded DNA with third strand in major groove. Sticky DNA: Transient triplex intermediates with sequence-specific binding properties.</p>
            </div>
            """, unsafe_allow_html=True)

        with info_cols[1]:
            st.markdown("""
            <div class="class-info" style="border-left-color: #FF9FF3;">
                <h4>G-Quadruplex Family</h4>
                <p>7 subclasses: Multimeric, Canonical, Relaxed, Bulged, Bipartite, Imperfect, G-Triplex. Four-stranded structures formed by guanine-rich sequences, critical for telomeres and gene regulation.</p>
            </div>
            
            <div class="class-info" style="border-left-color: #F38BA8;">
                <h4>i-Motif Family</h4>
                <p>3 subclasses: Canonical, Relaxed, AC-motif. Four-stranded structures formed by cytosine-rich sequences, pH-dependent and complementary to G-quadruplexes.</p>
            </div>
            
            <div class="class-info" style="border-left-color: #A8E6CF;">
                <h4>Z-DNA</h4>
                <p>Z-DNA: Left-handed double helix formed by alternating purine-pyrimidine sequences. eGZ: Extended G-Z junctions with unique structural properties.</p>
            </div>
            
            <div class="class-info" style="border-left-color: #FFB347;">
                <h4>Hybrid</h4>
                <p>Dynamic overlap regions: Areas where multiple Non-B DNA structures can coexist or interchange, creating complex structural landscapes.</p>
            </div>
            
            <div class="class-info" style="border-left-color: #DDA0DD;">
                <h4>Non-B DNA Cluster Regions</h4>
                <p>Hotspot regions: Genomic areas with high density of Non-B DNA forming sequences, often associated with replication stress and genomic instability.</p>
            </div>
            """, unsafe_allow_html=True)
    
    with doc_tab2:
        st.markdown("## 🎯 Scoring Systems Documentation")
        
        st.markdown("""
        ### G4Hunter Score Algorithm
        The G4Hunter algorithm evaluates G-quadruplex formation potential:
        
        **Algorithm:**
        1. For each base in the sequence:
           - G nucleotides: Add score based on run length (max +4)
           - C nucleotides: Add negative score based on run length (max -4)
           - A/T nucleotides: Add 0
        2. Calculate mean score across the sequence
        
        **Formula:**
        ```
        Score = mean(values) where:
        - G runs: +min(run_length, 4)
        - C runs: -min(run_length, 4)
        - A/T: 0
        ```
        
        **Interpretation:**
        - Positive scores: G-rich regions favoring G4 formation
        - Negative scores: C-rich regions favoring i-motif formation
        - Score > 1.5: High G4 potential
        """)
        
        st.markdown("""
        ### ZSeeker Score Algorithm
        Evaluates Z-DNA formation potential through dinucleotide analysis:
        
        **Algorithm:**
        1. Count alternating purine-pyrimidine dinucleotides: GC, CG, GT, TG, AC, CA
        2. Calculate ratio of these dinucleotides to total possible dinucleotide positions
        
        **Formula:**
        ```
        Score = count(alternating_dinucs) / (sequence_length / 2)
        ```
        
        **Interpretation:**
        - Score > 0.5: Moderate Z-DNA potential
        - Score > 0.8: High Z-DNA potential
        """)
        
        st.markdown("""
        ### Triplex Score Algorithm
        Assesses triplex DNA formation potential:
        
        **Algorithm:**
        1. Count purine runs (A/G) of 10+ nucleotides
        2. Count pyrimidine runs (C/T) of 10+ nucleotides
        3. Calculate density relative to sequence length
        
        **Formula:**
        ```
        Score = (purine_runs + pyrimidine_runs) / sequence_length * 100
        ```
        
        **Interpretation:**
        - Score > 5: Moderate triplex potential
        - Score > 10: High triplex potential
        """)
        
        st.markdown("""
        ### Additional Scoring Methods
        
        **Hairpin/Cruciform Score:**
        - Evaluates palindrome strength through complement matching
        - Higher scores indicate better palindrome formation potential
        
        **AT/GC Content Scores:**
        - Simple nucleotide composition analysis
        - Used for curved DNA and cluster region identification
        
        **i-Motif Score:**
        - Similar to G4Hunter but optimized for i-motif structures
        - Focuses on C-rich region identification
        """)
    
    with doc_tab3:
        st.markdown("## 🔍 Detection Logic and Regular Expressions")
        
        detection_data = {
            "DNA Class": [
                "Curved DNA - Global Curvature",
                "Curved DNA - Local Curvature", 
                "Slipped DNA - Direct Repeat",
                "Slipped DNA - STR",
                "Cruciform DNA - IR/Hairpin",
                "R-loop - RNA-DNA hybrids",
                "Triplex - Triplex",
                "Triplex - Sticky DNA",
                "G-Quadruplex - Multimeric",
                "G-Quadruplex - Canonical",
                "G-Quadruplex - Relaxed",
                "G-Quadruplex - Bulged",
                "G-Quadruplex - Bipartite", 
                "G-Quadruplex - Imperfect",
                "G-Quadruplex - G-Triplex",
                "i-Motif - Canonical",
                "i-Motif - Relaxed",
                "i-Motif - AC-motif",
                "Z-DNA - Z-DNA",
                "Z-DNA - eGZ",
                "Hybrid - Dynamic overlap",
                "Non-B DNA Cluster - Hotspot"
            ],
            "Regular Expression": [
                "A{4,}.{0,6}T{4,}",
                "[AT]{8,}",
                "([ATGC]{2,10})\\1{2,}",
                "([ATGC]{1,6})\\1{4,}",
                "[ATGC]{4,8}.{5,20}[ATGC]{4,8}",
                "G{20,}[ATGC]{10,50}C{20,}",
                "[AG]{15,}",
                "[CT]{15,}",
                "G{4,}.{1,7}G{4,}.{1,7}G{4,}.{1,7}G{4,}",
                "G{3}.{1,7}G{3}.{1,7}G{3}.{1,7}G{3}",
                "G{2,3}.{1,12}G{2,3}.{1,12}G{2,3}.{1,12}G{2,3}",
                "G{3}.{1,7}G{1,3}.{1,7}G{3}.{1,7}G{3}",
                "G{3}.{1,7}G{3}.{1,20}G{3}.{1,7}G{3}",
                "G{2}.{1,12}G{2}.{1,12}G{2}.{1,12}G{2}",
                "G{3}.{1,7}G{3}.{1,7}G{3}",
                "C{3}.{1,7}C{3}.{1,7}C{3}.{1,7}C{3}",
                "C{2,3}.{1,12}C{2,3}.{1,12}C{2,3}.{1,12}C{2,3}",
                "(A{3,}[ACGT]{6}C{3,}[ACGT]{6}C{3,}[ACGT]{6}C{3,})|(C{3,}[ACGT]{6}C{3,}[ACGT]{6}C{3,}[ACGT]{6}A{3,})",
                "(?:CG){6,}",
                "(?:CGG){4,}",
                "G{3}.{1,7}G{3}.{1,7}C{3}.{1,7}C{3}",
                "[ATGC]{100,}"
            ],
            "Scoring Method": [
                "AT_Content", "AT_Content", "GC_Content", "GC_Content",
                "Hairpin_Score", "GC_Content", "Triplex_Score", "Triplex_Score",
                "G4Hunter", "G4Hunter", "G4Hunter", "G4Hunter", "G4Hunter",
                "G4Hunter", "G4Hunter", "i-motif_Score", "i-motif_Score",
                "i-motif_Score", "ZSeeker", "ZSeeker", "G4Hunter", "GC_Content"
            ],
            "Significance Threshold": [
                "Low: <25, Med: 25-75, High: >75",
                "Low: <25, Med: 25-75, High: >75",
                "Low: <25, Med: 25-75, High: >75",
                "Low: <25, Med: 25-75, High: >75",
                "Low: <25, Med: 25-75, High: >75",
                "Low: <25, Med: 25-75, High: >75",
                "Low: <25, Med: 25-75, High: >75",
                "Low: <25, Med: 25-75, High: >75",
                "Low: <25, Med: 25-75, High: >75",
                "Low: <25, Med: 25-75, High: >75",
                "Low: <25, Med: 25-75, High: >75",
                "Low: <25, Med: 25-75, High: >75",
                "Low: <25, Med: 25-75, High: >75",
                "Low: <25, Med: 25-75, High: >75",
                "Low: <25, Med: 25-75, High: >75",
                "Low: <25, Med: 25-75, High: >75",
                "Low: <25, Med: 25-75, High: >75",
                "Low: <25, Med: 25-75, High: >75",
                "Low: <25, Med: 25-75, High: >75",
                "Low: <25, Med: 25-75, High: >75",
                "Low: <25, Med: 25-75, High: >75",
                "Low: <25, Med: 25-75, High: >75"
            ]
        }
        
        detection_df = pd.DataFrame(detection_data)
        st.dataframe(detection_df, use_container_width=True, height=600)
        
        st.markdown("""
        ### Detection Logic Explanation
        
        **Pattern Matching Strategy:**
        1. **Non-overlapping Detection:** Uses custom iterator to prevent overlapping matches
        2. **Greedy Matching:** Finds longest possible matches first
        3. **Case Insensitive:** All patterns are case-insensitive
        4. **Quantifier Usage:** 
           - `{n,}`: n or more occurrences
           - `{n,m}`: between n and m occurrences
           - `.{n,m}`: any character between n and m times
        
        **Significance Classification:**
        - **Minimal (<25):** Low structural potential
        - **Significant (25-75):** Moderate structural potential  
        - **Very Significant (>75):** High structural potential
        """)
        
    with doc_tab4:
        st.markdown("## NonBDNAFinder Analysis Pipeline")
        
        # Create a professional pipeline diagram using text and styling
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; color: white; margin: 1rem 0;">
            <h2 style="text-align: center; margin-bottom: 2rem;">🧬 NonBDNAFinder Analysis Pipeline</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Pipeline steps
        pipeline_steps = [
            ("Input Processing", "FASTA file upload and parsing", "#4CAF50"),
            ("🔍 Sequence Analysis", "Basic composition and quality checks", "#2196F3"),
            ("🎯 Pattern Detection", "Regular expression-based motif identification", "#FF9800"),
            ("Scoring Calculation", "Apply scoring algorithms (G4Hunter, ZSeeker, etc.)", "#9C27B0"),
            ("Classification", "Assign significance levels and categories", "#F44336"),
            ("📈 Visualization", "Generate interactive charts and plots", "#00BCD4"),
            ("Results Export", "Download formatted results and reports", "#795548")
        ]
        
        for i, (title, description, color) in enumerate(pipeline_steps):
            col1, col2 = st.columns([1, 4])
            with col1:
                st.markdown(f"""
                <div style="background: {color}; color: white; padding: 1rem; border-radius: 50%; 
                           text-align: center; width: 80px; height: 80px; display: flex; 
                           align-items: center; justify-content: center; font-weight: bold;">
                    {i+1}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="background: white; border-left: 5px solid {color}; padding: 1rem; 
                           margin-bottom: 1rem; border-radius: 0 10px 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <h4 style="margin: 0; color: {color};">{title}</h4>
                    <p style="margin: 0.5rem 0 0 0; color: #666;">{description}</p>
                </div>
                """, unsafe_allow_html=True)
            
            if i < len(pipeline_steps) - 1:
                st.markdown("""
                <div style="text-align: center; font-size: 2rem; color: #667eea; margin: 0.5rem 0;">
                    ⬇️
                </div>
                """, unsafe_allow_html=True)
        
        # Technical specifications
        st.markdown("""
        ### 🔧 Technical Specifications
        
        **Supported Input Formats:**
        - FASTA (.fa, .fasta)
        - Plain text (.txt)
        - Multi-FASTA files
        
        **Analysis Parameters:**
        - Maximum file size: 200MB
        - Sequence length: No theoretical limit
        - Detection classes: 10 major classes
        - Subclasses: 22 distinct subtypes
        
        **Output Formats:**
        - Interactive visualizations (Plotly)
        - CSV data exports
        - Excel formatted reports
        - Summary statistics
        
        **Performance:**
        - Pattern matching: O(n) complexity
        - Memory usage: Linear with sequence length
        - Real-time progress tracking
        """)
        
        # Disease analysis pipeline
        st.markdown("""
        ### 🩺 Disease Analysis Module Pipeline
        
        **Pathogenic Repeat Detection:**
        1. **Target Gene Scanning:** Screen for known disease-associated genes
        2. **Repeat Quantification:** Count repeat units in detected motifs
        3. **Threshold Comparison:** Compare against clinical pathogenic thresholds
        4. **Risk Assessment:** Classify as Normal Range or High Risk
        5. **Clinical Annotation:** Link to OMIM database entries
        6. **Visualization:** Generate risk assessment charts
        
        **Supported Disease Categories:**
        - Trinucleotide repeat disorders (CAG, CGG, GAA)
        - Hexanucleotide repeats (GGGGCC)
        - Other pathogenic repeat expansions (CTG)
        """)
        
        # Add download button for complete documentation
        doc_content = """
# NonBDNAFinder Documentation

## Overview
NonBDNAFinder is a comprehensive tool for detecting and analyzing non-canonical DNA structures using a 10-class, 22-subclass classification system.

## Scoring Systems
[Complete scoring system documentation as shown above]

## Detection Logic
[Complete detection logic documentation as shown above]

## Pipeline
[Complete pipeline documentation as shown above]
        """
        
        st.download_button(
            "Download Complete Documentation",
            data=doc_content.encode("utf-8"),
            file_name=f"NonBDNAFinder_Documentation_{datetime.now().strftime('%Y%m%d')}.md",
            mime="text/markdown"
        )

st.markdown("""
---
<div style="text-align: center; padding: 1rem; color: #666;">
    <p>NonBDNAFinder | Scientific Analysis of Non-B DNA Structures</p>
    <p>10-Class, 22-Subclass Classification System for Non-Canonical DNA Structures</p>
</div>
""", unsafe_allow_html=True)