
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

def triplex_score(seq: str) -> float:
    """Score for triplex DNA formation potential"""
    purine_runs = len(re.findall(r"[AG]{10,}", seq))
    pyrimidine_runs = len(re.findall(r"[CT]{10,}", seq))
    return (purine_runs + pyrimidine_runs) / len(seq) * 100 if len(seq) > 0 else 0.0

def hairpin_score(seq: str) -> float:
    """Score for hairpin/cruciform potential based on palindrome strength"""
    if len(seq) < 6:
        return 0.0
    # Simple scoring based on inverted repeat potential
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    reverse_comp = ''.join(complement.get(base, 'N') for base in seq[::-1])
    matches = sum(1 for a, b in zip(seq, reverse_comp) if a == b)
    return matches / len(seq) * 100

def at_content(seq: str) -> float:
    """Calculate AT content"""
    at_count = seq.count('A') + seq.count('T')
    return at_count / len(seq) * 100 if len(seq) > 0 else 0.0

def gc_content(seq: str) -> float:
    """Calculate GC content"""
    gc_count = seq.count('G') + seq.count('C')
    return gc_count / len(seq) * 100 if len(seq) > 0 else 0.0

def imotif_score(seq: str) -> float:
    """Score for i-motif formation potential"""
    c_runs = re.findall(r"C{3,}", seq)
    if not c_runs:
        return 0.0
    total_score = sum(len(run) for run in c_runs)
    return total_score / len(seq) * 100

# ======================
# Color Scheme for DNA Classes (Updated for New Classification)
# ======================
CLASS_COLORS = {
    'Curved DNA': '#FF6B6B',                    # Red
    'Slipped DNA': '#4ECDC4',                   # Teal
    'Cruciform DNA': '#45B7D1',                 # Sky blue
    'R-loop': '#96CEB4',                        # Mint green
    'Triplex': '#FECA57',                       # Yellow
    'G-Quadruplex Family': '#FF9FF3',           # Pink
    'i-Motif Family': '#F38BA8',                # Rose
    'Z-DNA': '#A8E6CF',                         # Light green
    'Hybrid': '#FFB347',                        # Orange
    'Non-B DNA Cluster Regions': '#DDA0DD'     # Plum
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

# Page configuration with custom styling
st.set_page_config(
    page_title="Non-B DNA Motif Finder", 
    layout="wide",
    page_icon="🧬",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .info-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .class-info {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: transform 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .disease-card {
        background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #e53e3e;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a section:", 
    ["Main Analysis", "Disease Analysis", "About"])

if page == "Main Analysis":

if page == "Main Analysis":
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🧬 Non-B DNA Motif Finder</h1>
        <p>10-Class, 22-Subclass Classification System for Non-B DNA Structures</p>
    </div>
    """, unsafe_allow_html=True)

    # Information section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>🔬 Advanced DNA Structure Analysis</h3>
            <p>Non-canonical DNA structures play key roles in genome stability, regulation, and evolution.
            This application detects and analyzes 18 distinct Non-B DNA motifs in any DNA sequence or multi-FASTA file.
            Motif Classes: G-quadruplex-related (G4, Relaxed G4, Bulged G4, Bipartite G4, Multimeric G4, G-Triplex, i-Motif, Hybrid),
            helix/curvature (Z-DNA, eGZ (Extruded-G), Curved DNA, AC-Motif),
            repeat/junction (Slipped DNA, Cruciform, Sticky DNA, Triplex DNA),
            hybrid/cluster (R-Loop, Non-B DNA Clusters).
            Upload single or multi-FASTA files for comprehensive analysis.</p>
        </div>
        """, unsafe_allow_html=True)

    # File upload section
    st.markdown("### 📁 Upload Your DNA Sequence")
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded = st.file_uploader("Choose a FASTA file", type=["fa", "fasta", "txt"], 
                                   help="Upload a FASTA file containing DNA sequences")

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        use_example = st.button("🧪 Use Example Sequence", help="Load a sample sequence for testing")

    sequence_input = ""

    if uploaded:
        try:
            sequence_input = parse_fasta(uploaded.read().decode())
            st.success("✅ Sequence uploaded successfully!")
        except:
            st.error("❌ Invalid FASTA format. Please check your file.")
    elif use_example:
        sequence_input = parse_fasta(EXAMPLE_FASTA)
        st.info("🧪 Example sequence loaded!")

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
        with st.expander("📋 View Sequence", expanded=False):
                st.text_area("Input Sequence", sequence_input, height=150)

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
        for m in non_overlapping_finditer(pattern, seq):
            score = score_func(m.group(group)) if score_func else 0
            results.append(create_motif_dict(cls, subtype, m, seq, score_method, score, group))
        return results

    # Progress indicator
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("🔍 Analyzing DNA sequence for non-B structures...")
    
    motifs = []
    total_steps = 22
    current_step = 0
    
    # 1. Curved DNA (2 subclasses)
    status_text.text("🔍 Detecting Curved DNA structures...")
    motifs += find_motif(sequence_input, r"A{4,}.{0,6}T{4,}", "Curved DNA", "Global Curvature", "AT_Content", at_content)
    current_step += 1
    progress_bar.progress(current_step / total_steps)
    
    motifs += find_motif(sequence_input, r"[AT]{8,}", "Curved DNA", "Local Curvature", "AT_Content", at_content)
    current_step += 1
    progress_bar.progress(current_step / total_steps)
    
    # 2. Slipped DNA (2 subclasses)
    status_text.text("🔍 Detecting Slipped DNA structures...")
    motifs += find_motif(sequence_input, r"([ATGC]{2,10})\1{3,}", "Slipped DNA", "Direct Repeat", "Repeat_Score", lambda x: len(x))
    current_step += 1
    progress_bar.progress(current_step / total_steps)
    
    motifs += find_motif(sequence_input, r"([ATGC]{1,4})\1{5,}", "Slipped DNA", "STR", "Repeat_Score", lambda x: len(x))
    current_step += 1
    progress_bar.progress(current_step / total_steps)
    
    # 3. Cruciform DNA (2 subclasses)
    status_text.text("🔍 Detecting Cruciform DNA structures...")
    motifs += find_motif(sequence_input, r"[ATGC]{8,}.{10,30}[ATGC]{8,}", "Cruciform DNA", "IR/Hairpin structures", "Hairpin_Score", hairpin_score)
    current_step += 1
    progress_bar.progress(current_step / total_steps)
    
    motifs += find_motif(sequence_input, r"[ATGC]{4,8}.{5,20}[ATGC]{4,8}", "Cruciform DNA", "IR/Hairpin structures", "Hairpin_Score", hairpin_score)
    current_step += 1
    progress_bar.progress(current_step / total_steps)
    
    # 4. R-loop (1 subclass)
    status_text.text("🔍 Detecting R-loop structures...")
    motifs += find_motif(sequence_input, r"G{20,}[ATGC]{10,50}C{20,}", "R-loop", "RNA-DNA hybrids", "GC_Content", gc_content)
    current_step += 1
    progress_bar.progress(current_step / total_steps)
    
    # 5. Triplex (2 subclasses)
    status_text.text("🔍 Detecting Triplex structures...")
    motifs += find_motif(sequence_input, r"[AG]{15,}", "Triplex", "Triplex", "Triplex_Score", triplex_score)
    current_step += 1
    progress_bar.progress(current_step / total_steps)
    
    motifs += find_motif(sequence_input, r"[CT]{15,}", "Triplex", "Sticky DNA", "Triplex_Score", triplex_score)
    current_step += 1
    progress_bar.progress(current_step / total_steps)
    
    # 6. G-Quadruplex Family (7 subclasses)
    status_text.text("🔍 Detecting G-Quadruplex Family structures...")
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
    status_text.text("🔍 Detecting i-Motif Family structures...")
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
    status_text.text("🔍 Detecting Z-DNA structures...")
    motifs += find_motif(sequence_input, r"(?:CG){6,}", "Z-DNA", "Z-DNA", "ZSeeker", zseeker_score)
    current_step += 1
    progress_bar.progress(current_step / total_steps)
    
    motifs += find_motif(sequence_input, r"(?:CGG){4,}", "Z-DNA", "eGZ", "ZSeeker", zseeker_score)
    current_step += 1
    progress_bar.progress(current_step / total_steps)
    
    # 9. Hybrid (1 subclass)
    status_text.text("🔍 Detecting Hybrid structures...")
    motifs += find_motif(sequence_input, r"G{3}.{1,7}G{3}.{1,7}C{3}.{1,7}C{3}", "Hybrid", "Dynamic overlap regions", "G4Hunter", g4hunter_score)
    current_step += 1
    progress_bar.progress(current_step / total_steps)
    
    # 10. Non-B DNA Cluster Regions (1 subclass)
    status_text.text("🔍 Detecting Non-B DNA Cluster Regions...")
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
        st.markdown("## 📊 Analysis Results")
        
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
        st.markdown("### 🎨 Class Distribution")
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
        
        # Subtype analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔬 Subtype Distribution")
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
        
        with col2:
            st.markdown("### 📏 Length Distribution")
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
        
        # Sequence position visualization
        st.markdown("### 📍 Motif Positions Along Sequence")
        
        fig4 = px.scatter(
            df,
            x='Start',
            y='Class',
            color='Class',
            size='Length',
            hover_data=['Subtype', 'Score'],
            color_discrete_map=CLASS_COLORS,
            title="Motif Positions and Sizes"
        )
        fig4.update_layout(height=400)
        st.plotly_chart(fig4, use_container_width=True)
        
        # Detailed results table
        st.markdown("### 📋 Detailed Results")
        
        # Add color coding to the dataframe display
        def color_class(val):
            color = CLASS_COLORS.get(val, '#FFFFFF')
            return f'background-color: {color}; color: white; font-weight: bold;'
        
        # Style the dataframe
        styled_df = df.style.applymap(color_class, subset=['Class'])
        
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        # Download section
        st.markdown("### 💾 Download Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📄 Download CSV",
                data=csv,
                file_name=f"non_b_dna_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Create Excel file
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='Non-B DNA Results', index=False)
                
                # Get workbook and worksheet
                workbook = writer.book
                worksheet = writer.sheets['Non-B DNA Results']
                
                # Add some formatting
                header_format = workbook.add_format({
                    'bold': True,
                    'text_wrap': True,
                    'valign': 'top',
                    'fg_color': '#667eea',
                    'font_color': 'white',
                    'border': 1
                })
                
                # Apply header format
                for col_num, value in enumerate(df.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                    
            excel_data = output.getvalue()
            
            st.download_button(
                "📊 Download Excel",
                data=excel_data,
                file_name=f"non_b_dna_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col3:
            # Create summary report
            summary_text = f"""
Non-B DNA Analysis Summary
=========================
Sequence Length: {len(sequence_input)} bp
GC Content: {gc_content(sequence_input):.1f}%
AT Content: {at_content(sequence_input):.1f}%

Total Motifs Found: {len(df)}
Classes Detected: {unique_classes}
Subclasses Found: {unique_subtypes}
Sequence Coverage: {coverage:.1f}%

Class Breakdown:
{chr(10).join([f"- {cls}: {count} motifs" for cls, count in class_counts.items()])}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            st.download_button(
                "📝 Download Summary",
                data=summary_text,
                file_name=f"non_b_dna_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    else:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); border-radius: 10px; margin: 2rem 0;">
            <h3>🔍 No Non-B DNA Motifs Found</h3>
            <p>The sequence appears to contain primarily standard B-form DNA structures.</p>
            <p>Try uploading a different sequence or use the example sequence to see detected motifs.</p>
        </div>
        """, unsafe_allow_html=True)

# Information about DNA classes
st.markdown("---")
st.markdown("## 🧬 About Non-B DNA Structures - 10-Class, 22-Subclass Classification System")

st.markdown("""
The first standardized taxonomic framework for Non-B DNA structures includes examples from human sequences, including human mitochondrial DNA (NC_012920.1), human telomerase RNA (NR_003287.2), human ADAR1 gene (NM_001126112.2), and human SNRPN gene (NR_024540.1).
""")

info_cols = st.columns(2)

with info_cols[0]:
    st.markdown("""
    <div class="class-info" style="border-left-color: #FF6B6B;">
        <h4>🔴 Curved DNA</h4>
        <p>Global Curvature: DNA with intrinsic bending due to sequence-specific features. Local Curvature: Short-range bends caused by specific base arrangements.</p>
    </div>
    
    <div class="class-info" style="border-left-color: #4ECDC4;">
        <h4>🟢 Slipped DNA</h4>
        <p>Direct Repeat: Formed when repetitive sequences slip during replication. STR: Short Tandem Repeats creating secondary structures.</p>
    </div>
    
    <div class="class-info" style="border-left-color: #45B7D1;">
        <h4>🔵 Cruciform DNA</h4>
        <p>IR/Hairpin structures: Four-way junctions and stem-loop structures formed by inverted repeat sequences creating cross-like formations.</p>
    </div>
    
    <div class="class-info" style="border-left-color: #96CEB4;">
        <h4>🟦 R-loop</h4>
        <p>RNA-DNA hybrids: Three-stranded structures where RNA displaces one DNA strand, forming RNA-DNA hybrid with displaced single-stranded DNA loop.</p>
    </div>
    
    <div class="class-info" style="border-left-color: #FECA57;">
        <h4>🟡 Triplex</h4>
        <p>Triplex: Three-stranded DNA with third strand in major groove. Sticky DNA: Transient triplex intermediates with sequence-specific binding properties.</p>
    </div>
    """, unsafe_allow_html=True)

with info_cols[1]:
    st.markdown("""
    <div class="class-info" style="border-left-color: #FF9FF3;">
        <h4>🟣 G-Quadruplex Family</h4>
        <p>7 subclasses: Multimeric, Canonical, Relaxed, Bulged, Bipartite, Imperfect, G-Triplex. Four-stranded structures formed by guanine-rich sequences, critical for telomeres and gene regulation.</p>
    </div>
    
    <div class="class-info" style="border-left-color: #F38BA8;">
        <h4>🌸 i-Motif Family</h4>
        <p>3 subclasses: Canonical, Relaxed, AC-motif. Four-stranded structures formed by cytosine-rich sequences, pH-dependent and complementary to G-quadruplexes.</p>
    </div>
    
    <div class="class-info" style="border-left-color: #A8E6CF;">
        <h4>🟢 Z-DNA</h4>
        <p>Z-DNA: Left-handed double helix formed by alternating purine-pyrimidine sequences. eGZ: Extended G-Z junctions with unique structural properties.</p>
    </div>
    
    <div class="class-info" style="border-left-color: #FFB347;">
        <h4>🟠 Hybrid</h4>
        <p>Dynamic overlap regions: Areas where multiple Non-B DNA structures can coexist or interchange, creating complex structural landscapes.</p>
    </div>
    
    <div class="class-info" style="border-left-color: #DDA0DD;">
        <h4>🟣 Non-B DNA Cluster Regions</h4>
        <p>Hotspot regions: Genomic areas with high density of Non-B DNA forming sequences, often associated with replication stress and genomic instability.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
---
<div style="text-align: center; padding: 1rem; color: #666;">
    <p>🧬 Non-B DNA Motif Finder | Advanced Genomic Structure Analysis Tool</p>
    <p>10-Class, 22-Subclass Classification System for Non-Canonical DNA Structures</p>
</div>
""", unsafe_allow_html=True)
