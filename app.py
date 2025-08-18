
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
# Color Scheme for DNA Classes
# ======================
CLASS_COLORS = {
    'A-form DNA': '#FF6B6B',        # Coral red
    'Z-DNA': '#4ECDC4',             # Teal
    'Quadruplex': '#45B7D1',        # Sky blue
    'i-motif': '#96CEB4',           # Mint green
    'Triplex': '#FECA57',           # Yellow
    'Hairpin': '#FF9FF3',           # Pink
    'Cruciform': '#F38BA8',         # Rose
    'Slipped': '#A8E6CF',           # Light green
    'Bent DNA': '#FFB347',          # Orange
    'Supercoiled': '#DDA0DD'        # Plum
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
    initial_sidebar_state="collapsed"
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
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🧬 Non-B DNA Motif Finder</h1>
    <p>Comprehensive Detection of 10 Classes & 22 Subclasses of Non-B DNA Structures</p>
</div>
""", unsafe_allow_html=True)

# Information section
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div class="info-card">
        <h3>🔬 Advanced DNA Structure Analysis</h3>
        <p>This tool identifies and analyzes various non-canonical DNA structures that deviate from the standard B-form double helix. 
        These structures play crucial roles in gene regulation, DNA replication, and genomic stability.</p>
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

    # Comprehensive motif detection with 10 classes and 22 subclasses
    def non_overlapping_finditer(pattern, seq):
        regex = re.compile(pattern)
        pos = 0
        while pos < len(seq):
            match = regex.search(seq, pos)
            if not match:
                break
            yield match
            pos = match.end()

    def create_motif_dict(cls, subtype, match, seq, score_method="None", score="0", group=0):
        sequence = match.group(group)
        return {
            "Class": cls, 
            "Subtype": subtype, 
            "Start": match.start()+1,
            "End": match.start()+len(sequence), 
            "Length": len(sequence),
            "Sequence": wrap(sequence), 
            "ScoreMethod": score_method, 
            "Score": score
        }

    def find_motif(seq, pattern, cls, subtype, score_method="None", score_func=None, group=0):
        results = []
        for m in non_overlapping_finditer(pattern, seq):
            score = f"{score_func(m.group(group)):.2f}" if score_func else "0"
            results.append(create_motif_dict(cls, subtype, m, seq, score_method, score, group))
        return results

    # Progress indicator
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("🔍 Analyzing DNA sequence for non-B structures...")
    
    motifs = []
    total_steps = 22
    current_step = 0
    
    # Class 1: A-form DNA (2 subclasses)
    status_text.text("🔍 Detecting A-form DNA structures...")
    motifs += find_motif(sequence_input, r"[AG]{10,}", "A-form DNA", "Purine_Rich_A-form", "AT_Content", at_content)
    current_step += 1
    progress_bar.progress(current_step / total_steps)
    
    motifs += find_motif(sequence_input, r"[CT]{10,}", "A-form DNA", "Pyrimidine_Rich_A-form", "AT_Content", at_content)
    current_step += 1
    progress_bar.progress(current_step / total_steps)
    
    # Class 2: Z-DNA (3 subclasses)  
    status_text.text("🔍 Detecting Z-DNA structures...")
    motifs += find_motif(sequence_input, r"(?:CG){6,}", "Z-DNA", "CG_Repeat", "ZSeeker", zseeker_score)
    current_step += 1
    progress_bar.progress(current_step / total_steps)
    
    motifs += find_motif(sequence_input, r"(?:CA){4,}", "Z-DNA", "CA_Repeat", "ZSeeker", zseeker_score)
    current_step += 1
    progress_bar.progress(current_step / total_steps)
    
    motifs += find_motif(sequence_input, r"(?:GT){4,}", "Z-DNA", "GT_Repeat", "ZSeeker", zseeker_score)
    current_step += 1
    progress_bar.progress(current_step / total_steps)
    
    # Class 3: Quadruplex (3 subclasses)
    status_text.text("🔍 Detecting G-Quadruplex structures...")
    motifs += find_motif(sequence_input, r"G{3,}.{1,7}G{3,}.{1,7}G{3,}.{1,7}G{3,}", "Quadruplex", "Canonical_G-Quadruplex", "G4Hunter", g4hunter_score)
    current_step += 1
    progress_bar.progress(current_step / total_steps)
    
    motifs += find_motif(sequence_input, r"G{2}.{1,12}G{2}.{1,12}G{2}.{1,12}G{2}", "Quadruplex", "Non-canonical_G-Quadruplex", "G4Hunter", g4hunter_score)
    current_step += 1
    progress_bar.progress(current_step / total_steps)
    
    motifs += find_motif(sequence_input, r"G{3,}.{10,50}G{3,}.{10,50}G{3,}.{10,50}G{3,}", "Quadruplex", "Long_Loop_G-Quadruplex", "G4Hunter", g4hunter_score)
    current_step += 1
    progress_bar.progress(current_step / total_steps)
    
    # Class 4: i-motif (2 subclasses)
    status_text.text("🔍 Detecting i-motif structures...")
    motifs += find_motif(sequence_input, r"C{3,}.{1,7}C{3,}.{1,7}C{3,}.{1,7}C{3,}", "i-motif", "Canonical_i-motif", "i-motif_Score", imotif_score)
    current_step += 1
    progress_bar.progress(current_step / total_steps)
    
    motifs += find_motif(sequence_input, r"C{2}.{1,12}C{2}.{1,12}C{2}.{1,12}C{2}", "i-motif", "Non-canonical_i-motif", "i-motif_Score", imotif_score)
    current_step += 1
    progress_bar.progress(current_step / total_steps)
    
    # Class 5: Triplex (2 subclasses)
    status_text.text("🔍 Detecting Triplex DNA structures...")
    motifs += find_motif(sequence_input, r"[AG]{15,}", "Triplex", "H-DNA", "Triplex_Score", triplex_score)
    current_step += 1
    progress_bar.progress(current_step / total_steps)
    
    motifs += find_motif(sequence_input, r"[CT]{15,}", "Triplex", "Mirror_Repeats", "Triplex_Score", triplex_score)
    current_step += 1
    progress_bar.progress(current_step / total_steps)
    
    # Class 6: Hairpin (2 subclasses)
    status_text.text("🔍 Detecting Hairpin structures...")
    motifs += find_motif(sequence_input, r"[ATGC]{4,}.{4,20}[ATGC]{4,}", "Hairpin", "Palindromic_Sequences", "Hairpin_Score", hairpin_score)
    current_step += 1
    progress_bar.progress(current_step / total_steps)
    
    motifs += find_motif(sequence_input, r"[ATGC]{6,}.{3,15}[ATGC]{6,}", "Hairpin", "Inverted_Repeats", "Hairpin_Score", hairpin_score)
    current_step += 1
    progress_bar.progress(current_step / total_steps)
    
    # Class 7: Cruciform (2 subclasses)
    status_text.text("🔍 Detecting Cruciform structures...")
    motifs += find_motif(sequence_input, r"[ATGC]{8,}.{10,30}[ATGC]{8,}", "Cruciform", "Large_Palindromes", "Hairpin_Score", hairpin_score)
    current_step += 1
    progress_bar.progress(current_step / total_steps)
    
    motifs += find_motif(sequence_input, r"[ATGC]{4,8}.{5,20}[ATGC]{4,8}", "Cruciform", "Short_Palindromes", "Hairpin_Score", hairpin_score)
    current_step += 1
    progress_bar.progress(current_step / total_steps)
    
    # Class 8: Slipped structures (2 subclasses)
    status_text.text("🔍 Detecting Slipped structures...")
    motifs += find_motif(sequence_input, r"([ATGC]{2,10})\1{3,}", "Slipped", "Direct_Repeats", "Repeat_Score", lambda x: len(x))
    current_step += 1
    progress_bar.progress(current_step / total_steps)
    
    motifs += find_motif(sequence_input, r"([ATGC]{1,4})\1{5,}", "Slipped", "Short_Tandem_Repeats", "Repeat_Score", lambda x: len(x))
    current_step += 1
    progress_bar.progress(current_step / total_steps)
    
    # Class 9: Bent DNA (2 subclasses)
    status_text.text("🔍 Detecting Bent DNA structures...")
    motifs += find_motif(sequence_input, r"A{4,}.{0,6}T{4,}", "Bent DNA", "AT_Tracts", "AT_Content", at_content)
    current_step += 1
    progress_bar.progress(current_step / total_steps)
    
    motifs += find_motif(sequence_input, r"[AT]{10,}", "Bent DNA", "Phased_AT_Tracts", "AT_Content", at_content)
    current_step += 1
    progress_bar.progress(current_step / total_steps)
    
    # Class 10: Supercoiled DNA (2 subclasses)
    status_text.text("🔍 Detecting Supercoiled DNA structures...")
    motifs += find_motif(sequence_input, r"[ATGC]{200,}", "Supercoiled", "Negative_Supercoiling", "GC_Content", gc_content)
    current_step += 1
    progress_bar.progress(current_step / total_steps)
    
    motifs += find_motif(sequence_input, r"[GC]{100,}", "Supercoiled", "Positive_Supercoiling", "GC_Content", gc_content)
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
st.markdown("## 🧬 About Non-B DNA Structures")

info_cols = st.columns(2)

with info_cols[0]:
    st.markdown("""
    <div class="class-info" style="border-left-color: #FF6B6B;">
        <h4>🔴 A-form DNA</h4>
        <p>Right-handed double helix with wider major groove, typically forms under dehydrating conditions.</p>
    </div>
    
    <div class="class-info" style="border-left-color: #4ECDC4;">
        <h4>🟢 Z-DNA</h4>
        <p>Left-handed double helix formed by alternating purine-pyrimidine sequences, especially CG repeats.</p>
    </div>
    
    <div class="class-info" style="border-left-color: #45B7D1;">
        <h4>🔵 G-Quadruplex</h4>
        <p>Four-stranded structures formed by guanine-rich sequences, important in telomeres and gene regulation.</p>
    </div>
    
    <div class="class-info" style="border-left-color: #96CEB4;">
        <h4>🟦 i-motif</h4>
        <p>Four-stranded structures formed by cytosine-rich sequences, pH-dependent and complementary to G-quadruplexes.</p>
    </div>
    
    <div class="class-info" style="border-left-color: #FECA57;">
        <h4>🟡 Triplex DNA</h4>
        <p>Three-stranded structures formed by hydrogen bonding of a third strand to the major groove of duplex DNA.</p>
    </div>
    """, unsafe_allow_html=True)

with info_cols[1]:
    st.markdown("""
    <div class="class-info" style="border-left-color: #FF9FF3;">
        <h4>🟣 Hairpin Structures</h4>
        <p>Single-stranded DNA folds back on itself forming stem-loop structures with palindromic sequences.</p>
    </div>
    
    <div class="class-info" style="border-left-color: #F38BA8;">
        <h4>🌸 Cruciform Structures</h4>
        <p>Four-way junctions formed by large palindromic sequences, creating cross-like structures.</p>
    </div>
    
    <div class="class-info" style="border-left-color: #A8E6CF;">
        <h4>🟢 Slipped Structures</h4>
        <p>Formed by repetitive sequences that can slip during replication, creating loops and bulges.</p>
    </div>
    
    <div class="class-info" style="border-left-color: #FFB347;">
        <h4>🟠 Bent DNA</h4>
        <p>DNA with intrinsic curvature, often caused by AT-rich sequences and phased A-tracts.</p>
    </div>
    
    <div class="class-info" style="border-left-color: #DDA0DD;">
        <h4>🟣 Supercoiled DNA</h4>
        <p>Over- or under-wound DNA resulting from topological strain, important for gene regulation.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
---
<div style="text-align: center; padding: 1rem; color: #666;">
    <p>🧬 Non-B DNA Motif Finder | Advanced Genomic Structure Analysis Tool</p>
    <p>Detecting 10 Classes & 22 Subclasses of Non-Canonical DNA Structures</p>
</div>
""", unsafe_allow_html=True)
