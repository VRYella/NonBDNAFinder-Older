"""
Publication-Quality Visualization Suite for NBDFinder
Comprehensive high-resolution visualizations for scientific manuscripts

This module provides 10 major visualization types:
1. Bar Plots and Stacked Bar Plots
2. Linear Motif Maps (Genome Tracks)
3. Heatmaps
4. Pie/Donut Charts
5. Violin and Box Plots
6. UpSet Plots
7. Lollipop Plots
8. Bubble/Scatter Plots
9. Circos Plots
10. Sankey Diagrams
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from upsetplot import plot as upset_plot
from upsetplot import from_contents
import networkx as nx
import io
import base64
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Publication-quality color schemes
WONG_PALETTE = [
    '#E69F00',  # Orange
    '#56B4E9',  # Sky Blue
    '#009E73',  # Bluish Green
    '#F0E442',  # Yellow
    '#0072B2',  # Blue
    '#D55E00',  # Vermillion
    '#CC79A7',  # Reddish Purple
    '#000000'   # Black
]

NATURE_COLORS = {
    'Curved DNA': '#E69F00',
    'Slipped DNA': '#56B4E9',
    'Cruciform DNA': '#009E73',
    'R-loop': '#F0E442',
    'Triplex': '#0072B2',
    'G-Quadruplex Family': '#D55E00',
    'i-Motif Family': '#CC79A7',
    'Z-DNA': '#848484',
    'Hybrid': '#FF1493',
    'Non-B DNA Cluster Regions': '#8B4513'
}

class PublicationVisualizer:
    """Main class for generating publication-quality visualizations"""
    
    def __init__(self, motifs_df: pd.DataFrame, sequence_length: int = 5000):
        """
        Initialize the visualizer with motif data
        
        Args:
            motifs_df: DataFrame with motif detection results
            sequence_length: Length of the analyzed sequence
        """
        self.df = motifs_df.copy() if not motifs_df.empty else pd.DataFrame()
        self.sequence_length = sequence_length
        self.font_family = "Times New Roman"
        self.dpi = 300
        
    def set_publication_style(self):
        """Set matplotlib parameters for publication quality"""
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 11,
            'figure.titlesize': 16,
            'figure.dpi': self.dpi,
            'savefig.dpi': self.dpi,
            'savefig.bbox': 'tight',
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.linewidth': 1.2,
            'xtick.major.width': 1.2,
            'ytick.major.width': 1.2
        })
        
        # Set seaborn style
        sns.set_style("whitegrid", {"axes.spines.right": False, "axes.spines.top": False})
        sns.set_palette(WONG_PALETTE)

    def create_enhanced_bar_plots(self) -> Dict[str, go.Figure]:
        """Create publication-quality bar and stacked bar plots"""
        plots = {}
        
        if self.df.empty:
            return plots
            
        # 1. Class frequency bar plot
        class_counts = self.df['Class'].value_counts()
        
        fig_bar = go.Figure(data=[
            go.Bar(
                x=class_counts.index,
                y=class_counts.values,
                marker_color=[NATURE_COLORS.get(cls, '#888888') for cls in class_counts.index],
                text=class_counts.values,
                textposition='auto',
            )
        ])
        
        fig_bar.update_layout(
            title={
                'text': "Non-B DNA Motif Class Distribution",
                'font': {'family': self.font_family, 'size': 16}
            },
            xaxis_title="DNA Class",
            yaxis_title="Count",
            font=dict(family=self.font_family, size=12),
            plot_bgcolor='white',
            xaxis=dict(tickangle=45),
            height=500
        )
        plots['class_frequency'] = fig_bar
        
        # 2. Stacked bar plot by significance
        if 'Significance' in self.df.columns:
            sig_stack = pd.crosstab(self.df['Class'], self.df['Significance'])
            
            fig_stack = go.Figure()
            significance_colors = {
                'Low Stability': '#FFF2CC',
                'Moderate Stability': '#FFD966', 
                'High Stability': '#F1C232'
            }
            
            for sig in sig_stack.columns:
                fig_stack.add_trace(go.Bar(
                    name=sig,
                    x=sig_stack.index,
                    y=sig_stack[sig],
                    marker_color=significance_colors.get(sig, '#888888')
                ))
            
            fig_stack.update_layout(
                title={
                    'text': "Motif Distribution by Structural Stability",
                    'font': {'family': self.font_family, 'size': 16}
                },
                xaxis_title="DNA Class",
                yaxis_title="Count",
                font=dict(family=self.font_family, size=12),
                barmode='stack',
                plot_bgcolor='white',
                xaxis=dict(tickangle=45),
                height=500
            )
            plots['significance_stack'] = fig_stack
        
        return plots

    def create_linear_motif_maps(self) -> Dict[str, go.Figure]:
        """Create genome track-style linear motif maps"""
        plots = {}
        
        if self.df.empty:
            return plots
            
        # Create main genome track visualization
        fig = go.Figure()
        
        # Add tracks for each class
        classes = self.df['Class'].unique()
        y_positions = {cls: i for i, cls in enumerate(classes)}
        
        for cls in classes:
            class_df = self.df[self.df['Class'] == cls]
            
            # Add rectangles for each motif
            for _, row in class_df.iterrows():
                fig.add_shape(
                    type="rect",
                    x0=row['Start'],
                    y0=y_positions[cls] - 0.3,
                    x1=row['End'],
                    y1=y_positions[cls] + 0.3,
                    fillcolor=NATURE_COLORS.get(cls, '#888888'),
                    opacity=0.7,
                    line=dict(width=0)
                )
                
                # Add score information as text
                fig.add_annotation(
                    x=(row['Start'] + row['End']) / 2,
                    y=y_positions[cls],
                    text=f"{row['Score']:.1f}",
                    showarrow=False,
                    font=dict(size=8, color='white'),
                    bgcolor='rgba(0,0,0,0.5)',
                    bordercolor='white',
                    borderwidth=1
                )
        
        fig.update_layout(
            title={
                'text': "Linear Motif Map - Genomic Distribution",
                'font': {'family': self.font_family, 'size': 16}
            },
            xaxis_title="Sequence Position (bp)",
            yaxis_title="DNA Class",
            font=dict(family=self.font_family, size=12),
            yaxis=dict(
                tickmode='array',
                tickvals=list(y_positions.values()),
                ticktext=list(y_positions.keys()),
                tickangle=0
            ),
            plot_bgcolor='white',
            height=max(400, len(classes) * 60),
            xaxis=dict(range=[0, self.sequence_length])
        )
        
        plots['linear_map'] = fig
        
        return plots

    def create_publication_heatmaps(self) -> Dict[str, go.Figure]:
        """Create publication-quality heatmaps"""
        plots = {}
        
        if self.df.empty:
            return plots
            
        # 1. Motif density heatmap
        bin_size = max(10, self.sequence_length // 50)
        bins = np.arange(0, self.sequence_length + bin_size, bin_size)
        
        heatmap_data = []
        classes = self.df['Class'].unique()
        
        for cls in classes:
            class_df = self.df[self.df['Class'] == cls]
            counts, _ = np.histogram(class_df['Start'], bins=bins)
            heatmap_data.append(counts)
        
        if heatmap_data:
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=heatmap_data,
                x=bins[:-1],
                y=classes,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Motif Count")
            ))
            
            fig_heatmap.update_layout(
                title={
                    'text': "Motif Density Distribution",
                    'font': {'family': self.font_family, 'size': 16}
                },
                xaxis_title="Sequence Position (bp)",
                yaxis_title="DNA Class",
                font=dict(family=self.font_family, size=12),
                plot_bgcolor='white',
                height=400
            )
            plots['density_heatmap'] = fig_heatmap
        
        # 2. Score correlation heatmap
        if len(self.df) > 1:
            numerical_cols = ['Score', 'Length', 'Start', 'End']
            available_cols = [col for col in numerical_cols if col in self.df.columns]
            
            if len(available_cols) >= 2:
                corr_matrix = self.df[available_cols].corr()
                
                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu_r',
                    zmid=0,
                    text=np.round(corr_matrix.values, 2),
                    texttemplate="%{text}",
                    textfont={"size": 12},
                    showscale=True,
                    colorbar=dict(title="Correlation")
                ))
                
                fig_corr.update_layout(
                    title={
                        'text': "Feature Correlation Matrix",
                        'font': {'family': self.font_family, 'size': 16}
                    },
                    font=dict(family=self.font_family, size=12),
                    plot_bgcolor='white',
                    height=400
                )
                plots['correlation_heatmap'] = fig_corr
        
        return plots

    def create_enhanced_pie_charts(self) -> Dict[str, go.Figure]:
        """Create publication-quality pie and donut charts"""
        plots = {}
        
        if self.df.empty:
            return plots
            
        # 1. Class distribution pie chart
        class_counts = self.df['Class'].value_counts()
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=class_counts.index,
            values=class_counts.values,
            marker_colors=[NATURE_COLORS.get(cls, '#888888') for cls in class_counts.index],
            textinfo='label+percent',
            textposition='auto',
            textfont=dict(family=self.font_family, size=11)
        )])
        
        fig_pie.update_layout(
            title={
                'text': "Non-B DNA Class Composition",
                'font': {'family': self.font_family, 'size': 16}
            },
            font=dict(family=self.font_family, size=12),
            height=500,
            showlegend=True
        )
        plots['class_pie'] = fig_pie
        
        # 2. Significance donut chart
        if 'Significance' in self.df.columns:
            sig_counts = self.df['Significance'].value_counts()
            
            fig_donut = go.Figure(data=[go.Pie(
                labels=sig_counts.index,
                values=sig_counts.values,
                hole=0.4,
                marker_colors=['#FFF2CC', '#FFD966', '#F1C232'],
                textinfo='label+percent',
                textposition='auto',
                textfont=dict(family=self.font_family, size=11)
            )])
            
            fig_donut.update_layout(
                title={
                    'text': "Structural Stability Distribution",
                    'font': {'family': self.font_family, 'size': 16}
                },
                font=dict(family=self.font_family, size=12),
                height=500,
                annotations=[dict(text='Stability', x=0.5, y=0.5, font_size=14, showarrow=False)]
            )
            plots['significance_donut'] = fig_donut
        
        return plots

    def create_violin_box_plots(self) -> Dict[str, go.Figure]:
        """Create publication-quality violin and box plots"""
        plots = {}
        
        if self.df.empty:
            return plots
            
        # 1. Score distribution violin plot
        fig_violin = go.Figure()
        
        for cls in self.df['Class'].unique():
            class_data = self.df[self.df['Class'] == cls]['Score']
            
            fig_violin.add_trace(go.Violin(
                y=class_data,
                name=cls,
                box_visible=True,
                meanline_visible=True,
                fillcolor=NATURE_COLORS.get(cls, '#888888'),
                opacity=0.7,
                line_color='black'
            ))
        
        fig_violin.update_layout(
            title={
                'text': "Score Distribution by DNA Class",
                'font': {'family': self.font_family, 'size': 16}
            },
            xaxis_title="DNA Class",
            yaxis_title="Normalized Score (1-3)",
            font=dict(family=self.font_family, size=12),
            plot_bgcolor='white',
            height=500,
            xaxis=dict(tickangle=45)
        )
        plots['score_violin'] = fig_violin
        
        # 2. Length distribution box plot
        fig_box = go.Figure()
        
        for cls in self.df['Class'].unique():
            class_data = self.df[self.df['Class'] == cls]['Length']
            
            fig_box.add_trace(go.Box(
                y=class_data,
                name=cls,
                marker_color=NATURE_COLORS.get(cls, '#888888'),
                boxpoints='outliers'
            ))
        
        fig_box.update_layout(
            title={
                'text': "Motif Length Distribution by Class",
                'font': {'family': self.font_family, 'size': 16}
            },
            xaxis_title="DNA Class",
            yaxis_title="Length (bp)",
            font=dict(family=self.font_family, size=12),
            plot_bgcolor='white',
            height=500,
            xaxis=dict(tickangle=45)
        )
        plots['length_box'] = fig_box
        
        return plots

    def create_upset_plots(self) -> Dict[str, Any]:
        """Create UpSet plots for motif intersections"""
        plots = {}
        
        if self.df.empty:
            return plots
            
        try:
            # Create overlapping regions analysis
            # Bin the sequence and check for overlaps
            bin_size = max(100, self.sequence_length // 100)
            bins = np.arange(0, self.sequence_length + bin_size, bin_size)
            
            # Create sets for each class
            class_sets = {}
            for cls in self.df['Class'].unique():
                class_df = self.df[self.df['Class'] == cls]
                bins_with_motifs = set()
                
                for _, row in class_df.iterrows():
                    start_bin = int(row['Start'] // bin_size)
                    end_bin = int(row['End'] // bin_size)
                    bins_with_motifs.update(range(start_bin, end_bin + 1))
                
                class_sets[cls] = bins_with_motifs
            
            # Create UpSet plot data
            if len(class_sets) >= 2:
                self.set_publication_style()
                
                fig, ax = plt.subplots(figsize=(12, 8))
                upset_data = from_contents(class_sets)
                
                if len(upset_data) > 0:
                    upset_plot(upset_data, ax=ax, show_counts=True)
                    plt.title("Motif Class Intersections", fontsize=16, fontfamily=self.font_family)
                    plt.tight_layout()
                    
                    # Convert to base64 for web display
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png', dpi=self.dpi, bbox_inches='tight')
                    buffer.seek(0)
                    plots['upset_matplotlib'] = buffer
                    plt.close()
                
        except Exception as e:
            print(f"Error creating UpSet plot: {e}")
        
        return plots

    def create_lollipop_plots(self) -> Dict[str, go.Figure]:
        """Create lollipop plots for genomic annotations"""
        plots = {}
        
        if self.df.empty:
            return plots
            
        # Create lollipop plot for high-scoring motifs
        high_score_df = self.df[self.df['Score'] >= 2.5].copy()
        
        if len(high_score_df) > 0:
            fig = go.Figure()
            
            # Add stems
            for _, row in high_score_df.iterrows():
                fig.add_trace(go.Scatter(
                    x=[row['Start'], row['Start']],
                    y=[0, row['Score']],
                    mode='lines',
                    line=dict(color=NATURE_COLORS.get(row['Class'], '#888888'), width=3),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Add lollipop heads
            fig.add_trace(go.Scatter(
                x=high_score_df['Start'],
                y=high_score_df['Score'],
                mode='markers',
                marker=dict(
                    size=12,
                    color=[NATURE_COLORS.get(cls, '#888888') for cls in high_score_df['Class']],
                    line=dict(width=2, color='white')
                ),
                text=[f"{row['Class']}<br>{row['Subtype']}<br>Score: {row['Score']:.2f}" 
                      for _, row in high_score_df.iterrows()],
                hovertemplate='%{text}<br>Position: %{x}<extra></extra>',
                name='High Stability Motifs'
            ))
            
            fig.update_layout(
                title={
                    'text': "High Stability Motif Positions",
                    'font': {'family': self.font_family, 'size': 16}
                },
                xaxis_title="Sequence Position (bp)",
                yaxis_title="Stability Score",
                font=dict(family=self.font_family, size=12),
                plot_bgcolor='white',
                height=500,
                yaxis=dict(range=[0, 3.2])
            )
            plots['lollipop_high_score'] = fig
        
        return plots

    def create_bubble_scatter_plots(self) -> Dict[str, go.Figure]:
        """Create bubble and scatter plots with trend lines"""
        plots = {}
        
        if self.df.empty:
            return plots
            
        # 1. Score vs Length bubble plot
        fig_bubble = go.Figure()
        
        for cls in self.df['Class'].unique():
            class_df = self.df[self.df['Class'] == cls]
            
            fig_bubble.add_trace(go.Scatter(
                x=class_df['Length'],
                y=class_df['Score'],
                mode='markers',
                name=cls,
                marker=dict(
                    size=8,
                    color=NATURE_COLORS.get(cls, '#888888'),
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                text=[f"{row['Subtype']}<br>Position: {row['Start']}" 
                      for _, row in class_df.iterrows()],
                hovertemplate='%{text}<br>Length: %{x}<br>Score: %{y}<extra></extra>'
            ))
        
        fig_bubble.update_layout(
            title={
                'text': "Motif Stability vs Length Relationship",
                'font': {'family': self.font_family, 'size': 16}
            },
            xaxis_title="Motif Length (bp)",
            yaxis_title="Stability Score (1-3)",
            font=dict(family=self.font_family, size=12),
            plot_bgcolor='white',
            height=500
        )
        plots['score_length_bubble'] = fig_bubble
        
        return plots

    def create_comprehensive_suite(self) -> Dict[str, Dict[str, Any]]:
        """Create the complete publication visualization suite"""
        suite = {}
        
        print("🎨 Generating publication-quality visualizations...")
        
        # Generate all visualization types
        suite['bar_plots'] = self.create_enhanced_bar_plots()
        suite['linear_maps'] = self.create_linear_motif_maps()
        suite['heatmaps'] = self.create_publication_heatmaps()
        suite['pie_charts'] = self.create_enhanced_pie_charts()
        suite['violin_box'] = self.create_violin_box_plots()
        suite['upset_plots'] = self.create_upset_plots()
        suite['lollipop'] = self.create_lollipop_plots()
        suite['bubble_scatter'] = self.create_bubble_scatter_plots()
        
        print(f"✅ Generated {sum(len(plots) for plots in suite.values())} publication-quality plots")
        
        return suite

def create_comprehensive_publication_suite(motifs_df: pd.DataFrame, sequence_length: int = 5000) -> Dict[str, Dict[str, Any]]:
    """
    Main function to create comprehensive publication visualization suite
    
    Args:
        motifs_df: DataFrame with motif detection results
        sequence_length: Length of the analyzed sequence
        
    Returns:
        Dictionary containing all visualization categories and plots
    """
    visualizer = PublicationVisualizer(motifs_df, sequence_length)
    return visualizer.create_comprehensive_suite()

def export_all_formats(fig, filename: str, output_dir: str = "publication_figures") -> Dict[str, str]:
    """
    Export figure in all publication formats
    
    Args:
        fig: Plotly figure object
        filename: Base filename
        output_dir: Output directory
        
    Returns:
        Dictionary with format: filepath mappings
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    exported_files = {}
    
    try:
        # PNG (high resolution)
        png_path = os.path.join(output_dir, f"{filename}.png")
        fig.write_image(png_path, format="png", width=1200, height=800, scale=2.5)
        exported_files['png'] = png_path
        
        # PDF (vector)
        pdf_path = os.path.join(output_dir, f"{filename}.pdf")
        fig.write_image(pdf_path, format="pdf", width=1200, height=800)
        exported_files['pdf'] = pdf_path
        
        # SVG (vector)
        svg_path = os.path.join(output_dir, f"{filename}.svg")
        fig.write_image(svg_path, format="svg", width=1200, height=800)
        exported_files['svg'] = svg_path
        
    except Exception as e:
        print(f"Warning: Could not export {filename} in some formats: {e}")
    
    return exported_files