"""
utils.py

This module contains utility functions for generating interactive Plotly charts,
and exporting prediction reports in CSV and PDF formats.
"""

import io
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple, List
import plotly.graph_objects as go
import plotly.express as px

# ReportLab imports
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- PLOTLY INTERACTIVE CHARTS ---

def plot_confusion_matrix(cm_data: Dict[str, Any], is_dark: bool = False) -> go.Figure:
    """
    Creates an interactive Plotly heatmap for the confusion matrix with theme support.
    """
    matrix = np.array(cm_data['matrix'])
    
    # Heatmap text annotations
    annot = [
        [f"True Negative<br><b>{matrix[0][0]}</b>", f"False Positive<br><b>{matrix[0][1]}</b>"],
        [f"False Negative<br><b>{matrix[1][0]}</b>", f"True Positive<br><b>{matrix[1][1]}</b>"]
    ]
    
    text_color = '#F8FAFC' if is_dark else '#0F172A'
    grid_color = '#334155' if is_dark else '#E2E8F0'
    colorscale = [[0.0, '#1E293B' if is_dark else '#EFF6FF'], [1.0, '#2563EB']]
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=['Predicted: Fully Repaid', 'Predicted: Default Risk'],
        y=['Actual: Fully Repaid', 'Actual: Default Risk'],
        hoverinfo='none',
        colorscale=colorscale,
        text=annot,
        texttemplate="%{text}",
        showscale=False
    ))
    
    fig.update_layout(
        title={
            'text': "Confusion Matrix Matrix",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 16, 'color': text_color, 'family': "'Plus Jakarta Sans', sans-serif"}
        },
        xaxis=dict(
            title=dict(text="Predicted Label", font=dict(color=text_color, size=12)),
            tickfont=dict(color=text_color, size=10),
            gridcolor=grid_color,
            zeroline=False
        ),
        yaxis=dict(
            title=dict(text="True Label", font=dict(color=text_color, size=12)),
            tickfont=dict(color=text_color, size=10),
            gridcolor=grid_color,
            zeroline=False
        ),
        height=320,
        margin=dict(l=50, r=50, t=50, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, is_dark: bool = False) -> go.Figure:
    """
    Creates an interactive Plotly line chart for the ROC curve with theme support.
    """
    fig = go.Figure()
    
    text_color = '#F8FAFC' if is_dark else '#0F172A'
    grid_color = '#334155' if is_dark else '#E2E8F0'
    
    # ROC Line
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.4f})',
        line=dict(color='#2563EB', width=3),
        hovertemplate='False Positive Rate: %{x:.2f}<br>True Positive Rate: %{y:.2f}<extra></extra>'
    ))
    
    # Diagonal baseline
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Baseline',
        line=dict(color='#64748B', dash='dash', width=2),
        hoverinfo='none'
    ))
    
    fig.update_layout(
        title={
            'text': f"ROC Curve (AUC: {roc_auc:.4f})",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 16, 'color': text_color, 'family': "'Plus Jakarta Sans', sans-serif"}
        },
        xaxis=dict(
            title=dict(text="False Positive Rate (FPR)", font=dict(color=text_color, size=12)),
            range=[-0.01, 1.01],
            tickfont=dict(color=text_color, size=10),
            gridcolor=grid_color,
            zeroline=False
        ),
        yaxis=dict(
            title=dict(text="True Positive Rate (TPR)", font=dict(color=text_color, size=12)),
            range=[-0.01, 1.01],
            tickfont=dict(color=text_color, size=10),
            gridcolor=grid_color,
            zeroline=False
        ),
        legend=dict(
            x=0.55, y=0.1, 
            borderwidth=0,
            font=dict(color=text_color, size=10),
            bgcolor='rgba(0,0,0,0)'
        ),
        height=320,
        margin=dict(l=50, r=50, t=50, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def plot_feature_importance(importance_df: pd.DataFrame, max_features: int = 10, is_dark: bool = False) -> go.Figure:
    """
    Creates an interactive Plotly horizontal bar chart of feature importances with theme support.
    """
    df_slice = importance_df.head(max_features).sort_values(by='Importance', ascending=True)
    
    text_color = '#F8FAFC' if is_dark else '#0F172A'
    grid_color = '#334155' if is_dark else '#E2E8F0'
    colorscale = [[0.0, '#6366F1'], [1.0, '#2563EB']]
    
    fig = px.bar(
        df_slice,
        x='Importance',
        y='Feature',
        orientation='h',
        labels={'Importance': 'Importance Weight', 'Feature': 'Features'},
        color='Importance',
        color_continuous_scale=colorscale
    )
    
    fig.update_layout(
        title={
            'text': f"Top {max_features} Feature Contributions",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 16, 'color': text_color, 'family': "'Plus Jakarta Sans', sans-serif"}
        },
        showlegend=False,
        coloraxis_showscale=False,
        xaxis=dict(
            title=dict(text="Importance Weight", font=dict(color=text_color, size=12)),
            tickfont=dict(color=text_color, size=10),
            gridcolor=grid_color,
            zeroline=False
        ),
        yaxis=dict(
            title="",
            tickfont=dict(color=text_color, size=10),
            gridcolor=grid_color,
            zeroline=False
        ),
        height=320,
        margin=dict(l=120, r=40, t=50, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# --- REPORT GENERATION HELPERS ---

def generate_csv_report(input_features: Dict[str, Any], prediction: int, probability: float, confidence: float, risk: str) -> str:
    """
    Generates a CSV report as a string.
    """
    report_dict = input_features.copy()
    report_dict['Predicted Class'] = 'Fully Repaid' if prediction == 0 else 'Not Fully Repaid'
    report_dict['Model Probability'] = f"{probability:.4f}"
    report_dict['Confidence Score (%)'] = f"{confidence:.2f}"
    report_dict['Risk Indicator'] = risk
    
    df_report = pd.DataFrame([report_dict])
    return df_report.to_csv(index=False)

def generate_pdf_report(input_features: Dict[str, Any], prediction: int, probability: float, confidence: float, risk: str) -> bytes:
    """
    Generates a structured PDF report using ReportLab.
    
    Returns:
        bytes: Binary bytes representing the generated PDF report.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54)
    story = []
    
    # Colors matching Fintech / stripe theme
    primary_color = colors.HexColor('#002B49') # Deep Navy
    accent_color = colors.HexColor('#0066CC')  # Tech Blue
    text_color = colors.HexColor('#2D3748')    # Dark Slate
    light_bg = colors.HexColor('#F7FAFC')      # Light grey/blue
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=primary_color,
        spaceAfter=15
    )
    
    section_style = ParagraphStyle(
        'SecTitle',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=accent_color,
        spaceBefore=15,
        spaceAfter=8
    )
    
    text_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=text_color
    )
    
    header_style = ParagraphStyle(
        'HeaderStyle',
        parent=text_style,
        fontName='Helvetica-Bold',
        textColor=colors.white
    )
    
    # Header Section
    story.append(Paragraph("Loan Repayment Prediction Report", title_style))
    story.append(Paragraph("This document contains the applicant parameters and machine learning prediction risk details.", text_style))
    story.append(Spacer(1, 15))
    
    # Section 1: Prediction Results
    story.append(Paragraph("1. Prediction Summary", section_style))
    
    pred_text = "Fully Repaid" if prediction == 0 else "Not Fully Repaid"
    summary_data = [
        [Paragraph("Metric", header_style), Paragraph("Value / Prediction Details", header_style)],
        ["Predicted Status", pred_text],
        ["Model Probability Score", f"{probability:.4f}"],
        ["Prediction Confidence Score", f"{confidence:.2f}%"],
        ["Risk Assessment", risk]
    ]
    
    t_summary = Table(summary_data, colWidths=[200, 300])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, light_bg]),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
    ]))
    story.append(t_summary)
    story.append(Spacer(1, 20))
    
    # Section 2: Input Applicant Parameters
    story.append(Paragraph("2. Submitted Applicant Parameters", section_style))
    
    param_rows = [[Paragraph("Feature Name", header_style), Paragraph("Input Value", header_style)]]
    for key, val in input_features.items():
        # Humanize keys for display
        human_key = key.replace('.', ' ').replace('_', ' ').title()
        param_rows.append([human_key, str(val)])
        
    t_params = Table(param_rows, colWidths=[200, 300])
    t_params.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), accent_color),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, light_bg]),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_params)
    
    # Footer Notice
    story.append(Spacer(1, 30))
    story.append(Paragraph("Disclaimer: This prediction is generated by a machine learning model. Loan approval decisions should incorporate independent credit analysis in compliance with internal policy.", ParagraphStyle('Disclaimer', parent=text_style, fontName='Helvetica-Oblique', fontSize=8, textColor=colors.gray)))
    
    # Build Document
    doc.build(story)
    
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
