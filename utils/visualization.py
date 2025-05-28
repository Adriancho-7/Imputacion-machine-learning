import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class ImputationVisualizer:
    """
    Visualization utilities for imputation analysis and results.
    """
    
    def __init__(self):
        # Set color palettes
        self.color_palette = px.colors.qualitative.Set3
        self.sequential_palette = px.colors.sequential.Viridis
        
    def create_missing_data_heatmap(self, data: pd.DataFrame, title: str = "Missing Data Pattern") -> go.Figure:
        """
        Create a heatmap showing missing data patterns.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input dataframe
        title : str
            Title for the plot
            
        Returns:
        --------
        Plotly figure object
        """
        # Create missing data matrix (1 for missing, 0 for present)
        missing_matrix = data.isnull().astype(int)
        
        # Calculate missing percentages for sorting
        missing_percentages = missing_matrix.sum() / len(data) * 100
        sorted_columns = missing_percentages.sort_values(ascending=False).index.tolist()
        
        # Reorder columns by missing percentage
        missing_matrix_sorted = missing_matrix[sorted_columns]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=missing_matrix_sorted.T.values,
            x=missing_matrix_sorted.index,
            y=sorted_columns,
            colorscale=[[0, 'lightblue'], [1, 'red']],
            showscale=True,
            colorbar=dict(
                title="Missing",
                tickvals=[0, 1],
                ticktext=['Present', 'Missing']
            ),
            hovertemplate='Row: %{x}<br>Column: %{y}<br>Status: %{customdata}<extra></extra>',
            customdata=np.where(missing_matrix_sorted.T.values == 1, 'Missing', 'Present')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Data Points",
            yaxis_title="Columns",
            height=max(400, len(sorted_columns) * 20),
            template="plotly_white"
        )
        
        return fig
    
    def create_missing_data_summary(self, data: pd.DataFrame) -> go.Figure:
        """
        Create a summary plot of missing data by column.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        Plotly figure object
        """
        missing_counts = data.isnull().sum()
        missing_percentages = (missing_counts / len(data)) * 100
        
        # Filter only columns with missing data
        missing_data = missing_percentages[missing_percentages > 0].sort_values(ascending=True)
        
        if len(missing_data) == 0:
            # No missing data
            fig = go.Figure()
            fig.add_annotation(
                text="No Missing Data Found",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=20, color="green")
            )
            fig.update_layout(
                title="Missing Data Summary",
                template="plotly_white",
                height=400
            )
            return fig
        
        # Create horizontal bar chart
        fig = go.Figure(data=[
            go.Bar(
                y=missing_data.index,
                x=missing_data.values,
                orientation='h',
                marker=dict(
                    color=missing_data.values,
                    colorscale='Reds',
                    colorbar=dict(title="Missing %")
                ),
                text=[f"{val:.1f}%" for val in missing_data.values],
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title="Missing Data Percentage by Column",
            xaxis_title="Missing Data Percentage",
            yaxis_title="Columns",
            template="plotly_white",
            height=max(400, len(missing_data) * 30)
        )
        
        return fig
    
    def create_distribution_comparison(self, original_data: pd.DataFrame, 
                                     imputed_data: pd.DataFrame, 
                                     column: str) -> go.Figure:
        """
        Create distribution comparison between original and imputed data.
        
        Parameters:
        -----------
        original_data : pd.DataFrame
            Original dataframe
        imputed_data : pd.DataFrame
            Imputed dataframe
        column : str
            Column to compare
            
        Returns:
        --------
        Plotly figure object
        """
        # Get original non-missing values
        original_values = original_data[column].dropna()
        
        # Get imputed values (only the ones that were missing)
        missing_mask = original_data[column].isnull()
        imputed_values = imputed_data.loc[missing_mask, column]
        
        # Get all values from imputed dataset
        all_imputed_values = imputed_data[column].dropna()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'Original {column} Distribution',
                f'Imputed Values Distribution',
                f'Combined Distribution Comparison',
                f'Box Plot Comparison'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Original distribution
        fig.add_trace(
            go.Histogram(
                x=original_values,
                name='Original',
                opacity=0.7,
                marker_color='blue',
                nbinsx=30
            ),
            row=1, col=1
        )
        
        # Imputed values distribution
        fig.add_trace(
            go.Histogram(
                x=imputed_values,
                name='Imputed Values',
                opacity=0.7,
                marker_color='red',
                nbinsx=30
            ),
            row=1, col=2
        )
        
        # Combined comparison
        fig.add_trace(
            go.Histogram(
                x=original_values,
                name='Original (Complete)',
                opacity=0.6,
                marker_color='blue',
                nbinsx=30
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Histogram(
                x=all_imputed_values,
                name='After Imputation',
                opacity=0.6,
                marker_color='green',
                nbinsx=30
            ),
            row=2, col=1
        )
        
        # Box plots
        fig.add_trace(
            go.Box(
                y=original_values,
                name='Original',
                marker_color='blue'
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Box(
                y=all_imputed_values,
                name='After Imputation',
                marker_color='green'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f"Distribution Analysis: {column}",
            template="plotly_white",
            height=800,
            showlegend=True
        )
        
        return fig
    
    def create_correlation_comparison(self, original_data: pd.DataFrame,
                                    imputed_data: pd.DataFrame,
                                    column1: str, column2: str) -> go.Figure:
        """
        Create scatter plot comparison showing correlations before and after imputation.
        
        Parameters:
        -----------
        original_data : pd.DataFrame
            Original dataframe
        imputed_data : pd.DataFrame
            Imputed dataframe
        column1 : str
            First column for correlation
        column2 : str
            Second column for correlation
            
        Returns:
        --------
        Plotly figure object
        """
        # Get complete cases from original data
        complete_mask = original_data[[column1, column2]].notna().all(axis=1)
        original_complete = original_data[complete_mask]
        
        # Get data after imputation
        imputed_complete = imputed_data[[column1, column2]].dropna()
        
        # Calculate correlations
        orig_corr = original_complete[column1].corr(original_complete[column2])
        imp_corr = imputed_complete[column1].corr(imputed_complete[column2])
        
        # Create subplot
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                f'Original Data (r = {orig_corr:.3f})',
                f'After Imputation (r = {imp_corr:.3f})'
            )
        )
        
        # Original data scatter
        fig.add_trace(
            go.Scatter(
                x=original_complete[column1],
                y=original_complete[column2],
                mode='markers',
                name='Original Data',
                marker=dict(color='blue', opacity=0.6),
                hovertemplate=f'{column1}: %{{x}}<br>{column2}: %{{y}}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Imputed data scatter
        fig.add_trace(
            go.Scatter(
                x=imputed_complete[column1],
                y=imputed_complete[column2],
                mode='markers',
                name='After Imputation',
                marker=dict(color='green', opacity=0.6),
                hovertemplate=f'{column1}: %{{x}}<br>{column2}: %{{y}}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Add trend lines
        from scipy import stats
        
        # Original trend line
        if len(original_complete) > 1:
            slope, intercept, _, _, _ = stats.linregress(original_complete[column1], 
                                                       original_complete[column2])
            line_x = np.array([original_complete[column1].min(), original_complete[column1].max()])
            line_y = slope * line_x + intercept
            
            fig.add_trace(
                go.Scatter(
                    x=line_x,
                    y=line_y,
                    mode='lines',
                    name='Original Trend',
                    line=dict(color='blue', dash='dash'),
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # Imputed trend line
        if len(imputed_complete) > 1:
            slope, intercept, _, _, _ = stats.linregress(imputed_complete[column1], 
                                                       imputed_complete[column2])
            line_x = np.array([imputed_complete[column1].min(), imputed_complete[column1].max()])
            line_y = slope * line_x + intercept
            
            fig.add_trace(
                go.Scatter(
                    x=line_x,
                    y=line_y,
                    mode='lines',
                    name='Imputed Trend',
                    line=dict(color='green', dash='dash'),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title=f"Correlation Analysis: {column1} vs {column2}",
            template="plotly_white",
            height=500
        )
        
        fig.update_xaxes(title_text=column1)
        fig.update_yaxes(title_text=column2)
        
        return fig
    
    def create_imputation_quality_metrics(self, report: Dict) -> go.Figure:
        """
        Create visualization of imputation quality metrics.
        
        Parameters:
        -----------
        report : Dict
            Imputation report from SVRImputer
            
        Returns:
        --------
        Plotly figure object
        """
        if 'model_scores' not in report or not report['model_scores']:
            fig = go.Figure()
            fig.add_annotation(
                text="No Model Scores Available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        columns = list(report['model_scores'].keys())
        scores = [report['model_scores'][col] for col in columns if report['model_scores'][col] is not None]
        valid_columns = [col for col in columns if report['model_scores'][col] is not None]
        
        if not scores:
            fig = go.Figure()
            fig.add_annotation(
                text="No Valid Model Scores",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Create color mapping based on score quality
        colors = []
        for score in scores:
            if score > 0.8:
                colors.append('green')
            elif score > 0.6:
                colors.append('orange')
            elif score > 0.4:
                colors.append('yellow')
            else:
                colors.append('red')
        
        fig = go.Figure(data=[
            go.Bar(
                x=valid_columns,
                y=scores,
                marker=dict(color=colors),
                text=[f"{score:.3f}" for score in scores],
                textposition='outside'
            )
        ])
        
        # Add performance threshold lines
        fig.add_hline(y=0.8, line_dash="dash", line_color="green", 
                      annotation_text="Excellent (>0.8)")
        fig.add_hline(y=0.6, line_dash="dash", line_color="orange", 
                      annotation_text="Good (>0.6)")
        fig.add_hline(y=0.4, line_dash="dash", line_color="yellow", 
                      annotation_text="Fair (>0.4)")
        
        fig.update_layout(
            title="SVR Model Performance (R² Scores)",
            xaxis_title="Imputed Columns",
            yaxis_title="R² Score",
            template="plotly_white",
            height=500
        )
        
        return fig
    
    def create_before_after_summary(self, original_data: pd.DataFrame, 
                                  imputed_data: pd.DataFrame) -> go.Figure:
        """
        Create summary comparison of dataset before and after imputation.
        
        Parameters:
        -----------
        original_data : pd.DataFrame
            Original dataframe
        imputed_data : pd.DataFrame
            Imputed dataframe
            
        Returns:
        --------
        Plotly figure object
        """
        # Calculate statistics
        original_missing = original_data.isnull().sum().sum()
        imputed_missing = imputed_data.isnull().sum().sum()
        total_cells = original_data.shape[0] * original_data.shape[1]
        
        original_complete_percentage = ((total_cells - original_missing) / total_cells) * 100
        imputed_complete_percentage = ((total_cells - imputed_missing) / total_cells) * 100
        
        # Create comparison chart
        categories = ['Data Completeness', 'Missing Values']
        original_values = [original_complete_percentage, (original_missing / total_cells) * 100]
        imputed_values = [imputed_complete_percentage, (imputed_missing / total_cells) * 100]
        
        fig = go.Figure(data=[
            go.Bar(name='Before Imputation', x=categories, y=original_values, 
                   marker_color='lightcoral'),
            go.Bar(name='After Imputation', x=categories, y=imputed_values, 
                   marker_color='lightgreen')
        ])
        
        fig.update_layout(
            title="Dataset Quality: Before vs After Imputation",
            xaxis_title="Metrics",
            yaxis_title="Percentage",
            barmode='group',
            template="plotly_white",
            height=400
        )
        
        return fig
    
    def create_imputation_impact_summary(self, report: Dict) -> go.Figure:
        """
        Create summary of imputation impact across columns.
        
        Parameters:
        -----------
        report : Dict
            Imputation report from SVRImputer
            
        Returns:
        --------
        Plotly figure object
        """
        if not report.get('imputed_counts'):
            fig = go.Figure()
            fig.add_annotation(
                text="No Imputation Data Available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        columns = list(report['imputed_counts'].keys())
        imputed_counts = list(report['imputed_counts'].values())
        
        # Create pie chart showing distribution of imputed values
        fig = go.Figure(data=[go.Pie(
            labels=columns,
            values=imputed_counts,
            hole=0.3,
            textinfo='label+percent+value',
            texttemplate='%{label}<br>%{value} values<br>(%{percent})'
        )])
        
        fig.update_layout(
            title="Distribution of Imputed Values by Column",
            template="plotly_white",
            height=500
        )
        
        return fig
