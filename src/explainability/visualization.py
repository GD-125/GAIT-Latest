# File: src/explainability/visualization.py
# Visualization tools for explainability results

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ExplainabilityVisualizer:
    """
    Comprehensive visualization tools for model explainability
    Creates interactive and static visualizations for SHAP and LIME
    """
    
    def __init__(self, style: str = 'seaborn'):
        """
        Initialize visualizer
        
        Args:
            style: Matplotlib style to use
        """
        plt.style.use(style)
        self.logger = logging.getLogger(__name__)
        
        # Color schemes
        self.colors = {
            'positive': '#2ecc71',
            'negative': '#e74c3c',
            'neutral': '#95a5a6',
            'primary': '#3498db',
            'secondary': '#9b59b6'
        }
    
    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 20,
        title: str = "Feature Importance",
        save_path: Optional[str] = None
    ):
        """
        Plot feature importance bar chart
        
        Args:
            importance_df: DataFrame with 'Feature' and 'Importance' columns
            top_n: Number of top features to show
            title: Plot title
            save_path: Path to save figure
        """
        # Get top features
        top_features = importance_df.head(top_n)
        
        # Create figure
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=top_features['Feature'],
            x=top_features['Importance'],
            orientation='h',
            marker=dict(
                color=top_features['Importance'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Importance")
            ),
            text=top_features['Importance'].round(4),
            textposition='auto'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            height=max(400, top_n * 25),
            template='plotly_white',
            font=dict(size=12)
        )
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"Feature importance plot saved to {save_path}")
        
        return fig
    
    def plot_shap_waterfall(
        self,
        explanation: Dict,
        max_display: int = 10,
        save_path: Optional[str] = None
    ):
        """
        Create SHAP waterfall plot showing contribution flow
        
        Args:
            explanation: SHAP explanation dictionary
            max_display: Maximum features to display
            save_path: Path to save figure
        """
        contributions = explanation['top_contributions'][:max_display]
        
        # Prepare data
        features = [c.get('sensor', f"Feature {c.get('feature_idx', '')}") for c in contributions]
        values = [c['shap_value'] for c in contributions]
        
        # Calculate cumulative sum
        base_value = explanation.get('base_value', 0)
        cumsum = np.cumsum([base_value] + values)
        
        # Create waterfall chart
        fig = go.Figure()
        
        # Add bars
        colors = [self.colors['positive'] if v > 0 else self.colors['negative'] for v in values]
        
        fig.add_trace(go.Waterfall(
            name="SHAP",
            orientation="v",
            measure=["relative"] * len(values) + ["total"],
            x=features + ["Predicted Value"],
            y=values + [cumsum[-1]],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": self.colors['positive']}},
            decreasing={"marker": {"color": self.colors['negative']}},
            totals={"marker": {"color": self.colors['primary']}}
        ))
        
        fig.update_layout(
            title=f"SHAP Waterfall Plot - {explanation['prediction']} ({explanation['confidence']*100:.1f}%)",
            yaxis_title="SHAP Value",
            xaxis_title="Features",
            height=500,
            template='plotly_white',
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"SHAP waterfall plot saved to {save_path}")
        
        return fig
    
    def plot_lime_explanation(
        self,
        explanation: Dict,
        max_display: int = 10,
        save_path: Optional[str] = None
    ):
        """
        Visualize LIME explanation with horizontal bar chart
        
        Args:
            explanation: LIME explanation dictionary
            max_display: Maximum features to display
            save_path: Path to save figure
        """
        contributions = explanation['top_contributions'][:max_display]
        
        # Prepare data
        features = [c['feature'][:50] for c in contributions]  # Truncate long names
        weights = [c['weight'] for c in contributions]
        colors = [self.colors['positive'] if w > 0 else self.colors['negative'] for w in weights]
        
        # Create figure
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=features,
            x=weights,
            orientation='h',
            marker=dict(color=colors),
            text=[f"{w:.4f}" for w in weights],
            textposition='auto'
        ))
        
        fig.update_layout(
            title=f"LIME Explanation - {explanation['prediction']} ({explanation['confidence']*100:.1f}%)",
            xaxis_title="Feature Weight",
            yaxis_title="Feature",
            height=max(400, max_display * 40),
            template='plotly_white',
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"LIME explanation plot saved to {save_path}")
        
        return fig
    
    def plot_attention_heatmap(
        self,
        attention_weights: np.ndarray,
        sensor_names: Optional[List[str]] = None,
        title: str = "Attention Heatmap",
        save_path: Optional[str] = None
    ):
        """
        Visualize attention weights as heatmap
        
        Args:
            attention_weights: Attention weights array
            sensor_names: Names of sensors/features
            title: Plot title
            save_path: Path to save figure
        """
        # Ensure 2D
        if len(attention_weights.shape) == 1:
            attention_weights = attention_weights.reshape(1, -1)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=attention_weights,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Attention")
        ))
        
        # Add labels if provided
        if sensor_names:
            fig.update_yaxes(ticktext=sensor_names, tickvals=list(range(len(sensor_names))))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time Step",
            yaxis_title="Feature/Sensor",
            height=400,
            template='plotly_white'
        )
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"Attention heatmap saved to {save_path}")
        
        return fig
    
    def plot_comparison_dashboard(
        self,
        shap_explanation: Dict,
        lime_explanation: Dict,
        save_path: Optional[str] = None
    ):
        """
        Create comprehensive comparison dashboard for SHAP and LIME
        
        Args:
            shap_explanation: SHAP explanation dictionary
            lime_explanation: LIME explanation dictionary
            save_path: Path to save figure
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'SHAP Feature Contributions',
                'LIME Feature Weights',
                'Prediction Confidence Comparison',
                'Top Features Overlap'
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "indicator"}, {"type": "bar"}]
            ]
        )
        
        # SHAP contributions
        shap_contribs = shap_explanation['top_contributions'][:8]
        fig.add_trace(
            go.Bar(
                y=[f"F{i}" for i in range(len(shap_contribs))],
                x=[c['shap_value'] for c in shap_contribs],
                orientation='h',
                name='SHAP',
                marker_color=self.colors['primary']
            ),
            row=1, col=1
        )
        
        # LIME weights
        lime_contribs = lime_explanation['top_contributions'][:8]
        fig.add_trace(
            go.Bar(
                y=[c['feature'][:20] for c in lime_contribs],
                x=[c['weight'] for c in lime_contribs],
                orientation='h',
                name='LIME',
                marker_color=self.colors['secondary']
            ),
            row=1, col=2
        )
        
        # Confidence comparison
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=shap_explanation['confidence'] * 100,
                title={'text': "SHAP Confidence"},
                delta={'reference': lime_explanation['confidence'] * 100},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': self.colors['primary']}}
            ),
            row=2, col=1
        )
        
        # Feature overlap
        shap_features = set([str(c)[:15] for c in shap_contribs])
        lime_features = set([c['feature'][:15] for c in lime_contribs])
        overlap = len(shap_features.intersection(lime_features))
        
        fig.add_trace(
            go.Bar(
                x=['SHAP Only', 'Overlap', 'LIME Only'],
                y=[len(shap_features) - overlap, overlap, len(lime_features) - overlap],
                marker_color=[self.colors['primary'], self.colors['positive'], self.colors['secondary']]
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text=f"Explainability Comparison: {shap_explanation['prediction']}",
            template='plotly_white'
        )
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"Comparison dashboard saved to {save_path}")
        
        return fig
    
    def plot_sensor_contributions(
        self,
        sensor_data: Dict[str, float],
        title: str = "Sensor Contributions",
        save_path: Optional[str] = None
    ):
        """
        Visualize sensor-wise contributions as pie/donut chart
        
        Args:
            sensor_data: Dictionary of sensor names to contribution values
            title: Plot title
            save_path: Path to save figure
        """
        # Sort by contribution
        sorted_sensors = sorted(sensor_data.items(), key=lambda x: abs(x[1]), reverse=True)
        sensors = [s[0] for s in sorted_sensors]
        contributions = [abs(s[1]) for s in sorted_sensors]
        
        # Create donut chart
        fig = go.Figure(data=[go.Pie(
            labels=sensors,
            values=contributions,
            hole=.4,
            marker=dict(colors=px.colors.qualitative.Set3)
        )])
        
        fig.update_layout(
            title=title,
            height=500,
            template='plotly_white',
            annotations=[dict(text='Sensors', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"Sensor contributions plot saved to {save_path}")
        
        return fig
    
    def plot_temporal_importance(
        self,
        time_series: np.ndarray,
        importance_scores: np.ndarray,
        sensor_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot time series data with importance overlay
        
        Args:
            time_series: Time series data (channels, timesteps)
            importance_scores: Importance score per timestep
            sensor_names: Names of sensors
            save_path: Path to save figure
        """
        n_channels, n_timesteps = time_series.shape
        
        # Create figure with subplots for each channel
        fig = make_subplots(
            rows=n_channels, cols=1,
            shared_xaxes=True,
            subplot_titles=sensor_names if sensor_names else [f"Channel {i}" for i in range(n_channels)],
            vertical_spacing=0.05
        )
        
        # Normalize importance for color mapping
        importance_norm = (importance_scores - importance_scores.min()) / (importance_scores.max() - importance_scores.min() + 1e-10)
        
        # Plot each channel
        for i in range(n_channels):
            # Add time series line
            fig.add_trace(
                go.Scatter(
                    x=list(range(n_timesteps)),
                    y=time_series[i],
                    mode='lines',
                    name=sensor_names[i] if sensor_names else f"Channel {i}",
                    line=dict(color=self.colors['primary'], width=2)
                ),
                row=i+1, col=1
            )
            
            # Add importance as background shading
            fig.add_trace(
                go.Scatter(
                    x=list(range(n_timesteps)),
                    y=time_series[i],
                    fill='tonexty',
                    mode='none',
                    fillcolor=f'rgba(255, 0, 0, {importance_norm.mean():.2f})',
                    showlegend=False
                ),
                row=i+1, col=1
            )
        
        fig.update_layout(
            height=200 * n_channels,
            title_text="Temporal Feature Importance",
            template='plotly_white',
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Time Step", row=n_channels, col=1)
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"Temporal importance plot saved to {save_path}")
        
        return fig
    
    def create_explainability_report(
        self,
        shap_explanation: Dict,
        lime_explanation: Dict,
        output_dir: str = "reports/explainability"
    ):
        """
        Generate comprehensive explainability report with all visualizations
        
        Args:
            shap_explanation: SHAP explanation dictionary
            lime_explanation: LIME explanation dictionary
            output_dir: Directory to save report files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate all plots
        plots = {}
        
        # 1. SHAP waterfall
        plots['shap_waterfall'] = self.plot_shap_waterfall(
            shap_explanation,
            save_path=str(output_path / "shap_waterfall.html")
        )
        
        # 2. LIME explanation
        plots['lime_explanation'] = self.plot_lime_explanation(
            lime_explanation,
            save_path=str(output_path / "lime_explanation.html")
        )
        
        # 3. Comparison dashboard
        plots['comparison'] = self.plot_comparison_dashboard(
            shap_explanation,
            lime_explanation,
            save_path=str(output_path / "comparison_dashboard.html")
        )
        
        # 4. Sensor contributions (if available)
        if 'sensor_contributions' in lime_explanation:
            plots['sensor_contrib'] = self.plot_sensor_contributions(
                lime_explanation['sensor_contributions'],
                save_path=str(output_path / "sensor_contributions.html")
            )
        
        # Create HTML report
        html_content = self._generate_html_report(shap_explanation, lime_explanation, output_path)
        
        report_path = output_path / "explainability_report.html"
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"Comprehensive explainability report generated at {report_path}")
        
        return str(report_path)
    
    def _generate_html_report(
        self,
        shap_explanation: Dict,
        lime_explanation: Dict,
        output_path: Path
    ) -> str:
        """Generate HTML report content"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Explainability Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; margin-top: 30px; }}
                .summary {{ background: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }}
                .metric {{ display: inline-block; margin: 10px 20px; }}
                .metric-label {{ font-weight: bold; color: #7f8c8d; }}
                .metric-value {{ font-size: 24px; color: #2c3e50; }}
                iframe {{ width: 100%; height: 600px; border: none; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🧠 Explainability Analysis Report</h1>
                
                <div class="summary">
                    <h2>Prediction Summary</h2>
                    <div class="metric">
                        <div class="metric-label">SHAP Prediction:</div>
                        <div class="metric-value">{shap_explanation['prediction']}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">LIME Prediction:</div>
                        <div class="metric-value">{lime_explanation['prediction']}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">SHAP Confidence:</div>
                        <div class="metric-value">{shap_explanation['confidence']*100:.1f}%</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">LIME Confidence:</div>
                        <div class="metric-value">{lime_explanation['confidence']*100:.1f}%</div>
                    </div>
                </div>
                
                <h2>📊 SHAP Analysis</h2>
                <p>{shap_explanation.get('summary', 'No summary available')}</p>
                <iframe src="shap_waterfall.html"></iframe>
                
                <h2>📈 LIME Analysis</h2>
                <p>{lime_explanation.get('summary', 'No summary available')}</p>
                <iframe src="lime_explanation.html"></iframe>
                
                <h2>🔄 Comparison Dashboard</h2>
                <iframe src="comparison_dashboard.html"></iframe>
            </div>
        </body>
        </html>
        """
        return html


if __name__ == "__main__":
    print("Testing Explainability Visualizer...")
    
    # Create dummy explanations
    shap_exp = {
        'prediction': 'Gait',
        'confidence': 0.92,
        'base_value': 0.5,
        'top_contributions': [
            {'sensor': 'acc_x', 'shap_value': 0.15},
            {'sensor': 'gyro_z', 'shap_value': -0.10},
            {'sensor': 'acc_y', 'shap_value': 0.08}
        ]
    }
    
    lime_exp = {
        'prediction': 'Gait',
        'confidence': 0.89,
        'top_contributions': [
            {'feature': 'acc_x_mean', 'weight': 0.25, 'contribution': 'positive'},
            {'feature': 'gyro_z_std', 'weight': -0.18, 'contribution': 'negative'}
        ]
    }
    
    # Create visualizer
    viz = ExplainabilityVisualizer()
    
    # Test plots
    print("✅ Creating feature importance plot...")
    importance_df = pd.DataFrame({
        'Feature': ['acc_x', 'acc_y', 'acc_z', 'gyro_x'],
        'Importance': [0.25, 0.20, 0.18, 0.15]
    })
    viz.plot_feature_importance(importance_df)
    
    print("✅ Creating SHAP waterfall...")
    viz.plot_shap_waterfall(shap_exp)
    
    print("✅ Creating LIME explanation...")
    viz.plot_lime_explanation(lime_exp)
    
    print("\n✅ Explainability Visualizer tests passed!")