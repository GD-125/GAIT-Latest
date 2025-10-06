# File: src/dashboard/results_module.py
# Results visualization and reporting module

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Optional
import logging
from datetime import datetime
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class ResultsModule:
    """
    Comprehensive results visualization and reporting module
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize session state
        if 'export_format' not in st.session_state:
            st.session_state.export_format = 'PDF'
    
    def render(self):
        """Render results interface"""
        st.markdown("""
        <div style="background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); 
                    padding: 2rem; border-radius: 10px; color: white; margin-bottom: 2rem;">
            <h2>📊 Results & Reports Module</h2>
            <p>Comprehensive visualization and export of analysis results</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Check if analysis results exist
        if st.session_state.analysis_results is None:
            st.warning("⚠️ No analysis results available")
            if st.button("🚀 Run Analysis"):
                st.session_state.current_page = "analysis"
                st.rerun()
            return
        
        results = st.session_state.analysis_results
        
        # Results dashboard
        self._display_executive_summary(results)
        
        # Detailed tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Visualizations",
            "📋 Detailed Report", 
            "📊 Statistics",
            "💾 Export",
            "📜 History"
        ])
        
        with tab1:
            self._display_visualizations(results)
        
        with tab2:
            self._display_detailed_report(results)
        
        with tab3:
            self._display_statistics(results)
        
        with tab4:
            self._display_export_options(results)
        
        with tab5:
            self._display_history()
    
    def _display_executive_summary(self, results: Dict):
        """Display executive summary"""
        st.markdown("### 📊 Executive Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            gait_status = "✅" if results['gait_results']['prediction'] == 'Gait' else "❌"
            st.metric(
                "Gait Status",
                f"{gait_status} {results['gait_results']['prediction']}",
                f"{results['gait_results']['confidence']:.1%} confidence"
            )
        
        with col2:
            disease_color = "🟢" if results['disease_results']['prediction'] == 'Healthy' else "🔴"
            st.metric(
                "Classification",
                f"{disease_color} {results['disease_results']['prediction']}",
                f"{results['disease_results']['confidence']:.1%} confidence"
            )
        
        with col3:
            quality_icon = "🟢" if results['quality_report']['overall_score'] > 0.8 else "🟡"
            st.metric(
                "Data Quality",
                f"{quality_icon} {results['quality_report']['overall_score']:.1%}",
                "Excellent" if results['quality_report']['overall_score'] > 0.9 else "Good"
            )
        
        with col4:
            st.metric(
                "Analysis Date",
                datetime.now().strftime("%Y-%m-%d"),
                datetime.now().strftime("%H:%M:%S")
            )
    
    def _display_visualizations(self, results: Dict):
        """Display comprehensive visualizations"""
        st.markdown("### 📈 Analysis Visualizations")
        
        # 1. Confidence Comparison
        st.markdown("#### Prediction Confidence Comparison")
        fig_confidence = self._create_confidence_chart(results)
        st.plotly_chart(fig_confidence, use_container_width=True)
        
        # 2. Disease Probability Distribution
        st.markdown("#### Disease Probability Distribution")
        fig_disease = self._create_disease_probability_chart(results)
        st.plotly_chart(fig_disease, use_container_width=True)
        
        # 3. Feature Importance (if available)
        if results.get('explanations'):
            st.markdown("#### Feature Importance")
            fig_importance = self._create_feature_importance_chart(results['explanations'])
            st.plotly_chart(fig_importance, use_container_width=True)
        
        # 4. Data Quality Breakdown
        st.markdown("#### Data Quality Metrics")
        fig_quality = self._create_quality_chart(results['quality_report'])
        st.plotly_chart(fig_quality, use_container_width=True)
    
    def _create_confidence_chart(self, results: Dict) -> go.Figure:
        """Create confidence comparison chart"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Gait Detection', 'Disease Classification'),
            specs=[[{"type": "indicator"}, {"type": "indicator"}]]
        )
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=results['gait_results']['confidence'] * 100,
                title={'text': results['gait_results']['prediction']},
                delta={'reference': 85, 'increasing': {'color': "green"}},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#3498db"},
                    'steps': [
                        {'range': [0, 70], 'color': "#e74c3c"},
                        {'range': [70, 85], 'color': "#f39c12"},
                        {'range': [85, 100], 'color': "#2ecc71"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 85
                    }
                }
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=results['disease_results']['confidence'] * 100,
                title={'text': results['disease_results']['prediction']},
                delta={'reference': 85, 'increasing': {'color': "green"}},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#9b59b6"},
                    'steps': [
                        {'range': [0, 70], 'color': "#e74c3c"},
                        {'range': [70, 85], 'color': "#f39c12"},
                        {'range': [85, 100], 'color': "#2ecc71"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 85
                    }
                }
            ),
            row=1, col=2
        )
        
        fig.update_layout(height=400, template='plotly_white')
        return fig
    
    def _create_disease_probability_chart(self, results: Dict) -> go.Figure:
        """Create disease probability distribution chart"""
        probs = results['disease_results']['probabilities']
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(probs.keys()),
                y=list(probs.values()),
                marker=dict(
                    color=list(probs.values()),
                    colorscale='RdYlGn_r',
                    showscale=True,
                    colorbar=dict(title="Probability")
                ),
                text=[f"{v:.1%}" for v in probs.values()],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            xaxis_title="Disease Category",
            yaxis_title="Probability",
            height=400,
            template='plotly_white'
        )
        
        return fig
    
    def _create_feature_importance_chart(self, explanations: Dict) -> go.Figure:
        """Create feature importance chart"""
        features = explanations['top_features']
        
        df = pd.DataFrame(features)
        
        fig = go.Figure(data=[
            go.Bar(
                x=df['importance'],
                y=df['feature'],
                orientation='h',
                marker=dict(
                    color=df['importance'],
                    colorscale='Viridis',
                    showscale=True
                )
            )
        ])
        
        fig.update_layout(
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            height=400,
            template='plotly_white'
        )
        
        return fig
    
    def _create_quality_chart(self, quality_report: Dict) -> go.Figure:
        """Create data quality breakdown chart"""
        metrics = quality_report['metrics']
        
        fig = go.Figure(data=[
            go.Scatterpolar(
                r=list(metrics.values()),
                theta=list(metrics.keys()),
                fill='toself',
                marker=dict(color='#3498db')
            )
        ])
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=False,
            height=500,
            template='plotly_white'
        )
        
        return fig
    
    def _display_detailed_report(self, results: Dict):
        """Display detailed textual report"""
        st.markdown("### 📋 Comprehensive Analysis Report")
        
        # Generate report sections
        report = self._generate_report_text(results)
        
        st.markdown(report)
        
        # Download report
        if st.button("📥 Download Report as Text"):
            st.download_button(
                "Download",
                data=report,
                file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    def _generate_report_text(self, results: Dict) -> str:
        """Generate comprehensive text report"""
        report = f"""
# GAIT ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## EXECUTIVE SUMMARY
{'-' * 80}

Gait Detection: {results['gait_results']['prediction']}
Confidence: {results['gait_results']['confidence']:.2%}

Disease Classification: {results['disease_results']['prediction']}
Confidence: {results['disease_results']['confidence']:.2%}

Data Quality Score: {results['quality_report']['overall_score']:.2%}

## DETAILED FINDINGS
{'-' * 80}

### 1. Gait Detection Analysis
The system detected {'gait patterns' if results['gait_results']['prediction'] == 'Gait' else 'no gait patterns'} 
in the provided data with {results['gait_results']['confidence']:.1%} confidence.

Movement Intensity: {results['gait_results'].get('movement_intensity', 'N/A')}

### 2. Disease Classification
Primary Classification: {results['disease_results']['prediction']}
Confidence Level: {results['disease_results']['confidence']:.2%}

Probability Distribution:
"""
        
        for disease, prob in results['disease_results']['probabilities'].items():
            report += f"  - {disease}: {prob:.2%}\n"
        
        report += f"""
### 3. Data Quality Assessment
Overall Quality Score: {results['quality_report']['overall_score']:.2%}

Quality Metrics:
"""
        
        for metric, value in results['quality_report']['metrics'].items():
            report += f"  - {metric.title()}: {value:.2%}\n"
        
        if results.get('explanations'):
            report += f"""
### 4. Explainability Analysis
{results['explanations']['summary']}

Top Contributing Features:
"""
            for feature in results['explanations']['top_features'][:5]:
                report += f"  - {feature['feature']}: {feature['importance']:.4f}\n"
        
        if results.get('recommendations'):
            report += f"""
### 5. Recommendations
"""
            for i, rec in enumerate(results['recommendations'], 1):
                report += f"{i}. {rec}\n"
        
        report += f"""
{'-' * 80}
End of Report
"""
        
        return report
    
    def _display_statistics(self, results: Dict):
        """Display statistical analysis"""
        st.markdown("### 📊 Statistical Analysis")
        
        # Create statistical summary
        stats_data = {
            'Metric': [
                'Gait Confidence',
                'Disease Confidence',
                'Data Quality',
                'Completeness',
                'Consistency',
                'Accuracy',
                'Validity'
            ],
            'Value': [
                f"{results['gait_results']['confidence']:.2%}",
                f"{results['disease_results']['confidence']:.2%}",
                f"{results['quality_report']['overall_score']:.2%}",
                f"{results['quality_report']['metrics']['completeness']:.2%}",
                f"{results['quality_report']['metrics']['consistency']:.2%}",
                f"{results['quality_report']['metrics']['accuracy']:.2%}",
                f"{results['quality_report']['metrics']['validity']:.2%}"
            ],
            'Status': [
                '✅' if results['gait_results']['confidence'] > 0.85 else '⚠️',
                '✅' if results['disease_results']['confidence'] > 0.85 else '⚠️',
                '✅' if results['quality_report']['overall_score'] > 0.80 else '⚠️',
                '✅' if results['quality_report']['metrics']['completeness'] > 0.90 else '⚠️',
                '✅' if results['quality_report']['metrics']['consistency'] > 0.80 else '⚠️',
                '✅' if results['quality_report']['metrics']['accuracy'] > 0.80 else '⚠️',
                '✅' if results['quality_report']['metrics']['validity'] > 0.90 else '⚠️'
            ]
        }
        
        df_stats = pd.DataFrame(stats_data)
        st.dataframe(df_stats, use_container_width=True)
        
        # Preprocessing steps
        if 'preprocessing' in results:
            st.markdown("#### Preprocessing Steps Applied")
            for step in results['preprocessing']['steps']:
                st.info(f"✓ {step}")
    
    def _display_export_options(self, results: Dict):
        """Display export options"""
        st.markdown("### 💾 Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Export Format")
            export_format = st.selectbox(
                "Choose format",
                ["PDF", "JSON", "CSV", "Excel", "HTML"]
            )
            
            include_visualizations = st.checkbox("Include Visualizations", value=True)
            include_explanations = st.checkbox("Include Explanations", value=True)
            include_recommendations = st.checkbox("Include Recommendations", value=True)
        
        with col2:
            st.markdown("#### Export Actions")
            
            if st.button("📥 Generate Export", type="primary", use_container_width=True):
                export_data = self._generate_export(
                    results,
                    export_format,
                    include_visualizations,
                    include_explanations,
                    include_recommendations
                )
                
                st.success("✅ Export generated successfully!")
                
                # Download button
                st.download_button(
                    label=f"Download {export_format}",
                    data=export_data,
                    file_name=f"gait_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format.lower()}",
                    mime=self._get_mime_type(export_format)
                )
            
            if st.button("📧 Email Report", use_container_width=True):
                st.info("Email functionality would be implemented here")
            
            if st.button("☁️ Save to Cloud", use_container_width=True):
                st.info("Cloud storage functionality would be implemented here")
    
    def _generate_export(
        self,
        results: Dict,
        format: str,
        include_viz: bool,
        include_exp: bool,
        include_rec: bool
    ) -> bytes:
        """Generate export data in specified format"""
        
        if format == "JSON":
            export_dict = {
                'timestamp': datetime.now().isoformat(),
                'gait_results': results['gait_results'],
                'disease_results': results['disease_results'],
                'quality_report': results['quality_report']
            }
            
            if include_exp and results.get('explanations'):
                export_dict['explanations'] = results['explanations']
            
            if include_rec and results.get('recommendations'):
                export_dict['recommendations'] = results['recommendations']
            
            return json.dumps(export_dict, indent=2).encode('utf-8')
        
        elif format == "CSV":
            # Create CSV with key metrics
            data = {
                'Metric': ['Gait Prediction', 'Gait Confidence', 'Disease Prediction', 'Disease Confidence', 'Data Quality'],
                'Value': [
                    results['gait_results']['prediction'],
                    results['gait_results']['confidence'],
                    results['disease_results']['prediction'],
                    results['disease_results']['confidence'],
                    results['quality_report']['overall_score']
                ]
            }
            df = pd.DataFrame(data)
            return df.to_csv(index=False).encode('utf-8')
        
        else:
            # For other formats, return text report
            return self._generate_report_text(results).encode('utf-8')
    
    def _get_mime_type(self, format: str) -> str:
        """Get MIME type for format"""
        mime_types = {
            'PDF': 'application/pdf',
            'JSON': 'application/json',
            'CSV': 'text/csv',
            'Excel': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'HTML': 'text/html'
        }
        return mime_types.get(format, 'text/plain')
    
    def _display_history(self):
        """Display analysis history"""
        st.markdown("### 📜 Analysis History")
        
        # Get history from session state
        if 'analysis_history' not in st.session_state:
            st.session_state.analysis_history = []
        
        if not st.session_state.analysis_history:
            st.info("No previous analyses found")
            return
        
        # Display history as table
        history_df = pd.DataFrame(st.session_state.analysis_history)
        st.dataframe(history_df, use_container_width=True)
        
        # Export history
        if st.button("📥 Export History"):
            csv = history_df.to_csv(index=False)
            st.download_button(
                "Download History CSV",
                data=csv,
                file_name=f"analysis_history_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    st.set_page_config(page_title="Results Module", layout="wide")
    
    module = ResultsModule()
    module.render()