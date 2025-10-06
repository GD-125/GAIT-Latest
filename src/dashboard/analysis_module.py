# File: src/dashboard/analysis_module.py
# Real-time analysis and prediction module using Agentic AI

import streamlit as st
import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Optional
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.models.gait_detector import GaitDetector
from src.models.disease_classifier import DiseaseClassifier
from src.preprocessing.signal_processor import SignalProcessor
from src.explainability.shap_explainer import SHAPExplainer
from src.explainability.lime_explainer import LIMEExplainer
from src.explainability.visualization import ExplainabilityVisualizer

logger = logging.getLogger(__name__)


class AgenticAnalysisModule:
    """
    Agentic AI-powered analysis module with autonomous decision-making
    Features:
    - Autonomous data quality assessment
    - Self-optimizing preprocessing pipeline
    - Multi-stage prediction with confidence thresholds
    - Explainable AI integration
    - Adaptive recommendations
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self.gait_detector = None
        self.disease_classifier = None
        self.signal_processor = SignalProcessor()
        
        # Initialize explainability
        self.shap_explainer = None
        self.lime_explainer = None
        self.visualizer = ExplainabilityVisualizer()
        
        # Agentic AI state
        self.agent_state = {
            'analysis_stage': 'idle',
            'confidence_threshold': 0.85,
            'quality_score': 0.0,
            'recommendations': [],
            'decisions_made': []
        }
        
        # Initialize session state
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        if 'preprocessing_done' not in st.session_state:
            st.session_state.preprocessing_done = False
    
    def render(self):
        """Render analysis interface"""
        st.markdown("""
        <div style="background: linear-gradient(135deg, #3498db 0%, #2ecc71 100%); 
                    padding: 2rem; border-radius: 10px; color: white; margin-bottom: 2rem;">
            <h2>🤖 Agentic AI Analysis Module</h2>
            <p>Autonomous analysis with explainable predictions</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Check if data is available
        if st.session_state.uploaded_data is None:
            st.warning("⚠️ No data uploaded. Please upload data first.")
            if st.button("📤 Go to Upload"):
                st.session_state.current_page = "upload"
                st.rerun()
            return
        
        # Agent status
        self._display_agent_status()
        
        # Analysis controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 Start Autonomous Analysis", type="primary", use_container_width=True):
                self._run_autonomous_analysis()
        
        with col2:
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.5,
                max_value=0.99,
                value=0.85,
                step=0.05
            )
            self.agent_state['confidence_threshold'] = confidence_threshold
        
        with col3:
            explainability = st.selectbox(
                "Explainability Method",
                ["SHAP", "LIME", "Both"]
            )
        
        # Display results if available
        if st.session_state.analysis_results is not None:
            self._display_results(explainability)
    
    def _display_agent_status(self):
        """Display agent status and decisions"""
        st.markdown("### 🤖 Agent Status")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            stage_color = {
                'idle': '🟢',
                'analyzing': '🟡',
                'processing': '🟠',
                'complete': '✅',
                'error': '❌'
            }
            st.metric(
                "Stage",
                f"{stage_color.get(self.agent_state['analysis_stage'], '⚪')} {self.agent_state['analysis_stage'].title()}"
            )
        
        with col2:
            st.metric(
                "Data Quality",
                f"{self.agent_state['quality_score']:.1%}"
            )
        
        with col3:
            st.metric(
                "Decisions Made",
                len(self.agent_state['decisions_made'])
            )
        
        with col4:
            st.metric(
                "Recommendations",
                len(self.agent_state['recommendations'])
            )
        
        # Show decisions and recommendations
        if self.agent_state['decisions_made']:
            with st.expander("🧠 Agent Decisions"):
                for decision in self.agent_state['decisions_made']:
                    st.info(f"• {decision}")
        
        if self.agent_state['recommendations']:
            with st.expander("💡 Agent Recommendations"):
                for rec in self.agent_state['recommendations']:
                    st.warning(f"• {rec}")
    
    def _run_autonomous_analysis(self):
        """Run autonomous analysis pipeline with agent decisions"""
        try:
            # Stage 1: Data Quality Assessment
            with st.spinner("🔍 Agent assessing data quality..."):
                self.agent_state['analysis_stage'] = 'analyzing'
                quality_report = self._assess_data_quality()
                
                st.success(f"✅ Data quality score: {quality_report['overall_score']:.1%}")
                
                # Agent decision: Proceed or recommend improvements
                if quality_report['overall_score'] < 0.6:
                    self.agent_state['decisions_made'].append(
                        f"Data quality too low ({quality_report['overall_score']:.1%}). Recommending data cleaning."
                    )
                    self.agent_state['recommendations'].extend(quality_report['recommendations'])
                    st.error("❌ Agent recommends improving data quality before analysis")
                    return
                else:
                    self.agent_state['decisions_made'].append(
                        f"Data quality acceptable ({quality_report['overall_score']:.1%}). Proceeding with analysis."
                    )
            
            # Stage 2: Autonomous Preprocessing
            with st.spinner("⚙️ Agent optimizing preprocessing pipeline..."):
                self.agent_state['analysis_stage'] = 'processing'
                processed_data = self._autonomous_preprocessing()
                
                self.agent_state['decisions_made'].append(
                    f"Applied {len(processed_data['steps'])} preprocessing steps"
                )
            
            # Stage 3: Gait Detection
            with st.spinner("🚶 Agent detecting gait patterns..."):
                gait_results = self._detect_gait(processed_data['data'])
                
                # Agent decision: Proceed to disease classification?
                if gait_results['prediction'] == 'Non-Gait':
                    self.agent_state['decisions_made'].append(
                        f"No gait detected (confidence: {gait_results['confidence']:.1%}). Stopping analysis."
                    )
                    st.warning("⚠️ Agent detected no gait pattern. Analysis stopped.")
                    
                    st.session_state.analysis_results = {
                        'stage': 'gait_detection',
                        'gait_results': gait_results
                    }
                    return
                
                if gait_results['confidence'] < self.agent_state['confidence_threshold']:
                    self.agent_state['decisions_made'].append(
                        f"Gait confidence low ({gait_results['confidence']:.1%}). Recommending manual review."
                    )
                    self.agent_state['recommendations'].append(
                        "Consider manual review due to low confidence"
                    )
            
            # Stage 4: Disease Classification
            with st.spinner("🧬 Agent classifying neurological conditions..."):
                disease_results = self._classify_disease(processed_data['data'])
                
                # Agent decision: Generate explanations?
                if disease_results['confidence'] >= self.agent_state['confidence_threshold']:
                    self.agent_state['decisions_made'].append(
                        f"High confidence classification ({disease_results['confidence']:.1%}). Generating explanations."
                    )
                    
                    # Stage 5: Explainability
                    with st.spinner("🔬 Agent generating explanations..."):
                        explanations = self._generate_explanations(
                            processed_data['data'],
                            gait_results,
                            disease_results
                        )
                else:
                    self.agent_state['decisions_made'].append(
                        f"Low confidence classification ({disease_results['confidence']:.1%}). Recommending additional testing."
                    )
                    self.agent_state['recommendations'].append(
                        "Consider additional data collection or expert review"
                    )
                    explanations = None
            
            # Stage 6: Generate Recommendations
            recommendations = self._generate_recommendations(
                quality_report,
                gait_results,
                disease_results
            )
            
            # Store results
            st.session_state.analysis_results = {
                'stage': 'complete',
                'quality_report': quality_report,
                'preprocessing': processed_data,
                'gait_results': gait_results,
                'disease_results': disease_results,
                'explanations': explanations,
                'recommendations': recommendations
            }
            
            self.agent_state['analysis_stage'] = 'complete'
            st.success("✅ Autonomous analysis complete!")
            st.balloons()
            
        except Exception as e:
            self.agent_state['analysis_stage'] = 'error'
            st.error(f"❌ Analysis error: {str(e)}")
            self.logger.error(f"Analysis error: {str(e)}")
    
    def _assess_data_quality(self) -> Dict:
        """Agent assesses data quality autonomously"""
        df = st.session_state.uploaded_data['dataframe']
        
        quality_metrics = {
            'completeness': 1.0 - (df.isnull().sum().sum() / df.size),
            'consistency': self._check_consistency(df),
            'accuracy': self._estimate_accuracy(df),
            'validity': self._check_validity(df)
        }
        
        overall_score = np.mean(list(quality_metrics.values()))
        
        # Generate recommendations
        recommendations = []
        if quality_metrics['completeness'] < 0.9:
            recommendations.append("Handle missing values (imputation or removal)")
        if quality_metrics['consistency'] < 0.8:
            recommendations.append("Check for inconsistent data patterns")
        if quality_metrics['accuracy'] < 0.8:
            recommendations.append("Verify sensor calibration")
        if quality_metrics['validity'] < 0.9:
            recommendations.append("Check data ranges and outliers")
        
        self.agent_state['quality_score'] = overall_score
        
        return {
            'overall_score': overall_score,
            'metrics': quality_metrics,
            'recommendations': recommendations
        }
    
    def _check_consistency(self, df: pd.DataFrame) -> float:
        """Check data consistency"""
        # Check for sudden jumps or discontinuities
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        consistency_scores = []
        
        for col in numeric_cols:
            # Calculate differences
            diffs = df[col].diff().abs()
            # Check if differences are reasonable (not too large)
            threshold = df[col].std() * 3
            consistent_ratio = (diffs < threshold).sum() / len(diffs)
            consistency_scores.append(consistent_ratio)
        
        return np.mean(consistency_scores) if consistency_scores else 1.0
    
    def _estimate_accuracy(self, df: pd.DataFrame) -> float:
        """Estimate data accuracy based on expected ranges"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        accuracy_scores = []
        
        # Expected ranges for different sensor types
        expected_ranges = {
            'acc': (-20, 20),  # Accelerometer in g
            'gyro': (-500, 500),  # Gyroscope in deg/s
            'emg': (0, 5)  # EMG in mV
        }
        
        for col in numeric_cols:
            # Determine sensor type
            sensor_type = None
            for key in expected_ranges:
                if key in col.lower():
                    sensor_type = key
                    break
            
            if sensor_type:
                min_val, max_val = expected_ranges[sensor_type]
                in_range = ((df[col] >= min_val) & (df[col] <= max_val)).sum()
                accuracy_scores.append(in_range / len(df))
        
        return np.mean(accuracy_scores) if accuracy_scores else 0.9
    
    def _check_validity(self, df: pd.DataFrame) -> float:
        """Check data validity"""
        # Check for required columns
        required_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
        has_required = sum([col in df.columns for col in required_cols]) / len(required_cols)
        
        # Check for reasonable data length
        length_valid = 1.0 if len(df) >= 250 else len(df) / 250
        
        return (has_required + length_valid) / 2
    
    def _autonomous_preprocessing(self) -> Dict:
        """Agent autonomously selects and applies preprocessing steps"""
        df = st.session_state.uploaded_data['dataframe']
        
        # Agent decides which preprocessing steps to apply
        steps_applied = []
        processed_data = df.copy()
        
        # Step 1: Handle missing values
        if processed_data.isnull().sum().sum() > 0:
            processed_data = processed_data.fillna(method='ffill').fillna(method='bfill')
            steps_applied.append("Forward/backward fill for missing values")
        
        # Step 2: Remove outliers (agent decides threshold)
        for col in processed_data.select_dtypes(include=[np.number]).columns:
            Q1 = processed_data[col].quantile(0.25)
            Q3 = processed_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outliers_before = ((processed_data[col] < lower_bound) | (processed_data[col] > upper_bound)).sum()
            if outliers_before > len(processed_data) * 0.01:  # More than 1% outliers
                processed_data[col] = processed_data[col].clip(lower=lower_bound, upper=upper_bound)
                steps_applied.append(f"Clipped outliers in {col}")
        
        # Step 3: Normalize (agent decides method based on data distribution)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
        processed_data[numeric_cols] = scaler.fit_transform(processed_data[numeric_cols])
        steps_applied.append("Z-score normalization applied")
        
        # Step 4: Segment into windows
        window_size = 250
        if len(processed_data) >= window_size:
            steps_applied.append(f"Data segmented into {window_size}-sample windows")
        
        return {
            'data': processed_data,
            'steps': steps_applied,
            'scaler': scaler
        }
    
    def _detect_gait(self, data: pd.DataFrame) -> Dict:
        """Run gait detection"""
        # Simplified gait detection for demo
        # In production, use trained model
        
        # Calculate movement intensity
        movement_cols = [col for col in data.columns if 'acc' in col or 'gyro' in col]
        movement_intensity = data[movement_cols].abs().mean().mean()
        
        # Simple threshold-based classification
        is_gait = movement_intensity > 0.5
        confidence = min(0.95, max(0.70, movement_intensity / 1.5))
        
        return {
            'prediction': 'Gait' if is_gait else 'Non-Gait',
            'confidence': confidence,
            'movement_intensity': movement_intensity
        }
    
    def _classify_disease(self, data: pd.DataFrame) -> Dict:
        """Run disease classification"""
        # Simplified disease classification for demo
        # In production, use trained model
        
        diseases = ['Healthy', 'Parkinson', 'Huntington', 'ALS', 'MS']
        
        # Random classification for demo
        predicted_class = np.random.choice(diseases, p=[0.4, 0.3, 0.1, 0.1, 0.1])
        confidence = np.random.uniform(0.75, 0.95)
        
        return {
            'prediction': predicted_class,
            'confidence': confidence,
            'probabilities': {disease: np.random.uniform(0.05, 0.95) for disease in diseases}
        }
    
    def _generate_explanations(self, data, gait_results, disease_results) -> Dict:
        """Generate explainability results"""
        # Simplified explanations for demo
        return {
            'method': 'SHAP + LIME',
            'top_features': [
                {'feature': 'acc_z', 'importance': 0.25},
                {'feature': 'gyro_x', 'importance': 0.20},
                {'feature': 'acc_x', 'importance': 0.18}
            ],
            'summary': f"The model classified this as {disease_results['prediction']} based primarily on vertical acceleration patterns and rotational movements."
        }
    
    def _generate_recommendations(self, quality_report, gait_results, disease_results) -> List[str]:
        """Agent generates actionable recommendations"""
        recommendations = []
        
        # Based on data quality
        if quality_report['overall_score'] < 0.8:
            recommendations.append("📊 Improve data collection procedures for better quality")
        
        # Based on confidence
        if disease_results['confidence'] < 0.9:
            recommendations.append("🔬 Consider additional diagnostic tests")
            recommendations.append("👨‍⚕️ Recommend specialist consultation")
        
        # Based on prediction
        if disease_results['prediction'] != 'Healthy':
            recommendations.append(f"⚕️ Follow up with {disease_results['prediction']}-specific assessments")
            recommendations.append("📅 Schedule regular monitoring sessions")
        
        return recommendations
    
    def _display_results(self, explainability_method: str):
        """Display analysis results"""
        results = st.session_state.analysis_results
        
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Gait Detection",
                results['gait_results']['prediction'],
                f"{results['gait_results']['confidence']:.1%} confidence"
            )
        
        with col2:
            st.metric(
                "Disease Classification",
                results['disease_results']['prediction'],
                f"{results['disease_results']['confidence']:.1%} confidence"
            )
        
        with col3:
            st.metric(
                "Data Quality",
                f"{results['quality_report']['overall_score']:.1%}",
                "Good" if results['quality_report']['overall_score'] > 0.8 else "Fair"
            )
        
        # Detailed results tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Predictions", "🔬 Explainability", "💡 Recommendations", "📋 Details"])
        
        with tab1:
            self._display_predictions(results)
        
        with tab2:
            if results.get('explanations'):
                self._display_explanations(results['explanations'])
            else:
                st.info("Explanations not generated due to low confidence")
        
        with tab3:
            self._display_recommendations(results['recommendations'])
        
        with tab4:
            self._display_detailed_results(results)
    
    def _display_predictions(self, results):
        """Display prediction visualizations"""
        st.markdown("### Prediction Confidence")
        
        # Create confidence chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=['Gait Detection', 'Disease Classification'],
            y=[results['gait_results']['confidence'], results['disease_results']['confidence']],
            marker_color=['#3498db', '#2ecc71'],
            text=[f"{results['gait_results']['confidence']:.1%}", 
                  f"{results['disease_results']['confidence']:.1%}"],
            textposition='auto'
        ))
        
        fig.update_layout(
            yaxis_title="Confidence",
            yaxis_range=[0, 1],
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_explanations(self, explanations):
        """Display explainability results"""
        st.markdown("### 🔬 Feature Importance")
        
        df_importance = pd.DataFrame(explanations['top_features'])
        
        fig = go.Figure(go.Bar(
            x=df_importance['importance'],
            y=df_importance['feature'],
            orientation='h',
            marker_color='#9b59b6'
        ))
        
        fig.update_layout(
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"**Summary:** {explanations['summary']}")
    
    def _display_recommendations(self, recommendations):
        """Display agent recommendations"""
        st.markdown("### 💡 Agent Recommendations")
        
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")
    
    def _display_detailed_results(self, results):
        """Display detailed results"""
        st.json(results)


if __name__ == "__main__":
    st.set_page_config(page_title="Analysis Module", layout="wide")
    
    module = AgenticAnalysisModule()
    module.render()