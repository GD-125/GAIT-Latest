# File: src/data/data_validator.py
# Advanced data validation for medical sensor data

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from scipy import stats
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class DataValidator:
    """Advanced validation for medical sensor data"""
    
    def __init__(self):
        self.validation_rules = {
            'accelerometer': {
                'range': (-50, 50),  # m/s²
                'typical_range': (-16, 16),
                'sampling_rate_min': 25,  # Hz
                'sampling_rate_max': 1000
            },
            'gyroscope': {
                'range': (-35, 35),  # rad/s
                'typical_range': (-10, 10),
                'sampling_rate_min': 25,
                'sampling_rate_max': 1000
            },
            'emg': {
                'range': (0, 5000),  # μV
                'typical_range': (0, 1000),
                'sampling_rate_min': 100,
                'sampling_rate_max': 2000
            }
        }
        
        self.quality_thresholds = {
            'missing_data_max': 0.10,  # 10% max missing data
            'outlier_percentage_max': 0.05,  # 5% max outliers
            'signal_noise_ratio_min': 10,  # dB
            'minimum_duration': 30,  # seconds
            'recommended_duration': 60  # seconds
        }
    
    def comprehensive_validation(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive data validation
        
        Args:
            data: Input DataFrame to validate
            
        Returns:
            Comprehensive validation report
        """
        
        report = {
            'overall_valid': True,
            'quality_score': 0.0,
            'errors': [],
            'warnings': [],
            'recommendations': [],
            'detailed_analysis': {}
        }
        
        try:
            # Basic structure validation
            structure_results = self._validate_structure(data)
            report['detailed_analysis']['structure'] = structure_results
            
            if not structure_results['valid']:
                report['overall_valid'] = False
                report['errors'].extend(structure_results['errors'])
            
            # Temporal validation
            temporal_results = self._validate_temporal_aspects(data)
            report['detailed_analysis']['temporal'] = temporal_results
            
            if temporal_results['warnings']:
                report['warnings'].extend(temporal_results['warnings'])
            
            # Signal quality validation
            quality_results = self._validate_signal_quality(data)
            report['detailed_analysis']['quality'] = quality_results
            
            if not quality_results['acceptable']:
                report['warnings'].extend(quality_results['issues'])
            
            # Statistical validation
            stats_results = self._validate_statistical_properties(data)
            report['detailed_analysis']['statistics'] = stats_results
            
            # Medical plausibility validation
            medical_results = self._validate_medical_plausibility(data)
            report['detailed_analysis']['medical'] = medical_results
            
            if medical_results['concerns']:
                report['warnings'].extend(medical_results['concerns'])
            
            # Anomaly detection
            anomaly_results = self._detect_anomalies(data)
            report['detailed_analysis']['anomalies'] = anomaly_results
            
            # Calculate overall quality score
            report['quality_score'] = self._calculate_quality_score(report['detailed_analysis'])
            
            # Generate recommendations
            report['recommendations'] = self._generate_recommendations(report['detailed_analysis'])
            
            logger.info(f"Data validation completed. Quality score: {report['quality_score']:.2f}")
            
        except Exception as e:
            logger.error(f"Error during comprehensive validation: {str(e)}")
            report['overall_valid'] = False
            report['errors'].append(f"Validation error: {str(e)}")
        
        return report
    
    def _validate_structure(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data structure and format"""
        
        results = {
            'valid': True,
            'errors': [],
            'info': {}
        }
        
        # Check if DataFrame is empty
        if data.empty:
            results['valid'] = False
            results['errors'].append("Dataset is empty")
            return results
        
        # Required sensor columns
        required_patterns = ['accel_', 'gyro_']
        found_sensors = []
        
        for pattern in required_patterns:
            sensor_cols = [col for col in data.columns if pattern in col.lower()]
            if sensor_cols:
                found_sensors.append(pattern.rstrip('_'))
            
        results['info']['found_sensors'] = found_sensors
        results['info']['total_columns'] = len(data.columns)
        results['info']['total_rows'] = len(data)
        
        # Check for minimum sensor requirements
        if len(found_sensors) < 2:
            results['valid'] = False
            results['errors'].append("Insufficient sensor data. Need at least accelerometer and gyroscope.")
        
        # Check data types
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 3:
            results['valid'] = False
            results['errors'].append("Insufficient numeric columns for sensor data")
        
        return results
    
    def _validate_temporal_aspects(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate temporal aspects of the data"""
        
        results = {
            'valid': True,
            'warnings': [],
            'info': {}
        }
        
        # Check for timestamp column
        timestamp_cols = [col for col in data.columns 
                         if any(term in col.lower() for term in ['time', 'timestamp', 'date'])]
        
        if timestamp_cols:
            timestamp_col = timestamp_cols[0]
            try:
                timestamps = pd.to_datetime(data[timestamp_col])
                
                # Calculate duration
                duration = (timestamps.max() - timestamps.min()).total_seconds()
                results['info']['duration_seconds'] = duration
                results['info']['duration_minutes'] = duration / 60
                
                # Check minimum duration
                if duration < self.quality_thresholds['minimum_duration']:
                    results['warnings'].append(
                        f"Recording duration ({duration:.1f}s) is below minimum "
                        f"recommended ({self.quality_thresholds['minimum_duration']}s)"
                    )
                
                # Estimate sampling rate
                time_diffs = timestamps.diff().dropna()
                if len(time_diffs) > 0:
                    median_interval = time_diffs.median().total_seconds()
                    if median_interval > 0:
                        sampling_rate = 1 / median_interval
                        results['info']['estimated_sampling_rate'] = sampling_rate
                        
                        # Check sampling rate adequacy
                        if sampling_rate < 25:
                            results['warnings'].append(
                                f"Low sampling rate ({sampling_rate:.1f} Hz). "
                                "Recommend ≥50 Hz for gait analysis"
                            )
                
                # Check for temporal gaps
                large_gaps = time_diffs[time_diffs > time_diffs.quantile(0.95) * 5]
                if len(large_gaps) > 0:
                    results['warnings'].append(
                        f"Found {len(large_gaps)} large temporal gaps in data"
                    )
                
            except Exception as e:
                results['warnings'].append(f"Could not process timestamps: {str(e)}")
        else:
            results['info']['estimated_sampling_rate'] = 50  # Default assumption
            results['warnings'].append("No timestamp column found. Cannot validate temporal consistency")
        
        return results
    
    def _validate_signal_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate signal quality metrics"""
        
        results = {
            'acceptable': True,
            'issues': [],
            'metrics': {}
        }
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_metrics = {}
            signal = data[col].dropna()
            
            if len(signal) == 0:
                results['issues'].append(f"Column {col} has no valid data")
                continue
            
            # Missing data percentage
            missing_pct = data[col].isnull().sum() / len(data)
            col_metrics['missing_percentage'] = missing_pct * 100
            
            if missing_pct > self.quality_thresholds['missing_data_max']:
                results['acceptable'] = False
                results['issues'].append(
                    f"Column {col}: {missing_pct*100:.1f}% missing data "
                    f"(max allowed: {self.quality_thresholds['missing_data_max']*100}%)"
                )
            
            # Signal-to-noise ratio estimation
            try:
                signal_power = np.var(signal)
                noise_estimate = np.var(np.diff(signal)) / 2  # High-frequency noise estimate
                if noise_estimate > 0:
                    snr_db = 10 * np.log10(signal_power / noise_estimate)
                    col_metrics['snr_db'] = snr_db
                    
                    if snr_db < self.quality_thresholds['signal_noise_ratio_min']:
                        results['issues'].append(
                            f"Column {col}: Low SNR ({snr_db:.1f} dB). "
                            f"Recommend ≥{self.quality_thresholds['signal_noise_ratio_min']} dB"
                        )
                else:
                    col_metrics['snr_db'] = float('inf')
                    
            except Exception as e:
                logger.warning(f"Could not calculate SNR for {col}: {str(e)}")
                col_metrics['snr_db'] = 0
            
            # Range validation based on sensor type
            sensor_type = self._identify_sensor_type(col)
            if sensor_type in self.validation_rules:
                rules = self.validation_rules[sensor_type]
                
                # Check for values outside valid range
                out_of_range = (signal < rules['range'][0]) | (signal > rules['range'][1])
                out_of_range_pct = out_of_range.sum() / len(signal)
                col_metrics['out_of_range_percentage'] = out_of_range_pct * 100
                
                if out_of_range_pct > 0.01:  # >1% out of range
                    results['issues'].append(
                        f"Column {col}: {out_of_range_pct*100:.1f}% values outside "
                        f"valid range {rules['range']}"
                    )
                
                # Check for values outside typical range
                out_of_typical = (signal < rules['typical_range'][0]) | (signal > rules['typical_range'][1])
                out_of_typical_pct = out_of_typical.sum() / len(signal)
                col_metrics['out_of_typical_percentage'] = out_of_typical_pct * 100
                
                if out_of_typical_pct > 0.10:  # >10% outside typical
                    results['issues'].append(
                        f"Column {col}: {out_of_typical_pct*100:.1f}% values outside "
                        f"typical range {rules['typical_range']} (may indicate sensor issues)"
                    )
            
            # Check for constant values (dead sensor)
            if signal.std() < 1e-6:
                results['acceptable'] = False
                results['issues'].append(f"Column {col}: Signal appears constant (possible sensor failure)")
            
            # Check for saturation
            value_range = signal.max() - signal.min()
            if value_range < signal.std() * 0.1:
                results['issues'].append(f"Column {col}: Signal may be saturated or clipped")
            
            results['metrics'][col] = col_metrics
        
        return results
    
    def _identify_sensor_type(self, column_name: str) -> str:
        """Identify sensor type from column name"""
        col_lower = column_name.lower()
        
        if 'accel' in col_lower:
            return 'accelerometer'
        elif 'gyro' in col_lower:
            return 'gyroscope'
        elif 'emg' in col_lower:
            return 'emg'
        else:
            return 'unknown'
    
    def _validate_statistical_properties(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate statistical properties of signals"""
        
        results = {
            'properties': {},
            'normality_tests': {},
            'correlations': {}
        }
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns[:10]  # Limit to first 10 columns
        
        for col in numeric_cols:
            signal = data[col].dropna()
            
            if len(signal) < 10:
                continue
            
            # Basic statistical properties
            properties = {
                'mean': float(signal.mean()),
                'std': float(signal.std()),
                'skewness': float(stats.skew(signal)),
                'kurtosis': float(stats.kurtosis(signal)),
                'median': float(signal.median()),
                'iqr': float(signal.quantile(0.75) - signal.quantile(0.25))
            }
            
            results['properties'][col] = properties
            
            # Normality test (if reasonable sample size)
            if 20 <= len(signal) <= 5000:
                try:
                    shapiro_stat, shapiro_p = stats.shapiro(signal[:5000])  # Shapiro-Wilk test
                    results['normality_tests'][col] = {
                        'shapiro_wilk_statistic': float(shapiro_stat),
                        'shapiro_wilk_p_value': float(shapiro_p),
                        'is_normal': shapiro_p > 0.05
                    }
                except Exception:
                    results['normality_tests'][col] = {'test_failed': True}
        
        # Cross-correlations between sensors
        if len(numeric_cols) >= 2:
            try:
                correlation_matrix = data[numeric_cols].corr()
                
                # Find high correlations (potential sensor redundancy or issues)
                high_corr_pairs = []
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
                        corr_value = correlation_matrix.iloc[i, j]
                        if abs(corr_value) > 0.95 and not np.isnan(corr_value):
                            high_corr_pairs.append({
                                'sensor1': correlation_matrix.columns[i],
                                'sensor2': correlation_matrix.columns[j],
                                'correlation': float(corr_value)
                            })
                
                results['correlations'] = {
                    'high_correlation_pairs': high_corr_pairs,
                    'correlation_matrix': correlation_matrix.to_dict()
                }
                
            except Exception as e:
                logger.warning(f"Could not calculate correlations: {str(e)}")
        
        return results
    
    def _validate_medical_plausibility(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate medical plausibility of sensor readings"""
        
        results = {
            'plausible': True,
            'concerns': [],
            'assessments': {}
        }
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        # Check for physiologically implausible values
        for col in numeric_cols:
            signal = data[col].dropna()
            
            if len(signal) == 0:
                continue
            
            sensor_type = self._identify_sensor_type(col)
            assessment = {}
            
            if sensor_type == 'accelerometer':
                # Check for impossible acceleration values
                extreme_values = signal[(signal.abs() > 100)]  # >100 m/s² is very extreme
                if len(extreme_values) > 0:
                    results['concerns'].append(
                        f"{col}: Found {len(extreme_values)} physiologically extreme acceleration values (>100 m/s²)"
                    )
                
                # Check for lack of gravity component (should see ~9.8 m/s² in vertical)
                if 'z' in col.lower() or 'vertical' in col.lower():
                    if abs(signal.mean()) < 5:  # Suspiciously low for vertical accelerometer
                        results['concerns'].append(
                            f"{col}: Vertical accelerometer not showing expected gravity component"
                        )
                
                assessment['extreme_count'] = len(extreme_values)
                assessment['mean_magnitude'] = float(signal.abs().mean())
                
            elif sensor_type == 'gyroscope':
                # Check for impossible angular velocities
                extreme_values = signal[(signal.abs() > 50)]  # >50 rad/s is very extreme for gait
                if len(extreme_values) > 0:
                    results['concerns'].append(
                        f"{col}: Found {len(extreme_values)} physiologically extreme gyroscope values (>50 rad/s)"
                    )
                
                assessment['extreme_count'] = len(extreme_values)
                assessment['max_angular_velocity'] = float(signal.abs().max())
            
            elif sensor_type == 'emg':
                # Check for impossible EMG values
                if signal.min() < 0:
                    results['concerns'].append(f"{col}: EMG signal contains negative values (should be rectified)")
                
                if signal.max() > 10000:  # Very high EMG values
                    results['concerns'].append(f"{col}: Extremely high EMG values (>10mV)")
                
                assessment['has_negatives'] = bool(signal.min() < 0)
                assessment['max_amplitude'] = float(signal.max())
            
            results['assessments'][col] = assessment
        
        # Check for realistic gait patterns if we have accelerometer data
        accel_cols = [col for col in numeric_cols if 'accel' in col.lower()]
        if len(accel_cols) >= 2:
            try:
                # Calculate resultant acceleration
                accel_data = data[accel_cols[:3]].dropna()  # Use up to 3 axes
                resultant = np.sqrt((accel_data ** 2).sum(axis=1))
                
                # Check for reasonable gait-related accelerations
                gait_range = resultant.quantile(0.95) - resultant.quantile(0.05)
                
                if gait_range < 2:  # Very low acceleration range
                    results['concerns'].append(
                        "Low acceleration variability - may indicate minimal movement or sensor issues"
                    )
                elif gait_range > 50:  # Very high acceleration range
                    results['concerns'].append(
                        "Extremely high acceleration variability - may indicate measurement artifacts"
                    )
                
                results['assessments']['gait_dynamics'] = {
                    'acceleration_range': float(gait_range),
                    'mean_resultant': float(resultant.mean()),
                    'std_resultant': float(resultant.std())
                }
                
            except Exception as e:
                logger.warning(f"Could not assess gait dynamics: {str(e)}")
        
        return results
    
    def _detect_anomalies(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies in the sensor data"""
        
        results = {
            'anomalies_detected': False,
            'anomaly_methods': {},
            'recommendations': []
        }
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return results
        
        # Prepare data for anomaly detection
        clean_data = data[numeric_cols].dropna()
        
        if len(clean_data) < 50:  # Too few samples for reliable anomaly detection
            results['recommendations'].append("Too few samples for reliable anomaly detection")
            return results
        
        try:
            # Isolation Forest for outlier detection
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_labels = iso_forest.fit_predict(clean_data.iloc[:1000])  # Limit to 1000 samples
            
            anomaly_count = np.sum(anomaly_labels == -1)
            anomaly_percentage = anomaly_count / len(anomaly_labels) * 100
            
            results['anomaly_methods']['isolation_forest'] = {
                'anomaly_count': int(anomaly_count),
                'anomaly_percentage': float(anomaly_percentage),
                'threshold_exceeded': anomaly_percentage > 15  # >15% anomalies is concerning
            }
            
            if anomaly_percentage > 15:
                results['anomalies_detected'] = True
                results['recommendations'].append(
                    f"High percentage of anomalies detected ({anomaly_percentage:.1f}%). "
                    "Consider data cleaning or sensor calibration."
                )
            
        except Exception as e:
            logger.warning(f"Isolation Forest anomaly detection failed: {str(e)}")
        
        # Statistical outlier detection (Z-score method)
        try:
            z_scores = np.abs(stats.zscore(clean_data.iloc[:1000], axis=0, nan_policy='omit'))
            outliers = (z_scores > 3).any(axis=1)
            outlier_percentage = outliers.sum() / len(outliers) * 100
            
            results['anomaly_methods']['statistical_outliers'] = {
                'outlier_count': int(outliers.sum()),
                'outlier_percentage': float(outlier_percentage)
            }
            
            if outlier_percentage > self.quality_thresholds['outlier_percentage_max'] * 100:
                results['anomalies_detected'] = True
                results['recommendations'].append(
                    f"High percentage of statistical outliers ({outlier_percentage:.1f}%). "
                    "Consider reviewing sensor placement and calibration."
                )
                
        except Exception as e:
            logger.warning(f"Statistical outlier detection failed: {str(e)}")
        
        return results
    
    def _calculate_quality_score(self, detailed_analysis: Dict[str, Any]) -> float:
        """Calculate overall data quality score (0-100)"""
        
        score = 100.0  # Start with perfect score
        
        # Deduct points for structural issues
        if not detailed_analysis.get('structure', {}).get('valid', True):
            score -= 50  # Major deduction for structural problems
        
        # Deduct points for temporal issues
        temporal = detailed_analysis.get('temporal', {})
        if len(temporal.get('warnings', [])) > 0:
            score -= len(temporal['warnings']) * 5  # 5 points per warning
        
        # Deduct points for signal quality issues
        quality = detailed_analysis.get('quality', {})
        if not quality.get('acceptable', True):
            score -= len(quality.get('issues', [])) * 3  # 3 points per issue
        
        # Deduct points for medical plausibility concerns
        medical = detailed_analysis.get('medical', {})
        if medical.get('concerns'):
            score -= len(medical['concerns']) * 4  # 4 points per concern
        
        # Deduct points for anomalies
        anomalies = detailed_analysis.get('anomalies', {})
        if anomalies.get('anomalies_detected', False):
            score -= 10  # 10 points for detected anomalies
        
        return max(0.0, score)  # Ensure non-negative score
    
    def _generate_recommendations(self, detailed_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results"""
        
        recommendations = []
        
        # Structure recommendations
        structure = detailed_analysis.get('structure', {})
        if not structure.get('valid', True):
            recommendations.append("Fix data structure issues before proceeding with analysis")
        
        # Temporal recommendations
        temporal = detailed_analysis.get('temporal', {})
        duration = temporal.get('info', {}).get('duration_seconds', 0)
        
        if duration < self.quality_thresholds['minimum_duration']:
            recommendations.append(
                f"Collect longer recordings (current: {duration:.1f}s, recommended: ≥60s) "
                "for more reliable analysis"
            )
        
        sampling_rate = temporal.get('info', {}).get('estimated_sampling_rate', 0)
        if sampling_rate > 0 and sampling_rate < 50:
            recommendations.append(
                f"Use higher sampling rate (current: {sampling_rate:.1f} Hz, recommended: ≥50 Hz) "
                "for better gait analysis resolution"
            )
        
        # Signal quality recommendations
        quality = detailed_analysis.get('quality', {})
        if not quality.get('acceptable', True):
            recommendations.append("Address signal quality issues through sensor calibration or replacement")
            
            # Check for specific quality metrics
            for sensor, metrics in quality.get('metrics', {}).items():
                if metrics.get('snr_db', float('inf')) < 15:
                    recommendations.append(f"Improve signal-to-noise ratio for {sensor} sensor")
                
                if metrics.get('missing_percentage', 0) > 5:
                    recommendations.append(f"Reduce missing data in {sensor} (currently {metrics['missing_percentage']:.1f}%)")
        
        # Medical plausibility recommendations
        medical = detailed_analysis.get('medical', {})
        if medical.get('concerns'):
            recommendations.append("Review sensor placement and ensure proper calibration")
            recommendations.append("Verify that sensors are functioning within physiological ranges")
        
        # Anomaly recommendations
        anomalies = detailed_analysis.get('anomalies', {})
        if anomalies.get('anomalies_detected', False):
            recommendations.append("Investigate and clean detected anomalies before analysis")
            recommendations.append("Consider implementing real-time data quality monitoring")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Data quality is good - proceed with confidence!")
        else:
            recommendations.append("Consider implementing automated data quality checks in your collection pipeline")
        
        return recommendations