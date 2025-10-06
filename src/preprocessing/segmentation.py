# File: src/preprocessing/segmentation.py
# Signal segmentation for time series analysis

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from scipy import signal
import logging

logger = logging.getLogger(__name__)

class SignalSegmenter:
    """Advanced signal segmentation for gait analysis"""
    
    def __init__(self):
        self.segments = []
        self.segment_metadata = []
        self.segmentation_params = {}
    
    def create_fixed_windows(self,
                           data: pd.DataFrame,
                           window_size: float = 5.0,
                           overlap: float = 0.25,
                           sampling_rate: int = 50) -> List[pd.DataFrame]:
        """
        Create fixed-size sliding windows
        
        Args:
            data: Input DataFrame
            window_size: Window size in seconds
            overlap: Overlap ratio (0-1)
            sampling_rate: Sampling rate in Hz
            
        Returns:
            List of DataFrame segments
        """
        
        segments = []
        
        try:
            window_samples = int(window_size * sampling_rate)
            step_samples = int(window_samples * (1 - overlap))
            
            logger.info(f"Creating fixed windows: size={window_size}s, overlap={overlap*100}%")
            
            for start_idx in range(0, len(data) - window_samples + 1, step_samples):
                end_idx = start_idx + window_samples
                
                segment = data.iloc[start_idx:end_idx].copy()
                segment.reset_index(drop=True, inplace=True)
                
                # Add segment metadata
                segment.attrs = {
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'duration': window_size,
                    'segment_id': len(segments)
                }
                
                segments.append(segment)
            
            self.segments = segments
            logger.info(f"Created {len(segments)} fixed windows")
            
            return segments
            
        except Exception as e:
            logger.error(f"Error creating fixed windows: {str(e)}")
            return []
    
    def create_adaptive_segments(self,
                               data: pd.DataFrame,
                               activity_column: str,
                               min_segment_duration: float = 2.0,
                               max_segment_duration: float = 30.0,
                               activity_threshold: float = None,
                               sampling_rate: int = 50) -> List[pd.DataFrame]:
        """
        Create adaptive segments based on activity level
        
        Args:
            data: Input DataFrame
            activity_column: Column to use for activity detection
            min_segment_duration: Minimum segment duration in seconds
            max_segment_duration: Maximum segment duration in seconds
            activity_threshold: Activity threshold for segmentation
            sampling_rate: Sampling rate in Hz
            
        Returns:
            List of DataFrame segments
        """
        
        segments = []
        
        try:
            if activity_column not in data.columns:
                logger.error(f"Activity column '{activity_column}' not found")
                return []
            
            activity_signal = data[activity_column].values
            
            # Calculate activity threshold if not provided
            if activity_threshold is None:
                activity_threshold = np.mean(activity_signal) + 0.5 * np.std(activity_signal)
            
            # Smooth activity signal
            smoothed_activity = signal.savgol_filter(activity_signal, window_length=min(51, len(activity_signal)//2*2+1), polyorder=3)
            
            # Find activity periods
            active_mask = smoothed_activity > activity_threshold
            
            # Find transitions
            transitions = np.where(np.diff(active_mask.astype(int)))[0]
            
            min_samples = int(min_segment_duration * sampling_rate)
            max_samples = int(max_segment_duration * sampling_rate)
            
            segment_starts = [0] + (transitions + 1).tolist()
            segment_ends = transitions.tolist() + [len(data)]
            
            for start_idx, end_idx in zip(segment_starts, segment_ends):
                segment_length = end_idx - start_idx
                
                # Split long segments
                if segment_length > max_samples:
                    for sub_start in range(start_idx, end_idx, max_samples):
                        sub_end = min(sub_start + max_samples, end_idx)
                        if sub_end - sub_start >= min_samples:
                            segment = data.iloc[sub_start:sub_end].copy()
                            segment.reset_index(drop=True, inplace=True)
                            
                            segment.attrs = {
                                'start_idx': sub_start,
                                'end_idx': sub_end,
                                'duration': (sub_end - sub_start) / sampling_rate,
                                'segment_id': len(segments),
                                'activity_level': np.mean(smoothed_activity[sub_start:sub_end])
                            }
                            
                            segments.append(segment)
                
                # Keep segments of appropriate length
                elif segment_length >= min_samples:
                    segment = data.iloc[start_idx:end_idx].copy()
                    segment.reset_index(drop=True, inplace=True)
                    
                    segment.attrs = {
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'duration': segment_length / sampling_rate,
                        'segment_id': len(segments),
                        'activity_level': np.mean(smoothed_activity[start_idx:end_idx])
                    }
                    
                    segments.append(segment)
            
            self.segments = segments
            logger.info(f"Created {len(segments)} adaptive segments")
            
            return segments
            
        except Exception as e:
            logger.error(f"Error creating adaptive segments: {str(e)}")
            return []
    
    def create_gait_cycle_segments(self,
                                 data: pd.DataFrame,
                                 accelerometer_columns: List[str],
                                 sampling_rate: int = 50,
                                 min_cycle_duration: float = 0.8,
                                 max_cycle_duration: float = 2.5) -> List[pd.DataFrame]:
        """
        Segment data into gait cycles based on accelerometer patterns
        
        Args:
            data: Input DataFrame
            accelerometer_columns: List of accelerometer column names
            sampling_rate: Sampling rate in Hz
            min_cycle_duration: Minimum gait cycle duration in seconds
            max_cycle_duration: Maximum gait cycle duration in seconds
            
        Returns:
            List of DataFrame segments representing gait cycles
        """
        
        segments = []
        
        try:
            # Check if accelerometer columns exist
            available_accel_cols = [col for col in accelerometer_columns if col in data.columns]
            
            if not available_accel_cols:
                logger.error("No accelerometer columns found")
                return []
            
            # Calculate resultant acceleration
            accel_data = data[available_accel_cols].fillna(method='ffill')
            resultant_accel = np.sqrt(np.sum(accel_data**2, axis=1))
            
            # Detect heel strikes (gait cycle start points)
            heel_strikes = self._detect_heel_strikes(resultant_accel, sampling_rate)
            
            min_samples = int(min_cycle_duration * sampling_rate)
            max_samples = int(max_cycle_duration * sampling_rate)
            
            # Create segments between heel strikes
            for i in range(len(heel_strikes) - 1):
                start_idx = heel_strikes[i]
                end_idx = heel_strikes[i + 1]
                cycle_length = end_idx - start_idx
                
                # Filter cycles by duration
                if min_samples <= cycle_length <= max_samples:
                    segment = data.iloc[start_idx:end_idx].copy()
                    segment.reset_index(drop=True, inplace=True)
                    
                    # Calculate cycle characteristics
                    cycle_duration = cycle_length / sampling_rate
                    cycle_cadence = 60 / cycle_duration  # steps per minute
                    
                    segment.attrs = {
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'duration': cycle_duration,
                        'segment_id': len(segments),
                        'cycle_cadence': cycle_cadence,
                        'segment_type': 'gait_cycle'
                    }
                    
                    segments.append(segment)
            
            self.segments = segments
            logger.info(f"Created {len(segments)} gait cycle segments")
            
            return segments
            
        except Exception as e:
            logger.error(f"Error creating gait cycle segments: {str(e)}")
            return []
    
    def _detect_heel_strikes(self, resultant_accel: np.ndarray, sampling_rate: int = 50) -> List[int]:
        """Detect heel strike events in accelerometer data"""
        
        try:
            # Smooth the signal to reduce noise
            from scipy.ndimage import gaussian_filter1d
            smoothed = gaussian_filter1d(resultant_accel, sigma=2)
            
            # Find peaks that could represent heel strikes
            min_distance = int(0.6 * sampling_rate)  # Minimum 0.6 seconds between heel strikes
            min_height = np.mean(smoothed) + 0.3 * np.std(smoothed)
            
            peaks, properties = signal.find_peaks(smoothed, 
                                                distance=min_distance, 
                                                height=min_height,
                                                prominence=0.2 * np.std(smoothed))
            
            # Refine peak detection by looking for the actual impact point
            # (usually occurs slightly before the peak)
            heel_strikes = []
            
            for peak in peaks:
                # Look for the steepest positive slope before the peak
                search_window = int(0.2 * sampling_rate)  # 0.2 seconds before peak
                start_search = max(0, peak - search_window)
                
                search_region = smoothed[start_search:peak]
                if len(search_region) > 5:
                    # Find point with maximum positive derivative
                    derivatives = np.diff(search_region)
                    max_slope_idx = np.argmax(derivatives)
                    heel_strike = start_search + max_slope_idx
                    heel_strikes.append(heel_strike)
                else:
                    heel_strikes.append(peak)
            
            return heel_strikes
            
        except Exception as e:
            logger.error(f"Error detecting heel strikes: {str(e)}")
            return []
    
    def create_event_based_segments(self,
                                  data: pd.DataFrame,
                                  event_column: str,
                                  event_values: List[Any],
                                  pre_event_duration: float = 2.0,
                                  post_event_duration: float = 3.0,
                                  sampling_rate: int = 50) -> List[pd.DataFrame]:
        """
        Create segments around specific events
        
        Args:
            data: Input DataFrame
            event_column: Column containing event markers
            event_values: List of event values to segment around
            pre_event_duration: Duration before event in seconds
            post_event_duration: Duration after event in seconds
            sampling_rate: Sampling rate in Hz
            
        Returns:
            List of DataFrame segments around events
        """
        
        segments = []
        
        try:
            if event_column not in data.columns:
                logger.error(f"Event column '{event_column}' not found")
                return []
            
            pre_samples = int(pre_event_duration * sampling_rate)
            post_samples = int(post_event_duration * sampling_rate)
            
            # Find event occurrences
            for event_value in event_values:
                event_indices = data[data[event_column] == event_value].index.tolist()
                
                for event_idx in event_indices:
                    start_idx = max(0, event_idx - pre_samples)
                    end_idx = min(len(data), event_idx + post_samples)
                    
                    if end_idx - start_idx >= pre_samples + post_samples - 10:  # Allow some tolerance
                        segment = data.iloc[start_idx:end_idx].copy()
                        segment.reset_index(drop=True, inplace=True)
                        
                        # Mark the event position within the segment
                        event_position = event_idx - start_idx
                        
                        segment.attrs = {
                            'start_idx': start_idx,
                            'end_idx': end_idx,
                            'duration': (end_idx - start_idx) / sampling_rate,
                            'segment_id': len(segments),
                            'event_type': event_value,
                            'event_position': event_position,
                            'segment_type': 'event_based'
                        }
                        
                        segments.append(segment)
            
            self.segments = segments
            logger.info(f"Created {len(segments)} event-based segments")
            
            return segments
            
        except Exception as e:
            logger.error(f"Error creating event-based segments: {str(e)}")
            return []
    
    def create_change_point_segments(self,
                                   data: pd.DataFrame,
                                   signal_column: str,
                                   min_segment_length: int = 100,
                                   penalty: float = 10.0) -> List[pd.DataFrame]:
        """
        Create segments based on change point detection
        
        Args:
            data: Input DataFrame
            signal_column: Column to analyze for change points
            min_segment_length: Minimum segment length in samples
            penalty: Penalty parameter for change point detection
            
        Returns:
            List of DataFrame segments
        """
        
        segments = []
        
        try:
            if signal_column not in data.columns:
                logger.error(f"Signal column '{signal_column}' not found")
                return []
            
            signal_data = data[signal_column].fillna(method='ffill').values
            
            # Simple change point detection using variance
            change_points = self._detect_change_points(signal_data, min_segment_length, penalty)
            
            # Create segments between change points
            segment_starts = [0] + change_points
            segment_ends = change_points + [len(data)]
            
            for start_idx, end_idx in zip(segment_starts, segment_ends):
                if end_idx - start_idx >= min_segment_length:
                    segment = data.iloc[start_idx:end_idx].copy()
                    segment.reset_index(drop=True, inplace=True)
                    
                    # Calculate segment statistics
                    segment_signal = signal_data[start_idx:end_idx]
                    segment_mean = np.mean(segment_signal)
                    segment_std = np.std(segment_signal)
                    
                    segment.attrs = {
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'duration': (end_idx - start_idx) / 50,  # Assuming 50 Hz
                        'segment_id': len(segments),
                        'segment_mean': segment_mean,
                        'segment_std': segment_std,
                        'segment_type': 'change_point'
                    }
                    
                    segments.append(segment)
            
            self.segments = segments
            logger.info(f"Created {len(segments)} change point segments")
            
            return segments
            
        except Exception as e:
            logger.error(f"Error creating change point segments: {str(e)}")
            return []
    
    def _detect_change_points(self, signal: np.ndarray, min_segment_length: int, penalty: float) -> List[int]:
        """Simple change point detection algorithm"""
        
        try:
            change_points = []
            n = len(signal)
            
            # Sliding window approach for change point detection
            window_size = min_segment_length
            
            for i in range(window_size, n - window_size, window_size // 2):
                # Calculate variance before and after potential change point
                before_segment = signal[i - window_size:i]
                after_segment = signal[i:i + window_size]
                
                var_before = np.var(before_segment)
                var_after = np.var(after_segment)
                
                # Combined segment variance
                combined_segment = signal[i - window_size:i + window_size]
                var_combined = np.var(combined_segment)
                
                # Change score (simplified)
                change_score = var_combined - (var_before + var_after) / 2
                
                # If change score exceeds penalty, mark as change point
                if change_score > penalty:
                    # Avoid change points too close to each other
                    if not change_points or (i - change_points[-1]) >= min_segment_length:
                        change_points.append(i)
            
            return change_points
            
        except Exception as e:
            logger.error(f"Error in change point detection: {str(e)}")
            return []
    
    def merge_short_segments(self,
                           segments: List[pd.DataFrame],
                           min_duration: float = 1.0,
                           sampling_rate: int = 50) -> List[pd.DataFrame]:
        """
        Merge segments that are too short with adjacent segments
        
        Args:
            segments: List of segments to process
            min_duration: Minimum acceptable duration in seconds
            sampling_rate: Sampling rate in Hz
            
        Returns:
            List of merged segments
        """
        
        try:
            min_samples = int(min_duration * sampling_rate)
            merged_segments = []
            
            i = 0
            while i < len(segments):
                current_segment = segments[i]
                
                if len(current_segment) < min_samples:
                    # Try to merge with next segment
                    if i + 1 < len(segments):
                        next_segment = segments[i + 1]
                        
                        # Merge segments
                        merged_data = pd.concat([current_segment, next_segment], ignore_index=True)
                        
                        # Update attributes
                        merged_data.attrs = {
                            'start_idx': current_segment.attrs.get('start_idx', 0),
                            'end_idx': next_segment.attrs.get('end_idx', len(merged_data)),
                            'duration': len(merged_data) / sampling_rate,
                            'segment_id': len(merged_segments),
                            'segment_type': 'merged'
                        }
                        
                        merged_segments.append(merged_data)
                        i += 2  # Skip next segment as it's been merged
                    else:
                        # Last segment is too short, try to merge with previous
                        if merged_segments:
                            last_merged = merged_segments[-1]
                            combined_data = pd.concat([last_merged, current_segment], ignore_index=True)
                            
                            combined_data.attrs = {
                                'start_idx': last_merged.attrs.get('start_idx', 0),
                                'end_idx': current_segment.attrs.get('end_idx', len(combined_data)),
                                'duration': len(combined_data) / sampling_rate,
                                'segment_id': len(merged_segments) - 1,
                                'segment_type': 'merged'
                            }
                            
                            merged_segments[-1] = combined_data
                        else:
                            # Keep the short segment if it's the only one
                            merged_segments.append(current_segment)
                        i += 1
                else:
                    # Segment is long enough, keep as is
                    merged_segments.append(current_segment)
                    i += 1
            
            logger.info(f"Merged segments: {len(segments)} -> {len(merged_segments)}")
            
            return merged_segments
            
        except Exception as e:
            logger.error(f"Error merging short segments: {str(e)}")
            return segments
    
    def get_segment_quality_scores(self, segments: List[pd.DataFrame]) -> List[Dict[str, float]]:
        """
        Calculate quality scores for segments
        
        Args:
            segments: List of segments to evaluate
            
        Returns:
            List of quality score dictionaries
        """
        
        quality_scores = []
        
        try:
            for i, segment in enumerate(segments):
                numeric_columns = segment.select_dtypes(include=[np.number]).columns
                
                if len(numeric_columns) == 0:
                    quality_scores.append({'overall_quality': 0.0})
                    continue
                
                scores = {}
                
                # Completeness score
                completeness = 1.0 - (segment[numeric_columns].isnull().sum().sum() / 
                                    (len(segment) * len(numeric_columns)))
                scores['completeness'] = float(completeness)
                
                # Signal-to-noise ratio (simplified)
                snr_scores = []
                for col in numeric_columns:
                    signal_data = segment[col].dropna()
                    if len(signal_data) > 10:
                        signal_power = np.var(signal_data)
                        noise_power = np.var(np.diff(signal_data)) / 2
                        if noise_power > 0:
                            snr = 10 * np.log10(signal_power / noise_power)
                            snr_scores.append(max(0, min(40, snr)) / 40)  # Normalize to 0-1
                
                scores['signal_quality'] = float(np.mean(snr_scores) if snr_scores else 0.5)
                
                # Duration appropriateness (segments should not be too short or too long)
                duration = len(segment) / 50  # Assuming 50 Hz
                if 2 <= duration <= 30:  # Good duration range
                    duration_score = 1.0
                elif duration < 2:
                    duration_score = duration / 2
                else:  # duration > 30
                    duration_score = max(0.3, 30 / duration)
                
                scores['duration_appropriateness'] = float(duration_score)
                
                # Overall quality (weighted average)
                overall_quality = (0.4 * scores['completeness'] + 
                                 0.4 * scores['signal_quality'] + 
                                 0.2 * scores['duration_appropriateness'])
                
                scores['overall_quality'] = float(overall_quality)
                scores['segment_id'] = i
                
                quality_scores.append(scores)
            
            return quality_scores
            
        except Exception as e:
            logger.error(f"Error calculating quality scores: {str(e)}")
            return [{'overall_quality': 0.5} for _ in segments]
    
    def filter_segments_by_quality(self,
                                 segments: List[pd.DataFrame],
                                 min_quality: float = 0.6) -> List[pd.DataFrame]:
        """
        Filter segments based on quality scores
        
        Args:
            segments: List of segments to filter
            min_quality: Minimum quality threshold (0-1)
            
        Returns:
            List of high-quality segments
        """
        
        try:
            quality_scores = self.get_segment_quality_scores(segments)
            
            high_quality_segments = []
            
            for segment, scores in zip(segments, quality_scores):
                if scores['overall_quality'] >= min_quality:
                    high_quality_segments.append(segment)
            
            logger.info(f"Quality filtering: {len(segments)} -> {len(high_quality_segments)} segments "
                       f"(threshold: {min_quality})")
            
            return high_quality_segments
            
        except Exception as e:
            logger.error(f"Error filtering segments by quality: {str(e)}")
            return segments
    
    def get_segmentation_summary(self) -> Dict[str, Any]:
        """Get summary of current segmentation"""
        
        if not self.segments:
            return {'message': 'No segments created'}
        
        durations = []
        segment_types = []
        
        for segment in self.segments:
            if hasattr(segment, 'attrs') and segment.attrs:
                durations.append(segment.attrs.get('duration', 0))
                segment_types.append(segment.attrs.get('segment_type', 'unknown'))
            else:
                durations.append(len(segment) / 50)  # Assume 50 Hz
                segment_types.append('unknown')
        
        summary = {
            'total_segments': len(self.segments),
            'mean_duration': float(np.mean(durations)) if durations else 0,
            'std_duration': float(np.std(durations)) if durations else 0,
            'min_duration': float(np.min(durations)) if durations else 0,
            'max_duration': float(np.max(durations)) if durations else 0,
            'segment_types': dict(pd.Series(segment_types).value_counts()),
            'total_data_duration': float(np.sum(durations)) if durations else 0
        }
        
        return summary

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    
    timestamps = pd.date_range('2024-01-01', periods=1000, freq='20ms')
    sample_data = pd.DataFrame({
        'timestamp': timestamps,
        'accel_x': np.random.randn(1000),
        'accel_y': np.random.randn(1000),
        'accel_z': 9.8 + np.random.randn(1000) * 0.5,
        'gyro_x': np.random.randn(1000) * 0.1,
        'activity_level': np.abs(np.random.randn(1000)) + np.sin(np.linspace(0, 10*np.pi, 1000))
    })
    
    # Test segmentation
    segmenter = SignalSegmenter()
    
    # Test fixed windows
    fixed_segments = segmenter.create_fixed_windows(
        sample_data, window_size=5.0, overlap=0.25, sampling_rate=50
    )
    print(f"Fixed windows: {len(fixed_segments)} segments")
    
    # Test adaptive segmentation
    adaptive_segments = segmenter.create_adaptive_segments(
        sample_data, 'activity_level', sampling_rate=50
    )
    print(f"Adaptive segments: {len(adaptive_segments)} segments")
    
    # Test quality filtering
    quality_scores = segmenter.get_segment_quality_scores(fixed_segments[:5])
    print(f"Quality scores for first 5 segments: {[s['overall_quality'] for s in quality_scores]}")
    
    # Get summary
    summary = segmenter.get_segmentation_summary()
    print(f"Segmentation summary: {summary}")            
    
    # Gait frequency band (0.5-3 Hz)
    gait_freq_mask = (positive_frequencies >= 0.5) & (positive_frequencies <= 3)
    features[f'{prefix}_power_gait_band'] = float(np.sum(power_spectrum[gait_freq_mask]))
            
    # Tremor frequency band (3-8 Hz)
    tremor_freq_mask = (positive_frequencies >= 3) & (positive_frequencies <= 8)
    features[f'{prefix}_power_tremor_band'] = float(np.sum(power_spectrum[tremor_freq_mask]))
            
    # High frequency (8-20 Hz)
    high_freq_mask = (positive_frequencies >= 8) & (positive_frequencies <= 20)
    features[f'{prefix}_power_high_band'] = float(np.sum(power_spectrum[high_freq_mask]))
           
    # Spectral entropy
    # Add small epsilon to avoid log(0)
    power_spectrum_norm = power_spectrum + 1e-12
    spectral_entropy = -np.sum(power_spectrum_norm * np.log2(power_spectrum_norm))
    features[f'{prefix}_spectral_entropy'] = float(spectral_entropy)
            
    # Peak frequency features
    peaks, _ = signal.find_peaks(magnitude_spectrum, height=np.max(magnitude_spectrum) * 0.1)
    features[f'{prefix}_num_spectral_peaks'] = len(peaks)
    try:
        if len(peaks) > 0:
            peak_frequencies = positive_frequencies[peaks]
            features[f'{prefix}_first_peak_freq'] = float(peak_frequencies[0])
            features[f'{prefix}_peak_freq_std'] = float(np.std(peak_frequencies))
        else:
            features[f'{prefix}_first_peak_freq'] = 0.0
            features[f'{prefix}_peak_freq_std'] = 0.0
    #return feature

    except Exception as e:
        logger.warning(f"Error extracting frequency domain features for {prefix}: {str(e)}")
        # Return zeros for failed features
        features.update({f'{prefix}_{feat}': 0.0 for feat in [
            'spectral_centroid', 'spectral_spread', 'spectral_rolloff', 'spectral_flux',
            'dominant_frequency', 'dominant_frequency_power', 'power_0_2hz', 'power_gait_band',
            'power_tremor_band', 'power_high_band', 'spectral_entropy', 'num_spectral_peaks',
            'first_peak_freq', 'peak_freq_std'
        ]})
        #return feature

    def _calculate_spectral_rolloff(self, frequencies: np.ndarray, power: np.ndarray, 
                                  rolloff_point: float = 0.85) -> float:
        """Calculate spectral rolloff frequency"""
        try:
            cumulative_power = np.cumsum(power)
            total_power = cumulative_power[-1]
            rolloff_idx = np.where(cumulative_power >= rolloff_point * total_power)[0]
            
            if len(rolloff_idx) > 0:
                return frequencies[rolloff_idx[0]]
            else:
                return frequencies[-1]
        except:
            return 0.0
    
    def _extract_wavelet_features(self, signal: np.ndarray, prefix: str) -> Dict[str, float]:
        """Extract wavelet domain features"""
        
        features = {}
        
        try:
            # Multi-level discrete wavelet transform
            wavelet = 'db4'  # Daubechies 4 wavelet
            levels = min(6, int(np.log2(len(signal))))  # Appropriate number of levels
            
            if levels < 1:
                # Signal too short for wavelet analysis
                features.update({f'{prefix}_wavelet_{feat}': 0.0 for feat in [
                    'energy_1', 'energy_2', 'energy_3', 'energy_4', 'energy_5', 'energy_6',
                    'rel_energy_1', 'rel_energy_2', 'rel_energy_3', 'rel_energy_4', 
                    'rel_energy_5', 'rel_energy_6', 'entropy_1', 'entropy_2', 'entropy_3',
                    'entropy_4', 'entropy_5', 'entropy_6'
                ]})
                return features
            
            coeffs = pywt.wavedec(signal, wavelet, level=levels)
            
            # Energy in each level
            total_energy = 0
            level_energies = []
            
            for i, coeff in enumerate(coeffs):
                energy = np.sum(coeff**2)
                level_energies.append(energy)
                total_energy += energy
                features[f'{prefix}_wavelet_energy_{i+1}'] = float(energy)
            
            # Relative energies
            if total_energy > 0:
                for i, energy in enumerate(level_energies):
                    features[f'{prefix}_wavelet_rel_energy_{i+1}'] = float(energy / total_energy)
            else:
                for i in range(len(level_energies)):
                    features[f'{prefix}_wavelet_rel_energy_{i+1}'] = 0.0
            
            # Wavelet entropy for each level
            for i, coeff in enumerate(coeffs):
                # Normalize coefficients
                coeff_norm = np.abs(coeff)
                coeff_norm = coeff_norm / (np.sum(coeff_norm) + 1e-12)
                
                # Calculate entropy
                entropy = -np.sum(coeff_norm * np.log2(coeff_norm + 1e-12))
                features[f'{prefix}_wavelet_entropy_{i+1}'] = float(entropy)
            
            # Pad features to consistent length (6 levels)
            for i in range(len(coeffs), 6):
                features[f'{prefix}_wavelet_energy_{i+1}'] = 0.0
                features[f'{prefix}_wavelet_rel_energy_{i+1}'] = 0.0
                features[f'{prefix}_wavelet_entropy_{i+1}'] = 0.0
                
        except Exception as e:
            logger.warning(f"Error extracting wavelet features for {prefix}: {str(e)}")
            # Return zeros for failed features
            features.update({f'{prefix}_wavelet_{feat}': 0.0 for feat in [
                'energy_1', 'energy_2', 'energy_3', 'energy_4', 'energy_5', 'energy_6',
                'rel_energy_1', 'rel_energy_2', 'rel_energy_3', 'rel_energy_4', 
                'rel_energy_5', 'rel_energy_6', 'entropy_1', 'entropy_2', 'entropy_3',
                'entropy_4', 'entropy_5', 'entropy_6'
            ]})
        
        return features
    
    def _extract_statistical_features(self, signal: np.ndarray, prefix: str) -> Dict[str, float]:
        """Extract advanced statistical features"""
        
        features = {}
        
        try:
            # Higher order moments
            features[f'{prefix}_moment_3'] = float(stats.moment(signal, moment=3))
            features[f'{prefix}_moment_4'] = float(stats.moment(signal, moment=4))
            
            # Distribution shape
            features[f'{prefix}_skewness'] = float(stats.skew(signal))
            features[f'{prefix}_kurtosis'] = float(stats.kurtosis(signal))
            
            # Robust statistics
            features[f'{prefix}_mad'] = float(np.median(np.abs(signal - np.median(signal))))  # Median Absolute Deviation
            features[f'{prefix}_trimmed_mean'] = float(stats.trim_mean(signal, 0.1))  # 10% trimmed mean
            
            # Signal variability measures
            features[f'{prefix}_coeff_variation'] = float(np.std(signal) / (np.mean(signal) + 1e-12))
            features[f'{prefix}_quartile_coeff'] = float(
                (np.percentile(signal, 75) - np.percentile(signal, 25)) / 
                (np.percentile(signal, 75) + np.percentile(signal, 25) + 1e-12)
            )
            
            # Distribution tests
            try:
                # Shapiro-Wilk test for normality (if sample size appropriate)
                if 3 <= len(signal) <= 5000:
                    shapiro_stat, shapiro_p = stats.shapiro(signal)
                    features[f'{prefix}_shapiro_stat'] = float(shapiro_stat)
                    features[f'{prefix}_shapiro_p'] = float(shapiro_p)
                else:
                    features[f'{prefix}_shapiro_stat'] = 0.0
                    features[f'{prefix}_shapiro_p'] = 1.0
            except:
                features[f'{prefix}_shapiro_stat'] = 0.0
                features[f'{prefix}_shapiro_p'] = 1.0
            
            # Sample entropy (complexity measure)
            features[f'{prefix}_sample_entropy'] = float(self._calculate_sample_entropy(signal))
            
            # Hjorth parameters (complexity measures)
            hjorth_params = self._calculate_hjorth_parameters(signal)
            features[f'{prefix}_hjorth_activity'] = float(hjorth_params['activity'])
            features[f'{prefix}_hjorth_mobility'] = float(hjorth_params['mobility'])
            features[f'{prefix}_hjorth_complexity'] = float(hjorth_params['complexity'])
            
        except Exception as e:
            logger.warning(f"Error extracting statistical features for {prefix}: {str(e)}")
            # Return zeros for failed features
            features.update({f'{prefix}_{feat}': 0.0 for feat in [
                'moment_3', 'moment_4', 'skewness', 'kurtosis', 'mad', 'trimmed_mean',
                'coeff_variation', 'quartile_coeff', 'shapiro_stat', 'shapiro_p',
                'sample_entropy', 'hjorth_activity', 'hjorth_mobility', 'hjorth_complexity'
            ]})
        
        return features
    
    def _calculate_sample_entropy(self, signal: np.ndarray, m: int = 2, r: float = None) -> float:
        """Calculate sample entropy of signal"""
        try:
            if r is None:
                r = 0.2 * np.std(signal)
            
            N = len(signal)
            if N < 10 or r == 0:
                return 0.0
            
            def _maxdist(xi, xj, m):
                return max([abs(ua - va) for ua, va in zip(xi, xj)])
            
            def _phi(m):
                patterns = np.array([signal[i:i + m] for i in range(N - m + 1)])
                C = np.zeros(N - m + 1)
                for i in range(N - m + 1):
                    template_i = patterns[i]
                    for j in range(N - m + 1):
                        if _maxdist(template_i, patterns[j], m) <= r:
                            C[i] += 1.0
                
                phi = (1.0 / (N - m + 1)) * sum(np.log(C / (N - m + 1.0)))
                return phi
            
            return _phi(m) - _phi(m + 1)
            
        except:
            return 0.0
    
    def _calculate_hjorth_parameters(self, signal: np.ndarray) -> Dict[str, float]:
        """Calculate Hjorth parameters (Activity, Mobility, Complexity)"""
        try:
            # First derivative
            d1 = np.diff(signal)
            # Second derivative  
            d2 = np.diff(d1)
            
            # Variances
            var_signal = np.var(signal)
            var_d1 = np.var(d1)
            var_d2 = np.var(d2)
            
            # Hjorth parameters
            activity = var_signal
            mobility = np.sqrt(var_d1 / (var_signal + 1e-12))
            complexity = np.sqrt(var_d2 / (var_d1 + 1e-12)) / (mobility + 1e-12)
            
            return {
                'activity': activity,
                'mobility': mobility,
                'complexity': complexity
            }
        except:
            return {'activity': 0.0, 'mobility': 0.0, 'complexity': 0.0}
    
    def _extract_morphological_features(self, signal: np.ndarray, prefix: str) -> Dict[str, float]:
        """Extract morphological and shape-based features"""
        
        features = {}
        
        try:
            # Peak and valley detection
            peaks, peak_properties = signal.find_peaks(signal, distance=10, prominence=np.std(signal)*0.5)
            valleys, valley_properties = signal.find_peaks(-signal, distance=10, prominence=np.std(signal)*0.5)
            
            features[f'{prefix}_num_peaks'] = len(peaks)
            features[f'{prefix}_num_valleys'] = len(valleys)
            
            if len(peaks) > 0:
                peak_values = signal[peaks]
                features[f'{prefix}_peak_mean'] = float(np.mean(peak_values))
                features[f'{prefix}_peak_std'] = float(np.std(peak_values))
                features[f'{prefix}_peak_max'] = float(np.max(peak_values))
                
                # Peak-to-peak intervals
                if len(peaks) > 1:
                    peak_intervals = np.diff(peaks)
                    features[f'{prefix}_peak_interval_mean'] = float(np.mean(peak_intervals))
                    features[f'{prefix}_peak_interval_std'] = float(np.std(peak_intervals))
                    features[f'{prefix}_peak_interval_cv'] = float(np.std(peak_intervals) / (np.mean(peak_intervals) + 1e-12))
                else:
                    features[f'{prefix}_peak_interval_mean'] = 0.0
                    features[f'{prefix}_peak_interval_std'] = 0.0
                    features[f'{prefix}_peak_interval_cv'] = 0.0
            else:
                features[f'{prefix}_peak_mean'] = 0.0
                features[f'{prefix}_peak_std'] = 0.0
                features[f'{prefix}_peak_max'] = 0.0
                features[f'{prefix}_peak_interval_mean'] = 0.0
                features[f'{prefix}_peak_interval_std'] = 0.0
                features[f'{prefix}_peak_interval_cv'] = 0.0
            
            if len(valleys) > 0:
                valley_values = signal[valleys]
                features[f'{prefix}_valley_mean'] = float(np.mean(valley_values))
                features[f'{prefix}_valley_std'] = float(np.std(valley_values))
                features[f'{prefix}_valley_min'] = float(np.min(valley_values))
            else:
                features[f'{prefix}_valley_mean'] = 0.0
                features[f'{prefix}_valley_std'] = 0.0
                features[f'{prefix}_valley_min'] = 0.0
            
            # Rise and fall times
            if len(peaks) > 0 and len(valleys) > 0:
                # Simple rise/fall time estimation
                rise_times = []
                fall_times = []
                
                for peak_idx in peaks:
                    # Find preceding valley
                    preceding_valleys = valleys[valleys < peak_idx]
                    if len(preceding_valleys) > 0:
                        valley_idx = preceding_valleys[-1]
                        rise_times.append(peak_idx - valley_idx)
                    
                    # Find following valley
                    following_valleys = valleys[valleys > peak_idx]
                    if len(following_valleys) > 0:
                        valley_idx = following_valleys[0]
                        fall_times.append(valley_idx - peak_idx)
                
                if rise_times:
                    features[f'{prefix}_rise_time_mean'] = float(np.mean(rise_times))
                    features[f'{prefix}_rise_time_std'] = float(np.std(rise_times))
                else:
                    features[f'{prefix}_rise_time_mean'] = 0.0
                    features[f'{prefix}_rise_time_std'] = 0.0
                
                if fall_times:
                    features[f'{prefix}_fall_time_mean'] = float(np.mean(fall_times))
                    features[f'{prefix}_fall_time_std'] = float(np.std(fall_times))
                else:
                    features[f'{prefix}_fall_time_mean'] = 0.0
                    features[f'{prefix}_fall_time_std'] = 0.0
            else:
                features[f'{prefix}_rise_time_mean'] = 0.0
                features[f'{prefix}_rise_time_std'] = 0.0
                features[f'{prefix}_fall_time_mean'] = 0.0
                features[f'{prefix}_fall_time_std'] = 0.0
            
            # Signal slope features
            slopes = np.diff(signal)
            features[f'{prefix}_slope_mean'] = float(np.mean(slopes))
            features[f'{prefix}_slope_std'] = float(np.std(slopes))
            
            positive_slopes = slopes[slopes > 0]
            negative_slopes = slopes[slopes < 0]
            
            features[f'{prefix}_positive_slope_mean'] = float(np.mean(positive_slopes)) if len(positive_slopes) > 0 else 0.0
            features[f'{prefix}_negative_slope_mean'] = float(np.mean(negative_slopes)) if len(negative_slopes) > 0 else 0.0
            
            # Area under curve
            features[f'{prefix}_area_under_curve'] = float(np.trapz(np.abs(signal)))
            features[f'{prefix}_area_positive'] = float(np.trapz(np.maximum(signal, 0)))
            features[f'{prefix}_area_negative'] = float(np.trapz(np.maximum(-signal, 0)))
            
        except Exception as e:
            logger.warning(f"Error extracting morphological features for {prefix}: {str(e)}")
            # Return zeros for failed features
            features.update({f'{prefix}_{feat}': 0.0 for feat in [
                'num_peaks', 'num_valleys', 'peak_mean', 'peak_std', 'peak_max',
                'peak_interval_mean', 'peak_interval_std', 'peak_interval_cv',
                'valley_mean', 'valley_std', 'valley_min', 'rise_time_mean', 'rise_time_std',
                'fall_time_mean', 'fall_time_std', 'slope_mean', 'slope_std',
                'positive_slope_mean', 'negative_slope_mean', 'area_under_curve',
                'area_positive', 'area_negative'
            ]})
        
        return features
    
    def _extract_nonlinear_features(self, signal: np.ndarray, prefix: str) -> Dict[str, float]:
        """Extract nonlinear dynamics features"""
        
        features = {}
        
        try:
            # Approximate entropy
            features[f'{prefix}_approx_entropy'] = float(self._approximate_entropy(signal))
            
            # Detrended fluctuation analysis (simplified)
            features[f'{prefix}_dfa_alpha'] = float(self._detrended_fluctuation_analysis(signal))
            
            # Largest Lyapunov exponent (simplified estimation)
            features[f'{prefix}_lyapunov_exp'] = float(self._largest_lyapunov_exponent(signal))
            
            # Correlation dimension (simplified)
            features[f'{prefix}_correlation_dim'] = float(self._correlation_dimension(signal))
            
            # Recurrence quantification analysis features
            rqa_features = self._recurrence_quantification_analysis(signal)
            for key, value in rqa_features.items():
                features[f'{prefix}_rqa_{key}'] = float(value)
                
        except Exception as e:
            logger.warning(f"Error extracting nonlinear features for {prefix}: {str(e)}")
            # Return zeros for failed features
            features.update({f'{prefix}_{feat}': 0.0 for feat in [
                'approx_entropy', 'dfa_alpha', 'lyapunov_exp', 'correlation_dim',
                'rqa_recurrence_rate', 'rqa_determinism', 'rqa_average_diagonal_length'
            ]})
        
        return features
    
    def _approximate_entropy(self, signal: np.ndarray, m: int = 2, r: float = None) -> float:
        """Calculate approximate entropy"""
        try:
            if r is None:
                r = 0.2 * np.std(signal)
            
            N = len(signal)
            if N < 10:
                return 0.0
            
            def _maxdist(x_i, x_j, m):
                return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
            
            def _phi(m):
                patterns = [signal[i:i+m] for i in range(N-m+1)]
                C = [0] * (N-m+1)
                
                for i in range(N-m+1):
                    template_i = patterns[i]
                    for j in range(N-m+1):
                        if _maxdist(template_i, patterns[j], m) <= r:
                            C[i] += 1
                
                phi = sum([np.log(c/(N-m+1.0)) for c in C]) / (N-m+1.0)
                return phi
            
            return _phi(m) - _phi(m+1)
            
        except:
            return 0.0
    
    def _detrended_fluctuation_analysis(self, signal: np.ndarray) -> float:
        """Simplified DFA alpha calculation"""
        try:
            N = len(signal)
            if N < 50:
                return 0.0
            
            # Integrate the signal
            y = np.cumsum(signal - np.mean(signal))
            
            # Define scales
            scales = np.logspace(1, np.log10(N//4), 10, dtype=int)
            fluctuations = []
            
            for scale in scales:
                # Divide into non-overlapping segments
                n_segments = N // scale
                
                if n_segments < 2:
                    continue
                    
                # Detrend each segment
                mse = 0
                for i in range(n_segments):
                    start = i * scale
                    end = start + scale
                    segment = y[start:end]
                    
                    # Linear detrending
                    x = np.arange(len(segment))
                    coeffs = np.polyfit(x, segment, 1)
                    trend = np.polyval(coeffs, x)
                    
                    mse += np.mean((segment - trend)**2)
                
                fluctuations.append(np.sqrt(mse / n_segments))
            
            if len(fluctuations) < 3:
                return 0.0
            
            # Fit power law: F(n) ~ n^alpha
            log_scales = np.log(scales[:len(fluctuations)])
            log_fluct = np.log(fluctuations)
            
            alpha = np.polyfit(log_scales, log_fluct, 1)[0]
            return alpha
            
        except:
            return 0.0
    
    def _largest_lyapunov_exponent(self, signal: np.ndarray) -> float:
        """Simplified largest Lyapunov exponent estimation"""
        try:
            # This is a very simplified estimation
            # In practice, proper reconstruction of phase space is needed
            
            # Calculate successive differences
            diffs = np.abs(np.diff(signal))
            
            if len(diffs) < 10:
                return 0.0
            
            # Estimate divergence rate
            log_diffs = np.log(diffs + 1e-12)
            
            # Linear fit to estimate exponential growth rate
            x = np.arange(len(log_diffs))
            slope = np.polyfit(x, log_diffs, 1)[0]
            
            return slope
            
        except:
            return 0.0
    
    def _correlation_dimension(self, signal: np.ndarray) -> float:
        """Simplified correlation dimension calculation"""
        try:
            # This is a very simplified version
            # Proper calculation requires phase space reconstruction
            
            N = len(signal)
            if N < 100:
                return 0.0
            
            # Embed in 2D phase space (simplified)
            embedded = np.column_stack([signal[:-1], signal[1:]])
            
            # Calculate correlation sum for different radii
            radii = np.logspace(-2, 0, 10) * np.std(signal)
            correlations = []
            
            for r in radii:
                count = 0
                for i in range(len(embedded)):
                    distances = np.linalg.norm(embedded - embedded[i], axis=1)
                    count += np.sum(distances < r) - 1  # Exclude self
                
                correlations.append(count / (N * (N - 1)))
            
            # Fit power law
            log_radii = np.log(radii)
            log_corr = np.log(np.array(correlations) + 1e-12)
            
            # Find linear region and calculate slope
            dim = np.polyfit(log_radii[2:8], log_corr[2:8], 1)[0]
            
            return max(0, dim)
            
        except:
            return 0.0
    
    def _recurrence_quantification_analysis(self, signal: np.ndarray) -> Dict[str, float]:
        """Simplified recurrence quantification analysis"""
        try:
            N = len(signal)
            if N < 50:
                return {
                    'recurrence_rate': 0.0,
                    'determinism': 0.0,
                    'average_diagonal_length': 0.0
                }
            
            # Create recurrence matrix (simplified)
            threshold = 0.1 * np.std(signal)
            recurrence_matrix = np.zeros((N, N))
            
            for i in range(N):
                for j in range(N):
                    if abs(signal[i] - signal[j]) < threshold:
                        recurrence_matrix[i, j] = 1
            
            # Calculate RQA measures
            recurrence_rate = np.mean(recurrence_matrix)
            
            # Determinism (percentage of recurrence points forming diagonal lines)
            diagonal_lines = 0
            min_line_length = 2
            
            for i in range(N - min_line_length + 1):
                for j in range(N - min_line_length + 1):
                    if recurrence_matrix[i, j] == 1:
                        # Check for diagonal line
                        line_length = 1
                        k = 1
                        while (i + k < N and j + k < N and 
                               recurrence_matrix[i + k, j + k] == 1):
                            line_length += 1
                            k += 1
                        
                        if line_length >= min_line_length:
                            diagonal_lines += line_length
            
            total_recurrence_points = np.sum(recurrence_matrix)
            determinism = diagonal_lines / (total_recurrence_points + 1e-12)
            
            # Average diagonal line length
            avg_diagonal_length = diagonal_lines / max(1, total_recurrence_points - diagonal_lines + 1)
            
            return {
                'recurrence_rate': recurrence_rate,
                'determinism': determinism,
                'average_diagonal_length': avg_diagonal_length
            }
            
        except:
            return {
                'recurrence_rate': 0.0,
                'determinism': 0.0,
                'average_diagonal_length': 0.0
            }
    
    def _extract_cross_sensor_features(self, sensor_data: pd.DataFrame, sensor_group: str) -> Dict[str, float]:
        """Extract features from relationships between sensors"""
        
        features = {}
        
        try:
            if len(sensor_data.columns) < 2:
                return features
            
            # Correlation features
            corr_matrix = sensor_data.corr()
            
            # Extract upper triangular correlations
            correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if not np.isnan(corr_val):
                        correlations.append(abs(corr_val))
                        col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                        features[f'{sensor_group}_corr_{col1}_{col2}'] = float(corr_val)
            
            if correlations:
                features[f'{sensor_group}_mean_abs_correlation'] = float(np.mean(correlations))
                features[f'{sensor_group}_max_abs_correlation'] = float(np.max(correlations))
                features[f'{sensor_group}_std_correlations'] = float(np.std(correlations))
            
            # Principal component analysis
            if len(sensor_data.columns) >= 2:
                try:
                    pca = PCA(n_components=min(3, len(sensor_data.columns)))
                    pca_result = pca.fit_transform(sensor_data.fillna(0))
                    
                    # Explained variance ratios
                    for i, var_ratio in enumerate(pca.explained_variance_ratio_):
                        features[f'{sensor_group}_pca_var_ratio_{i+1}'] = float(var_ratio)
                    
                    # First principal component features
                    pc1 = pca_result[:, 0]
                    features[f'{sensor_group}_pc1_mean'] = float(np.mean(pc1))
                    features[f'{sensor_group}_pc1_std'] = float(np.std(pc1))
                    
                except Exception as e:
                    logger.warning(f"PCA failed for {sensor_group}: {str(e)}")
            
            # Magnitude and phase relationships (for 3D sensors)
            if len(sensor_data.columns) == 3:
                # Calculate resultant magnitude
                magnitude = np.sqrt(np.sum(sensor_data**2, axis=1))
                features[f'{sensor_group}_magnitude_mean'] = float(np.mean(magnitude))
                features[f'{sensor_group}_magnitude_std'] = float(np.std(magnitude))
                features[f'{sensor_group}_magnitude_max'] = float(np.max(magnitude))
                
                # Calculate angles between axes
                col_names = sensor_data.columns.tolist()
                if len(col_names) == 3:
                    x_data = sensor_data.iloc[:, 0].values
                    y_data = sensor_data.iloc[:, 1].values
                    z_data = sensor_data.iloc[:, 2].values
                    
                    # Angle with respect to vertical (assuming z is vertical)
                    vertical_angles = np.arctan2(np.sqrt(x_data**2 + y_data**2), z_data) * 180 / np.pi
                    features[f'{sensor_group}_vertical_angle_mean'] = float(np.mean(vertical_angles))
                    features[f'{sensor_group}_vertical_angle_std'] = float(np.std(vertical_angles))
                    
        except Exception as e:
            logger.warning(f"Error extracting cross-sensor features for {sensor_group}: {str(e)}")
        
        return features
    
    def _extract_gait_specific_features(self, data: pd.DataFrame, sampling_rate: int = 50) -> Dict[str, float]:
        """Extract gait-specific biomechanical features"""
        
        features = {}
        
        try:
            # Get accelerometer data
            accel_cols = [col for col in data.columns if 'accel' in col.lower()]
            if len(accel_cols) < 2:
                return features
            
            # Calculate resultant acceleration
            accel_data = data[accel_cols].fillna(method='ffill')
            resultant_accel = np.sqrt(np.sum(accel_data**2, axis=1))
            
            # Step detection using resultant acceleration
            steps = self._detect_steps(resultant_accel, sampling_rate)
            
            if len(steps) > 1:
                # Step timing features
                step_times = np.array(steps) / sampling_rate
                step_intervals = np.diff(step_times)
                
                features['gait_step_frequency'] = float(len(steps) / (len(data) / sampling_rate))
                features['gait_cadence'] = float(features['gait_step_frequency'] * 60)  # steps per minute
                
                features['gait_step_time_mean'] = float(np.mean(step_intervals))
                features['gait_step_time_std'] = float(np.std(step_intervals))
                features['gait_step_time_cv'] = float(np.std(step_intervals) / (np.mean(step_intervals) + 1e-12))
                
                # Gait regularity and symmetry
                if len(step_intervals) > 4:
                    # Autocorrelation of step intervals for regularity
                    autocorr = np.correlate(step_intervals, step_intervals, mode='full')
                    autocorr = autocorr[autocorr.size // 2:]
                    autocorr = autocorr / autocorr[0]
                    
                    if len(autocorr) > 2:
                        features['gait_regularity'] = float(autocorr[1])  # First lag autocorrelation
                    
                    # Step-to-step variability
                    step_diff = np.abs(np.diff(step_intervals))
                    features['gait_step_variability'] = float(np.mean(step_diff))
                    
                    # Asymmetry (alternating step pattern)
                    even_steps = step_intervals[::2]
                    odd_steps = step_intervals[1::2]
                    min_len = min(len(even_steps), len(odd_steps))
                    
                    if min_len > 2:
                        asymmetry = np.abs(np.mean(even_steps[:min_len]) - np.mean(odd_steps[:min_len]))
                        features['gait_asymmetry'] = float(asymmetry)
                    else:
                        features['gait_asymmetry'] = 0.0
                else:
                    features['gait_regularity'] = 0.0
                    features['gait_step_variability'] = 0.0
                    features['gait_asymmetry'] = 0.0
            else:
                # No steps detected
                features['gait_step_frequency'] = 0.0
                features['gait_cadence'] = 0.0
                features['gait_step_time_mean'] = 0.0
                features['gait_step_time_std'] = 0.0
                features['gait_step_time_cv'] = 0.0
                features['gait_regularity'] = 0.0
                features['gait_step_variability'] = 0.0
                features['gait_asymmetry'] = 0.0
            
            # Gait quality measures
            # Smoothness (jerk-based measure)
            if len(accel_cols) >= 1:
                accel_signal = data[accel_cols[0]].fillna(method='ffill').values
                jerk = np.diff(accel_signal, n=3)  # Third derivative
                features['gait_smoothness'] = float(-np.mean(jerk**2))  # Negative because lower jerk = smoother
            
            # Harmonic ratio (gait quality measure)
            if len(accel_cols) >= 3:
                for i, col in enumerate(accel_cols[:3]):
                    axis = col.split('_')[-1] if '_' in col else str(i)
                    hr = self._calculate_harmonic_ratio(data[col].fillna(method='ffill').values)
                    features[f'gait_harmonic_ratio_{axis}'] = float(hr)
            
            # Postural transitions (sit-to-stand, stand-to-sit detection)
            transitions = self._detect_postural_transitions(resultant_accel, sampling_rate)
            features['gait_num_transitions'] = len(transitions)
            features['gait_transition_rate'] = float(len(transitions) / (len(data) / sampling_rate) * 60)  # per minute
            
        except Exception as e:
            logger.warning(f"Error extracting gait-specific features: {str(e)}")
            # Return zeros for failed features
            gait_feature_names = [
                'gait_step_frequency', 'gait_cadence', 'gait_step_time_mean', 'gait_step_time_std',
                'gait_step_time_cv', 'gait_regularity', 'gait_step_variability', 'gait_asymmetry',
                'gait_smoothness', 'gait_harmonic_ratio_x', 'gait_harmonic_ratio_y', 'gait_harmonic_ratio_z',
                'gait_num_transitions', 'gait_transition_rate'
            ]
            for feature_name in gait_feature_names:
                if feature_name not in features:
                    features[feature_name] = 0.0
        
        return features
    
    def _detect_steps(self, resultant_accel: np.ndarray, sampling_rate: int = 50) -> List[int]:
        """Simple step detection algorithm"""
        try:
            # Smooth the signal
            from scipy.ndimage import gaussian_filter1d
            smoothed = gaussian_filter1d(resultant_accel, sigma=2)
            
            # Find peaks (potential steps)
            min_distance = int(0.4 * sampling_rate)  # Minimum 0.4 seconds between steps
            min_height = np.mean(smoothed) + 0.5 * np.std(smoothed)
            
            steps, _ = signal.find_peaks(smoothed, distance=min_distance, height=min_height)
            
            return steps.tolist()
        except:
            return []
    
    def _calculate_harmonic_ratio(self, signal_data: np.ndarray) -> float:
        """Calculate harmonic ratio (gait quality measure)"""
        try:
            # Remove DC component
            signal_data = signal_data - np.mean(signal_data)
            
            if len(signal_data) < 100:
                return 0.0
            
            # Apply FFT
            fft_signal = np.fft.fft(signal_data)
            frequencies = np.fft.fftfreq(len(signal_data), 1/50)  # Assume 50 Hz
            
            # Keep positive frequencies only
            positive_mask = frequencies > 0
            fft_positive = fft_signal[positive_mask]
            freq_positive = frequencies[positive_mask]
            
            # Power spectral density
            psd = np.abs(fft_positive)**2
            
            # Find fundamental frequency (in gait frequency range)
            gait_mask = (freq_positive >= 0.5) & (freq_positive <= 3.0)
            
            if not np.any(gait_mask):
                return 0.0
            
            gait_freqs = freq_positive[gait_mask]
            gait_psd = psd[gait_mask]
            
            # Find fundamental frequency
            fund_freq_idx = np.argmax(gait_psd)
            fund_freq = gait_freqs[fund_freq_idx]
            
            # Calculate harmonic ratio
            # Sum of first 10 harmonics vs sum of remaining frequencies
            harmonic_power = 0
            for h in range(1, 11):  # First 10 harmonics
                harmonic_freq = fund_freq * h
                # Find closest frequency bin
                closest_idx = np.argmin(np.abs(freq_positive - harmonic_freq))
                
                if closest_idx < len(psd):
                    harmonic_power += psd[closest_idx]
            
            total_power = np.sum(psd)
            
            if total_power > 0:
                harmonic_ratio = harmonic_power / total_power
            else:
                harmonic_ratio = 0.0
            
            return harmonic_ratio
            
        except Exception as e:
            logger.warning(f"Error calculating harmonic ratio: {str(e)}")
            return 0.0
    
    def _detect_postural_transitions(self, resultant_accel: np.ndarray, sampling_rate: int = 50) -> List[int]:
        """Detect postural transitions (sit-to-stand, stand-to-sit)"""
        try:
            # Smooth the signal
            from scipy.ndimage import gaussian_filter1d
            smoothed = gaussian_filter1d(resultant_accel, sigma=sampling_rate//10)
            
            # Calculate derivative to find rapid changes
            derivative = np.diff(smoothed)
            
            # Find large positive and negative changes
            threshold = 3 * np.std(derivative)
            
            # Transitions are characterized by large accelerations followed by decelerations
            transitions = []
            min_interval = int(2 * sampling_rate)  # Minimum 2 seconds between transitions
            
            for i in range(len(derivative) - min_interval):
                if abs(derivative[i]) > threshold:
                    # Check if this is isolated from previous detections
                    if not transitions or (i - transitions[-1]) > min_interval:
                        transitions.append(i)
            
            return transitions
            
        except Exception as e:
            logger.warning(f"Error detecting postural transitions: {str(e)}")
            return []
    
    def select_optimal_features(self, 
                              feature_df: pd.DataFrame, 
                              target: pd.Series, 
                              k: int = 100,
                              method: str = 'f_classif') -> Tuple[pd.DataFrame, List[str]]:
        """
        Select optimal features using statistical tests
        
        Args:
            feature_df: DataFrame with features
            target: Target variable for selection
            k: Number of features to select
            method: Feature selection method
            
        Returns:
            Tuple of (selected_features_df, selected_feature_names)
        """
        
        try:
            if len(feature_df) == 0:
                return feature_df, []
            
            # Remove features with zero variance
            non_zero_var_features = feature_df.loc[:, feature_df.var() != 0]
            
            if len(non_zero_var_features.columns) == 0:
                logger.warning("All features have zero variance")
                return feature_df, list(feature_df.columns)
            
            # Handle missing values
            feature_data = non_zero_var_features.fillna(0)
            
            # Select top k features
            k = min(k, len(feature_data.columns))
            
            if method == 'f_classif':
                selector = SelectKBest(score_func=f_classif, k=k)
            else:
                selector = SelectKBest(k=k)
            
            selected_features = selector.fit_transform(feature_data, target)
            
            # Get selected feature names
            selected_mask = selector.get_support()
            selected_feature_names = feature_data.columns[selected_mask].tolist()
            
            selected_feature_df = pd.DataFrame(
                selected_features, 
                columns=selected_feature_names,
                index=feature_data.index
            )
            
            logger.info(f"Selected {len(selected_feature_names)} optimal features out of {len(feature_data.columns)}")
            
            return selected_feature_df, selected_feature_names
            
        except Exception as e:
            logger.error(f"Error in feature selection: {str(e)}")
            return feature_df, list(feature_df.columns)

"""
Preprocessing modules for FE-AI system
Provides signal processing, feature extraction, normalization, and segmentation
"""

from .signal_processor import SignalProcessor
from .feature_extractor import FeatureExtractor
from .normalizer import DataNormalizer
from .segmentation import SignalSegmenter

__all__ = ['SignalProcessor', 'FeatureExtractor', 'DataNormalizer', 'SignalSegmenter']
