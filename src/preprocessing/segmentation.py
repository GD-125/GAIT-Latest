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
            from scipy.signal import savgol_filter
            smoothed_activity = savgol_filter(activity_signal, window_length=min(51, len(activity_signal)//2*2+1), polyorder=3)
            
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
            accel_data = data[available_accel_cols].ffill()
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