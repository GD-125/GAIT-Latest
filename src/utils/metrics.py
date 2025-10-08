# File: src/utils/metrics.py
# Performance metrics and monitoring

import time
import psutil
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """Track and calculate performance metrics"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.metrics = {}
    
    def start(self):
        """Start metrics collection"""
        self.start_time = time.time()
        self.metrics['start_timestamp'] = datetime.now().isoformat()
    
    def stop(self):
        """Stop metrics collection"""
        self.end_time = time.time()
        self.metrics['end_timestamp'] = datetime.now().isoformat()
        self.metrics['duration_seconds'] = self.end_time - self.start_time
    
    def add_metric(self, name: str, value: Any):
        """Add a custom metric"""
        self.metrics[name] = value
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
            }
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {}
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics"""
        return self.metrics.copy()
    
    def reset(self):
        """Reset all metrics"""
        self.start_time = None
        self.end_time = None
        self.metrics = {}