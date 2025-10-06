# File: src/database/models.py
# Database models and schemas

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum

class UserRole(Enum):
    ADMIN = "admin"
    DOCTOR = "doctor"
    RESEARCHER = "researcher"
    VIEWER = "viewer"

class AnalysisStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class User:
    """User model"""
    username: str
    email: Optional[str] = None
    password_hash: str = ""
    role: UserRole = UserRole.VIEWER
    created_at: datetime = None
    last_login: Optional[datetime] = None
    is_active: bool = True
    profile: Dict[str, Any] = None
    preferences: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.profile is None:
            self.profile = {}
        if self.preferences is None:
            self.preferences = {}

@dataclass
class Analysis:
    """Analysis result model"""
    analysis_id: str
    user_id: str
    subject_id: Optional[str] = None
    timestamp: datetime = None
    data_info: Dict[str, Any] = None
    preprocessing: Dict[str, Any] = None
    gait_detection: Dict[str, Any] = None
    disease_classification: Dict[str, Any] = None
    explainability: Dict[str, Any] = None
    performance_metrics: Dict[str, Any] = None
    status: AnalysisStatus = AnalysisStatus.PENDING
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.data_info is None:
            self.data_info = {}
        if self.preprocessing is None:
            self.preprocessing = {}
        if self.gait_detection is None:
            self.gait_detection = {}
        if self.disease_classification is None:
            self.disease_classification = {}
        if self.explainability is None:
            self.explainability = {}
        if self.performance_metrics is None:
            self.performance_metrics = {}

@dataclass
class Model:
    """ML Model metadata"""
    model_name: str
    version: str
    model_type: str
    architecture: Dict[str, Any] = None
    parameters: Dict[str, Any] = None
    performance_metrics: Dict[str, Any] = None
    training_info: Dict[str, Any] = None
    federated_learning: Dict[str, Any] = None
    file_path: Optional[str] = None
    checksum: Optional[str] = None
    created_at: datetime = None
    created_by: Optional[str] = None
    is_active: bool = True
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.architecture is None:
            self.architecture = {}
        if self.parameters is None:
            self.parameters = {}
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if self.training_info is None:
            self.training_info = {}
        if self.federated_learning is None:
            self.federated_learning = {}

@dataclass
class SystemLog:
    """System log entry"""
    timestamp: datetime
    level: LogLevel
    category: str
    event: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: Dict[str, Any] = None
    security_relevant: bool = False

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.details is None:
            self.details = {}

@dataclass
class Subject:
    """Subject (patient/participant) metadata"""
    subject_id: str
    demographics: Dict[str, Any] = None
    medical_history: Dict[str, Any] = None
    notes: Optional[str] = None
    created_at: datetime = None
    created_by: Optional[str] = None
    is_active: bool = True

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.demographics is None:
            self.demographics = {}
        if self.medical_history is None:
            self.medical_history = {}

@dataclass
class Dataset:
    """Dataset metadata"""
    dataset_id: str
    name: str
    description: Optional[str] = None
    created_at: datetime = None
    created_by: Optional[str] = None
    subjects: List[str] = None
    data_files: List[str] = None
    metadata: Dict[str, Any] = None
    is_active: bool = True

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.subjects is None:
            self.subjects = []
        if self.data_files is None:
            self.data_files = []
        if self.metadata is None:
            self.metadata = {}
