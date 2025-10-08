# File: src/database/mongo_handler.py
# MongoDB operations and connection management

import pymongo
from pymongo import MongoClient
from motor.motor_asyncio import AsyncIOMotorClient
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import hashlib
import json
from bson import ObjectId
import asyncio

from src.utils.config import config

logger = logging.getLogger(__name__)

class MongoHandler:
    """MongoDB operations handler with connection pooling and security"""
    
    def __init__(self):
        self.client = None
        self.async_client = None
        self.db = None
        self.async_db = None
        self.connection_params = self._get_connection_params()
        
    def _get_connection_params(self) -> Dict[str, Any]:
        """Get MongoDB connection parameters from config"""
        
        return {
            'host': config.get('database.mongodb.host', 'localhost'),
            'port': config.get('database.mongodb.port', 27017),
            'username': config.get('database.mongodb.username'),
            'password': config.get('database.mongodb.password'),
            'database': config.get('database.mongodb.database', 'fe_ai_system'),
            'auth_source': config.get('database.mongodb.auth_database', 'admin'),
            'ssl': config.get('database.mongodb.ssl_enabled', False),
            'replica_set': config.get('database.mongodb.replica_set'),
            'max_pool_size': 100,
            'min_pool_size': 10,
            'max_idle_time_ms': 30000,
            'server_selection_timeout_ms': 5000,
            'connect_timeout_ms': 10000,
            'socket_timeout_ms': 30000
        }
    
    def connect(self) -> bool:
        """Establish MongoDB connection"""
        
        try:
            # Build connection URI
            uri_parts = []
            
            if self.connection_params['username'] and self.connection_params['password']:
                uri_parts.append(f"mongodb://{self.connection_params['username']}:"
                               f"{self.connection_params['password']}@")
            else:
                uri_parts.append("mongodb://")
            uri_parts.append(f"{self.connection_params['host']}:{self.connection_params['port']}")
            uri_parts.append(f"/{self.connection_params['database']}")
            
            # Add connection options
            options = []
            if self.connection_params['auth_source']:
                options.append(f"authSource={self.connection_params['auth_source']}")
            if self.connection_params['ssl']:
                options.append("ssl=true")
            if self.connection_params['replica_set']:
                options.append(f"replicaSet={self.connection_params['replica_set']}")
            
            options.extend([
                f"maxPoolSize={self.connection_params['max_pool_size']}",
                f"minPoolSize={self.connection_params['min_pool_size']}",
                f"maxIdleTimeMS={self.connection_params['max_idle_time_ms']}",
                f"serverSelectionTimeoutMS={self.connection_params['server_selection_timeout_ms']}",
                f"connectTimeoutMS={self.connection_params['connect_timeout_ms']}",
                f"socketTimeoutMS={self.connection_params['socket_timeout_ms']}"
            ])
            
            if options:
                uri_parts.append("?" + "&".join(options))
            
            connection_uri = "".join(uri_parts)
            
            # Create synchronous client
            self.client = MongoClient(connection_uri)
            self.db = self.client[self.connection_params['database']]
            
            # Create asynchronous client
            self.async_client = AsyncIOMotorClient(connection_uri)
            self.async_db = self.async_client[self.connection_params['database']]
            
            # Test connection
            self.client.admin.command('ismaster')
            
            logger.info("MongoDB connection established successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            return False
    
    def disconnect(self):
        """Close MongoDB connections"""
        try:
            if self.client:
                self.client.close()
            if self.async_client:
                self.async_client.close()
            logger.info("MongoDB connections closed")
        except Exception as e:
            logger.error(f"Error closing MongoDB connections: {str(e)}")
    
    def create_user(self, user_data: Dict[str, Any]) -> Optional[str]:
        """Create a new user"""
        try:
            # Hash password
            password_hash = hashlib.pbkdf2_hmac(
                'sha256',
                user_data['password'].encode('utf-8'),
                user_data['username'].encode('utf-8'),
                100000
            )
            
            user_doc = {
                'username': user_data['username'],
                'email': user_data.get('email'),
                'password_hash': password_hash.hex(),
                'role': user_data.get('role', 'viewer'),
                'created_at': datetime.utcnow(),
                'last_login': None,
                'is_active': True,
                'profile': user_data.get('profile', {}),
                'preferences': user_data.get('preferences', {})
            }
            
            result = self.db.users.insert_one(user_doc)
            logger.info(f"User created: {user_data['username']}")
            return str(result.inserted_id)
            
        except pymongo.errors.DuplicateKeyError:
            logger.error(f"User already exists: {user_data['username']}")
            return None
        except Exception as e:
            logger.error(f"Error creating user: {str(e)}")
            return None
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user credentials"""
        try:
            user = self.db.users.find_one({'username': username, 'is_active': True})
            
            if not user:
                return None
            
            # Verify password
            password_hash = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                username.encode('utf-8'),
                100000
            )
            
            if password_hash.hex() == user['password_hash']:
                # Update last login
                self.db.users.update_one(
                    {'_id': user['_id']},
                    {'$set': {'last_login': datetime.utcnow()}}
                )
                
                # Remove sensitive data
                user.pop('password_hash', None)
                user['_id'] = str(user['_id'])
                
                return user
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error authenticating user: {str(e)}")
            return None
    
    def save_analysis(self, analysis_data: Dict[str, Any]) -> Optional[str]:
        """Save analysis results to database"""
        try:
            analysis_doc = {
                'analysis_id': analysis_data.get('analysis_id'),
                'user_id': analysis_data.get('user_id'),
                'subject_id': analysis_data.get('subject_id'),
                'timestamp': datetime.utcnow(),
                'data_info': {
                    'filename': analysis_data.get('filename'),
                    'file_size': analysis_data.get('file_size'),
                    'duration': analysis_data.get('duration'),
                    'sampling_rate': analysis_data.get('sampling_rate')
                },
                'preprocessing': analysis_data.get('preprocessing_params', {}),
                'gait_detection': analysis_data.get('gait_results', {}),
                'disease_classification': analysis_data.get('disease_results', {}),
                'explainability': analysis_data.get('explainability_results', {}),
                'performance_metrics': analysis_data.get('performance_metrics', {}),
                'status': analysis_data.get('status', 'completed')
            }
            
            result = self.db.analyses.insert_one(analysis_doc)
            logger.info(f"Analysis saved: {analysis_data.get('analysis_id')}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Error saving analysis: {str(e)}")
            return None
    
    def get_user_analyses(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get user's analysis history"""
        try:
            analyses = list(
                self.db.analyses.find(
                    {'user_id': user_id}
                ).sort('timestamp', -1).limit(limit)
            )
            
            # Convert ObjectId to string
            for analysis in analyses:
                analysis['_id'] = str(analysis['_id'])
            
            return analyses
            
        except Exception as e:
            logger.error(f"Error retrieving user analyses: {str(e)}")
            return []
    
    def save_model_metadata(self, model_data: Dict[str, Any]) -> Optional[str]:
        """Save ML model metadata"""
        try:
            model_doc = {
                'model_name': model_data['model_name'],
                'version': model_data['version'],
                'model_type': model_data.get('model_type'),
                'architecture': model_data.get('architecture'),
                'parameters': model_data.get('parameters', {}),
                'performance_metrics': model_data.get('performance_metrics', {}),
                'training_info': {
                    'dataset_size': model_data.get('dataset_size'),
                    'training_time': model_data.get('training_time'),
                    'epochs': model_data.get('epochs'),
                    'batch_size': model_data.get('batch_size')
                },
                'federated_learning': {
                    'fl_round': model_data.get('fl_round'),
                    'participating_clients': model_data.get('participating_clients'),
                    'aggregation_method': model_data.get('aggregation_method')
                },
                'file_path': model_data.get('file_path'),
                'checksum': model_data.get('checksum'),
                'created_at': datetime.utcnow(),
                'created_by': model_data.get('created_by'),
                'is_active': model_data.get('is_active', True)
            }
            
            result = self.db.models.insert_one(model_doc)
            logger.info(f"Model metadata saved: {model_data['model_name']} v{model_data['version']}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Error saving model metadata: {str(e)}")
            return None
    
    def log_system_event(self, event_data: Dict[str, Any]):
        """Log system events for auditing"""
        try:
            log_doc = {
                'timestamp': datetime.utcnow(),
                'level': event_data.get('level', 'INFO'),
                'category': event_data.get('category', 'SYSTEM'),
                'event': event_data.get('event'),
                'user_id': event_data.get('user_id'),
                'session_id': event_data.get('session_id'),
                'ip_address': event_data.get('ip_address'),
                'user_agent': event_data.get('user_agent'),
                'details': event_data.get('details', {}),
                'security_relevant': event_data.get('security_relevant', False)
            }
            
            self.db.system_logs.insert_one(log_doc)
            
        except Exception as e:
            logger.error(f"Error logging system event: {str(e)}")
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get system usage statistics"""
        try:
            stats = {}
            
            # User statistics
            stats['users'] = {
                'total': self.db.users.count_documents({}),
                'active': self.db.users.count_documents({'is_active': True}),
                'last_week': self.db.users.count_documents({
                    'last_login': {'$gte': datetime.utcnow() - timedelta(days=7)}
                })
            }
            
            # Analysis statistics
            stats['analyses'] = {
                'total': self.db.analyses.count_documents({}),
                'last_24h': self.db.analyses.count_documents({
                    'timestamp': {'$gte': datetime.utcnow() - timedelta(days=1)}
                }),
                'last_week': self.db.analyses.count_documents({
                    'timestamp': {'$gte': datetime.utcnow() - timedelta(days=7)}
                })
            }
            
            # Disease distribution
            disease_pipeline = [
                {'$unwind': '$disease_classification.all_predictions'},
                {'$group': {
                    '_id': '$disease_classification.all_predictions.disease',
                    'count': {'$sum': 1}
                }},
                {'$sort': {'count': -1}}
            ]
            
            disease_stats = list(self.db.analyses.aggregate(disease_pipeline))
            stats['disease_distribution'] = {item['_id']: item['count'] for item in disease_stats}
            
            # Model statistics
            stats['models'] = {
                'total': self.db.models.count_documents({}),
                'active': self.db.models.count_documents({'is_active': True})
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting system statistics: {str(e)}")
            return {}
    
    def cleanup_old_data(self, retention_days: int = 90):
        """Clean up old data based on retention policy"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
            
            # Clean old system logs
            result = self.db.system_logs.delete_many({
                'timestamp': {'$lt': cutoff_date},
                'security_relevant': {'$ne': True}  # Keep security logs
            })
            logger.info(f"Cleaned up {result.deleted_count} old system log entries")
            
            # Clean old analyses (keep metadata, remove large data)
            self.db.analyses.update_many(
                {'timestamp': {'$lt': cutoff_date}},
                {'$unset': {
                    'raw_data': '',
                    'processed_data': '',
                    'intermediate_results': ''
                }}
            )
            
        except Exception as e:
            logger.error(f"Error during data cleanup: {str(e)}")
    
    def create_indexes(self):
        """Create database indexes for performance"""
        try:
            # Users collection indexes
            self.db.users.create_index('username', unique=True)
            self.db.users.create_index('email', unique=True, sparse=True)
            self.db.users.create_index([('username', 1), ('is_active', 1)])
            
            # Analyses collection indexes
            self.db.analyses.create_index('analysis_id', unique=True)
            self.db.analyses.create_index('user_id')
            self.db.analyses.create_index('subject_id')
            self.db.analyses.create_index([('timestamp', -1)])
            self.db.analyses.create_index([('user_id', 1), ('timestamp', -1)])
            
            # Models collection indexes
            self.db.models.create_index([('model_name', 1), ('version', -1)])
            self.db.models.create_index('is_active')
            self.db.models.create_index('created_at')
            
            # System logs indexes
            self.db.system_logs.create_index([('timestamp', -1)])
            self.db.system_logs.create_index([('level', 1), ('timestamp', -1)])
            self.db.system_logs.create_index('security_relevant')
            self.db.system_logs.create_index('user_id')
            
            # Expiry index for system logs (TTL)
            self.db.system_logs.create_index(
                'timestamp',
                expireAfterSeconds=365*24*3600  # 1 year
            )
            
            logger.info("Database indexes created successfully")
            
        except Exception as e:
            logger.error(f"Error creating indexes: {str(e)}")

# Global MongoDB handler instance
mongo_handler = MongoHandler()

# Connection management functions
def connect_database() -> bool:
    """Connect to MongoDB"""
    return mongo_handler.connect()

def disconnect_database():
    """Disconnect from MongoDB"""
    mongo_handler.disconnect()

def get_database():
    """Get database instance"""
    return mongo_handler.db

def get_async_database():
    """Get async database instance"""
    return mongo_handler.async_db