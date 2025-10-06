"""
File: tests/test_database.py
Unit tests for database operations
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from src.database.mongo_handler import MongoHandler
from src.database.models import UploadModel, PredictionModel, UserModel
from src.database.migrations import DatabaseMigration, DataMigration


class TestMongoHandler:
    """Tests for MongoDB handler"""
    
    @patch('src.database.mongo_handler.MongoClient')
    def test_mongo_handler_initialization(self, mock_client):
        """Test MongoDB handler initialization"""
        handler = MongoHandler()
        
        assert handler is not None
    
    @patch('src.database.mongo_handler.MongoClient')
    def test_insert_upload(self, mock_client):
        """Test inserting upload document"""
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_collection.insert_one.return_value.inserted_id = "test_id_123"
        mock_db.__getitem__.return_value = mock_collection
        mock_client.return_value.__getitem__.return_value = mock_db
        
        handler = MongoHandler()
        
        document = {
            'patient_id': 'TEST001',
            'filename': 'test.csv',
            'upload_timestamp': datetime.now(),
            'data': []
        }
        
        upload_id = handler.insert_upload(document)
        
        assert upload_id is not None
        mock_collection.insert_one.assert_called_once()
    
    @patch('src.database.mongo_handler.MongoClient')
    def test_get_upload(self, mock_client):
        """Test retrieving upload by ID"""
        mock_db = MagicMock()
        mock_collection = MagicMock()
        
        expected_doc = {
            '_id': 'test_id',
            'patient_id': 'TEST001',
            'filename': 'test.csv'
        }
        
        mock_collection.find_one.return_value = expected_doc
        mock_db.__getitem__.return_value = mock_collection
        mock_client.return_value.__getitem__.return_value = mock_db
        
        handler = MongoHandler()
        result = handler.get_upload('test_id')
        
        assert result == expected_doc
    
    @patch('src.database.mongo_handler.MongoClient')
    def test_update_upload_status(self, mock_client):
        """Test updating upload status"""
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_collection.update_one.return_value.modified_count = 1
        mock_db.__getitem__.return_value = mock_collection
        mock_client.return_value.__getitem__.return_value = mock_db
        
        handler = MongoHandler()
        
        result = handler.update_upload_status('test_id', 'processing')
        
        assert result == True
        mock_collection.update_one.assert_called_once()
    
    @patch('src.database.mongo_handler.MongoClient')
    def test_query_uploads_by_patient(self, mock_client):
        """Test querying uploads by patient ID"""
        mock_db = MagicMock()
        mock_collection = MagicMock()
        
        expected_uploads = [
            {'_id': '1', 'patient_id': 'TEST001'},
            {'_id': '2', 'patient_id': 'TEST001'}
        ]
        
        mock_collection.find.return_value = expected_uploads
        mock_db.__getitem__.return_value = mock_collection
        mock_client.return_value.__getitem__.return_value = mock_db
        
        handler = MongoHandler()
        results = handler.query_uploads_by_patient('TEST001')
        
        assert len(list(results)) == 2
    
    @patch('src.database.mongo_handler.MongoClient')
    def test_delete_upload(self, mock_client):
        """Test deleting upload"""
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_collection.delete_one.return_value.deleted_count = 1
        mock_db.__getitem__.return_value = mock_collection
        mock_client.return_value.__getitem__.return_value = mock_db
        
        handler = MongoHandler()
        result = handler.delete_upload('test_id')
        
        assert result == True


class TestDatabaseModels:
    """Tests for database models"""
    
    def test_upload_model_creation(self):
        """Test creating upload model"""
        upload = UploadModel(
            patient_id='TEST001',
            filename='test.csv',
            data=[],
            metadata={}
        )
        
        assert upload.patient_id == 'TEST001'
        assert upload.filename == 'test.csv'
    
    def test_prediction_model_creation(self):
        """Test creating prediction model"""
        prediction = PredictionModel(
            upload_id='upload_123',
            prediction=0.85,
            confidence=85.0,
            model_version='v1.0'
        )
        
        assert prediction.upload_id == 'upload_123'
        assert prediction.prediction == 0.85
        assert prediction.confidence == 85.0
    
    def test_user_model_creation(self):
        """Test creating user model"""
        user = UserModel(
            username='dr_smith',
            email='smith@hospital.com',
            role='Doctor'
        )
        
        assert user.username == 'dr_smith'
        assert user.role == 'Doctor'
    
    def test_model_to_dict(self):
        """Test converting model to dictionary"""
        upload = UploadModel(
            patient_id='TEST001',
            filename='test.csv',
            data=[],
            metadata={}
        )
        
        upload_dict = upload.to_dict()
        
        assert isinstance(upload_dict, dict)
        assert 'patient_id' in upload_dict
        assert upload_dict['patient_id'] == 'TEST001'


class TestDatabaseMigrations:
    """Tests for database migrations"""
    
    @patch('src.database.migrations.MongoClient')
    def test_migration_initialization(self, mock_client):
        """Test migration manager initialization"""
        mock_db = MagicMock()
        mock_db.list_collection_names.return_value = []
        
        migration = DatabaseMigration(mock_db)
        
        assert migration.db is not None
    
    @patch('src.database.migrations.MongoClient')
    def test_get_applied_migrations(self, mock_client):
        """Test getting applied migrations"""
        mock_db = MagicMock()
        mock_collection = MagicMock()
        
        mock_collection.find.return_value = [
            {'migration_id': '001_initial_schema'},
            {'migration_id': '002_add_indexes'}
        ]
        
        mock_db.__getitem__.return_value = mock_collection
        mock_db.list_collection_names.return_value = ['migrations']
        
        migration = DatabaseMigration(mock_db)
        applied = migration.get_applied_migrations()
        
        assert '001_initial_schema' in applied
        assert '002_add_indexes' in applied
    
    @patch('src.database.migrations.MongoClient')
    def test_mark_migration_applied(self, mock_client):
        """Test marking migration as applied"""
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_db.__getitem__.return_value = mock_collection
        mock_db.list_collection_names.return_value = ['migrations']
        
        migration = DatabaseMigration(mock_db)
        migration.mark_migration_applied('003_test_migration', 'Test migration')
        
        mock_collection.insert_one.assert_called_once()


class TestDatabaseIndexes:
    """Tests for database indexes"""
    
    @patch('src.database.mongo_handler.MongoClient')
    def test_index_creation(self, mock_client):
        """Test creating database indexes"""
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_db.__getitem__.return_value = mock_collection
        mock_client.return_value.__getitem__.return_value = mock_db
        
        handler = MongoHandler()
        
        # Create index
        handler.create_index('uploads', 'patient_id')
        
        mock_collection.create_index.assert_called()
    
    @patch('src.database.mongo_handler.MongoClient')
    def test_compound_index(self, mock_client):
        """Test creating compound index"""
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_db.__getitem__.return_value = mock_collection
        mock_client.return_value.__getitem__.return_value = mock_db
        
        handler = MongoHandler()
        
        # Create compound index
        handler.create_compound_index('uploads', ['patient_id', 'upload_timestamp'])
        
        mock_collection.create_index.assert_called()


class TestDatabaseQueries:
    """Tests for complex database queries"""
    
    @patch('src.database.mongo_handler.MongoClient')
    def test_aggregate_query(self, mock_client):
        """Test aggregation query"""
        mock_db = MagicMock()
        mock_collection = MagicMock()
        
        expected_result = [
            {'_id': 'TEST001', 'count': 5},
            {'_id': 'TEST002', 'count': 3}
        ]
        
        mock_collection.aggregate.return_value = expected_result
        mock_db.__getitem__.return_value = mock_collection
        mock_client.return_value.__getitem__.return_value = mock_db
        
        handler = MongoHandler()
        
        pipeline = [
            {'$group': {'_id': '$patient_id', 'count': {'$sum': 1}}}
        ]
        
        results = handler.aggregate('uploads', pipeline)
        
        assert len(list(results)) == 2
    
    @patch('src.database.mongo_handler.MongoClient')
    def test_date_range_query(self, mock_client):
        """Test querying by date range"""
        mock_db = MagicMock()
        mock_collection = MagicMock()
        
        start_date = datetime(2025, 1, 1)
        end_date = datetime(2025, 12, 31)
        
        mock_collection.find.return_value = []
        mock_db.__getitem__.return_value = mock_collection
        mock_client.return_value.__getitem__.return_value = mock_db
        
        handler = MongoHandler()
        
        results = handler.query_by_date_range('uploads', start_date, end_date)
        
        mock_collection.find.assert_called_once()


class TestDatabaseTransactions:
    """Tests for database transactions"""
    
    @patch('src.database.mongo_handler.MongoClient')
    def test_transaction_commit(self, mock_client):
        """Test transaction commit"""
        mock_db = MagicMock()
        mock_session = MagicMock()
        mock_client.return_value.start_session.return_value = mock_session
        mock_client.return_value.__getitem__.return_value = mock_db
        
        handler = MongoHandler()
        
        with handler.transaction() as session:
            # Perform operations
            pass
        
        mock_session.commit_transaction.assert_called()
    
    @patch('src.database.mongo_handler.MongoClient')
    def test_transaction_rollback(self, mock_client):
        """Test transaction rollback on error"""
        mock_db = MagicMock()
        mock_session = MagicMock()
        mock_client.return_value.start_session.return_value = mock_session
        mock_client.return_value.__getitem__.return_value = mock_db
        
        handler = MongoHandler()
        
        try:
            with handler.transaction() as session:
                raise Exception("Test error")
        except Exception:
            pass
        
        mock_session.abort_transaction.assert_called()


class TestDatabaseBackup:
    """Tests for database backup and restore"""
    
    @patch('src.database.mongo_handler.MongoClient')
    def test_backup_collection(self, mock_client):
        """Test backing up collection"""
        mock_db = MagicMock()
        mock_collection = MagicMock()
        
        mock_collection.find.return_value = [
            {'_id': '1', 'data': 'test1'},
            {'_id': '2', 'data': 'test2'}
        ]
        
        mock_db.__getitem__.return_value = mock_collection
        mock_client.return_value.__getitem__.return_value = mock_db
        
        handler = MongoHandler()
        
        backup_data = handler.backup_collection('uploads')
        
        assert len(list(backup_data)) == 2
    
    @patch('src.database.mongo_handler.MongoClient')
    def test_restore_collection(self, mock_client):
        """Test restoring collection"""
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_db.__getitem__.return_value = mock_collection
        mock_client.return_value.__getitem__.return_value = mock_db
        
        handler = MongoHandler()
        
        restore_data = [
            {'_id': '1', 'data': 'test1'},
            {'_id': '2', 'data': 'test2'}
        ]
        
        handler.restore_collection('uploads', restore_data)
        
        mock_collection.insert_many.assert_called_once()


class TestDatabasePerformance:
    """Tests for database performance"""
    
    @pytest.mark.slow
    @patch('src.database.mongo_handler.MongoClient')
    def test_bulk_insert_performance(self, mock_client):
        """Test bulk insert performance"""
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_db.__getitem__.return_value = mock_collection
        mock_client.return_value.__getitem__.return_value = mock_db
        
        handler = MongoHandler()
        
        # Generate bulk data
        documents = [
            {'patient_id': f'TEST{i:04d}', 'data': []}
            for i in range(1000)
        ]
        
        handler.bulk_insert('uploads', documents)
        
        mock_collection.insert_many.assert_called_once()
    
    @patch('src.database.mongo_handler.MongoClient')
    def test_query_with_projection(self, mock_client):
        """Test query with field projection for performance"""
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_collection.find.return_value = []
        mock_db.__getitem__.return_value = mock_collection
        mock_client.return_value.__getitem__.return_value = mock_db
        
        handler = MongoHandler()
        
        # Query with projection (only specific fields)
        projection = {'patient_id': 1, 'filename': 1, '_id': 0}
        
        handler.query_with_projection('uploads', {}, projection)
        
        mock_collection.find.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
