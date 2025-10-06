# ============================================================================
# MISSING DATABASE FILE
# ============================================================================

# File: src/database/migrations.py
# Database migration utilities

from pymongo import MongoClient
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class DatabaseMigrator:
    def __init__(self, connection_string: str):
        self.client = MongoClient(connection_string)
        self.db = self.client['feai_db']
    
    def run_migrations(self):
        '''Run all pending migrations'''
        migrations = [
            self.migration_001_create_collections,
            self.migration_002_add_indexes,
            self.migration_003_user_roles
        ]
        
        for migration in migrations:
            try:
                migration()
                logger.info(f"Migration {migration.__name__} completed")
            except Exception as e:
                logger.error(f"Migration failed: {e}")
    
    def migration_001_create_collections(self):
        '''Create initial collections'''
        collections = ['users', 'analyses', 'models', 'audit_logs']
        for coll in collections:
            if coll not in self.db.list_collection_names():
                self.db.create_collection(coll)
    
    def migration_002_add_indexes(self):
        '''Add indexes for performance'''
        self.db.analyses.create_index('user_id')
        self.db.analyses.create_index('timestamp')
        self.db.users.create_index('email', unique=True)
    
    def migration_003_user_roles(self):
        '''Add user roles field'''
        self.db.users.update_many(
            {'role': {'$exists': False}},
            {'$set': {'role': 'user'}}
        )