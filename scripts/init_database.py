# File: scripts/init_database.py
# Database initialization script

#!/usr/bin/env python3

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.mongo_handler import mongo_handler
from src.utils.config import config
from src.utils.logger import setup_logger
import logging

# ...existing code...

def main():
    """Initialize FE-AI database"""
    
    # Setup logging
    logger = setup_logger("DatabaseInit")
    logger.info("Starting database initialization...")
    
    try:
        # Connect to database
        if not mongo_handler.connect():
            logger.error("Failed to connect to database")
            return False
        
        # Create indexes
        logger.info("Creating database indexes...")
        mongo_handler.create_indexes()
        
        # Create default admin user
        logger.info("Creating default admin user...")
        admin_data = {
            'username': 'admin',
            'email': 'admin@fe-ai-system.com',
            'password': 'admin123',  # Change this in production!
            'role': 'admin'
        }
        
        admin_id = mongo_handler.create_user(admin_data)
        if admin_id:
            logger.info(f"Default admin user created with ID: {admin_id}")
        else:
            logger.info("Admin user already exists or creation failed")
        
        # Create sample data (optional)
        if config.get('development.mock_data', False):
            logger.info("Creating sample data...")
            create_sample_data()
        
        logger.info("Database initialization completed successfully")
        return True

    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        return False
        
    finally:
        mongo_handler.disconnect()

def create_sample_data():
    """Create sample data for development"""
    
    # Create sample users
    sample_users = [
        {
            'username': 'dr_smith',
            'email': 'dr.smith@hospital.com',
            'password': 'password123',
            'role': 'doctor'
        },
        {
            'username': 'researcher_jane',
            'email': 'jane@research.org',
            'password': 'research123',
            'role': 'researcher'
        }
    ]
    
    for user_data in sample_users:
        user_id = mongo_handler.create_user(user_data)
        if user_id:
            logging.info(f"Sample user created: {user_data['username']}")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
