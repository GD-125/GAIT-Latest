# File: src/database/__init__.py
# Database package initialization

"""
Database modules for FE-AI system
Provides MongoDB operations, models, and migrations
"""

from .mongo_handler import MongoHandler, mongo_handler, connect_database, disconnect_database

__all__ = ['MongoHandler', 'mongo_handler', 'connect_database', 'disconnect_database']