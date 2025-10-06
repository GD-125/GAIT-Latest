// File: scripts/init-mongo.js
// MongoDB initialization script

db = db.getSiblingDB('feai_db');

// Create collections
db.createCollection('users');
db.createCollection('analyses');
db.createCollection('models');
db.createCollection('audit_logs');

// Create indexes
db.users.createIndex({ "email": 1 }, { unique: true });
db.analyses.createIndex({ "user_id": 1 });
db.analyses.createIndex({ "timestamp": -1 });

// Insert admin user
db.users.insertOne({
    username: "admin",
    email: "admin@feai.com",
    role: "admin",
    created_at: new Date()
});

print("MongoDB initialized successfully");