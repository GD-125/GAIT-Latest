# File: scripts/restore_system.sh
# System restore script

echo "Starting system restore..."

# Stop services
docker-compose down

# Restore database
mongorestore --uri=$MONGODB_URI --archive=backups/latest_backup.gz --gzip

# Restore files
tar -xzf backups/files_backup.tar.gz -C /

# Restart services
docker-compose up -d

echo "✅ System restore complete"