# File: scripts/backup_system.sh
# Comprehensive backup script

#!/bin/bash
set -e

# Configuration
BACKUP_ROOT="/opt/fe-ai/backups"
APP_ROOT="/opt/fe-ai/app"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="${BACKUP_ROOT}/${DATE}"
RETENTION_DAYS=30

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Check if running as root or with sudo
if [ "$EUID" -ne 0 ]; then
    error "This script must be run as root or with sudo"
fi

# Create backup directory
log "Creating backup directory: ${BACKUP_DIR}"
mkdir -p "${BACKUP_DIR}"

# Backup MongoDB
log "Backing up MongoDB..."
docker exec fe-ai-mongodb-prod mongodump \
    --host localhost \
    --port 27017 \
    --db fe_ai_system \
    --out /tmp/mongodb_backup

docker cp fe-ai-mongodb-prod:/tmp/mongodb_backup "${BACKUP_DIR}/mongodb"
docker exec fe-ai-mongodb-prod rm -rf /tmp/mongodb_backup

# Backup Redis
log "Backing up Redis..."
docker exec fe-ai-redis-prod redis-cli --rdb /tmp/dump.rdb
docker cp fe-ai-redis-prod:/tmp/dump.rdb "${BACKUP_DIR}/redis_dump.rdb"

# Backup application data
log "Backing up application data..."
tar -czf "${BACKUP_DIR}/app_data.tar.gz" -C "${APP_ROOT}" data/

# Backup configuration files
log "Backing up configuration..."
cp "${APP_ROOT}/config.yaml" "${BACKUP_DIR}/"
cp "${APP_ROOT}/.env" "${BACKUP_DIR}/" 2>/dev/null || warn ".env file not found"

# Backup trained models
log "Backing up trained models..."
tar -czf "${BACKUP_DIR}/models.tar.gz" -C "${APP_ROOT}" data/models/

# Backup logs (last 7 days only)
log "Backing up recent logs..."
find "${APP_ROOT}/logs" -name "*.log" -mtime -7 -exec tar -czf "${BACKUP_DIR}/logs.tar.gz" {} +

# Create backup manifest
log "Creating backup manifest..."
cat > "${BACKUP_DIR}/manifest.txt" << EOF
FE-AI System Backup
==================
Date: $(date)
Backup ID: ${DATE}
Backup Path: ${BACKUP_DIR}

Contents:
- MongoDB database (fe_ai_system)
- Redis data
- Application data
- Configuration files
- Trained ML models
- Recent logs (7 days)

Backup Size: $(du -sh "${BACKUP_DIR}" | cut -f1)

System Information:
- Hostname: $(hostname)
- OS: $(uname -a)
- Docker Version: $(docker --version)
- Disk Usage: $(df -h | grep -E "/$|/opt")
EOF

# Compress entire backup
log "Compressing backup..."
cd "${BACKUP_ROOT}"
tar -czf "${DATE}.tar.gz" "${DATE}/"
rm -rf "${DATE}/"

# Calculate checksum
log "Calculating checksum..."
sha256sum "${DATE}.tar.gz" > "${DATE}.tar.gz.sha256"

# Clean old backups
log "Cleaning old backups (older than ${RETENTION_DAYS} days)..."
find "${BACKUP_ROOT}" -name "*.tar.gz" -mtime +${RETENTION_DAYS} -delete
find "${BACKUP_ROOT}" -name "*.sha256" -mtime +${RETENTION_DAYS} -delete

# Upload to cloud storage (optional)
if [ -n "${BACKUP_S3_BUCKET}" ]; then
    log "Uploading backup to S3..."
    aws s3 cp "${BACKUP_ROOT}/${DATE}.tar.gz" "s3://${BACKUP_S3_BUCKET}/fe-ai-backups/"
    aws s3 cp "${BACKUP_ROOT}/${DATE}.tar.gz.sha256" "s3://${BACKUP_S3_BUCKET}/fe-ai-backups/"
fi

# Send notification
if [ -n "${SLACK_WEBHOOK_URL}" ]; then
    BACKUP_SIZE=$(du -sh "${BACKUP_ROOT}/${DATE}.tar.gz" | cut -f1)
    curl -X POST -H 'Content-type: application/json' \
        --data "{\"text\":\"✅ FE-AI backup completed successfully\\nBackup ID: ${DATE}\\nSize: ${BACKUP_SIZE}\\nLocation: ${BACKUP_ROOT}/${DATE}.tar.gz\"}" \
        "${SLACK_WEBHOOK_URL}"
fi

log "Backup completed successfully!"
log "Backup file: ${BACKUP_ROOT}/${DATE}.tar.gz"
log "Backup size: $(du -sh "${BACKUP_ROOT}/${DATE}.tar.gz" | cut -f1)"

exit 0