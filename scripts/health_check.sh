# File: scripts/health_check.sh
# System health check script

echo "Performing health check..."

# Check API
curl -f http://localhost:8501/health || echo "❌ API Down"

# Check Database
mongosh --eval "db.adminCommand('ping')" || echo "❌ Database Down"

# Check disk space
df -h | grep -E '(^Filesystem|/dev/)' 

echo "✅ Health check complete"