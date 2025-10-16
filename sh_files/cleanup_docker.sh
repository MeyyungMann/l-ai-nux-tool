#!/bin/bash
# L-AI-NUX-TOOL Docker Cleanup Script
# Run this script to clean up Docker resources

echo "Docker Cleanup for L-AI-NUX-TOOL"
echo "================================"

# Stop all lai-nux-tool containers
echo "Stopping lai-nux-tool containers..."
docker stop $(docker ps -q --filter "ancestor=lai-nux-tool:latest") 2>/dev/null || echo "No containers to stop"

# Remove stopped containers
echo "Removing stopped containers..."
docker container prune -f

# Remove unused images (optional - uncomment if needed)
# echo "Removing unused images..."
# docker image prune -f

# Remove unused volumes (optional - uncomment if needed)
# echo "Removing unused volumes..."
# docker volume prune -f

echo "Cleanup completed!"
echo ""
echo "Current status:"
docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}"
