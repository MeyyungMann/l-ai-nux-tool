#!/usr/bin/env python3
"""
Docker Cleanup Manager for L-AI-NUX-TOOL
Automated cleanup and management of Docker containers and resources
"""

import subprocess
import argparse
import sys
from pathlib import Path

def run_command(cmd, capture_output=True):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=capture_output, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def get_running_containers():
    """Get list of running containers."""
    success, stdout, _ = run_command("docker ps --format 'table {{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Status}}'")
    if success:
        return stdout
    return "Error getting container list"

def stop_all_lai_containers():
    """Stop all running lai-nux-tool containers."""
    print("🛑 Stopping all lai-nux-tool containers...")
    success, stdout, stderr = run_command("docker stop $(docker ps -q --filter 'ancestor=lai-nux-tool:latest') 2>/dev/null || true")
    
    if success or "requires at least 1 argument" in stderr:
        print("✅ All lai-nux-tool containers stopped")
        return True
    else:
        print(f"❌ Error stopping containers: {stderr}")
        return False

def remove_stopped_containers():
    """Remove all stopped containers."""
    print("🗑️  Removing stopped containers...")
    success, stdout, stderr = run_command("docker container prune -f")
    
    if success:
        # Extract reclaimed space from output
        if "Total reclaimed space:" in stdout:
            space_line = [line for line in stdout.split('\n') if "Total reclaimed space:" in line][0]
            print(f"✅ {space_line}")
        else:
            print("✅ Cleaned up stopped containers")
        return True
    else:
        print(f"❌ Error removing containers: {stderr}")
        return False

def remove_unused_images():
    """Remove unused Docker images."""
    print("🖼️  Removing unused images...")
    success, stdout, stderr = run_command("docker image prune -f")
    
    if success:
        if "Total reclaimed space:" in stdout:
            space_line = [line for line in stdout.split('\n') if "Total reclaimed space:" in line][0]
            print(f"✅ {space_line}")
        else:
            print("✅ Cleaned up unused images")
        return True
    else:
        print(f"❌ Error removing images: {stderr}")
        return False

def remove_unused_volumes():
    """Remove unused Docker volumes."""
    print("💾 Removing unused volumes...")
    success, stdout, stderr = run_command("docker volume prune -f")
    
    if success:
        if "Total reclaimed space:" in stdout:
            space_line = [line for line in stdout.split('\n') if "Total reclaimed space:" in line][0]
            print(f"✅ {space_line}")
        else:
            print("✅ Cleaned up unused volumes")
        return True
    else:
        print(f"❌ Error removing volumes: {stderr}")
        return False

def system_prune():
    """Run docker system prune to clean everything."""
    print("🧹 Running full system cleanup...")
    success, stdout, stderr = run_command("docker system prune -f")
    
    if success:
        if "Total reclaimed space:" in stdout:
            space_line = [line for line in stdout.split('\n') if "Total reclaimed space:" in line][0]
            print(f"✅ {space_line}")
        else:
            print("✅ System cleanup completed")
        return True
    else:
        print(f"❌ Error during system cleanup: {stderr}")
        return False

def show_docker_status():
    """Show current Docker status and resource usage."""
    print("📊 Docker Status")
    print("=" * 50)
    
    # Show running containers
    print("\n🔄 Running Containers:")
    success, stdout, _ = run_command("docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}'")
    if success:
        print(stdout)
    else:
        print("No running containers")
    
    # Show disk usage
    print("\n💾 Disk Usage:")
    success, stdout, _ = run_command("docker system df")
    if success:
        print(stdout)
    
    # Show image sizes
    print("\n🖼️  Image Sizes:")
    success, stdout, _ = run_command("docker images --format 'table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}'")
    if success:
        print(stdout)

def create_cleanup_script():
    """Create a cleanup script for easy future use."""
    script_content = '''#!/bin/bash
# L-AI-NUX-TOOL Docker Cleanup Script
# Run this script to clean up Docker resources

echo "🐳 L-AI-NUX-TOOL Docker Cleanup"
echo "================================"

# Stop all lai-nux-tool containers
echo "🛑 Stopping lai-nux-tool containers..."
docker stop $(docker ps -q --filter "ancestor=lai-nux-tool:latest") 2>/dev/null || echo "No containers to stop"

# Remove stopped containers
echo "🗑️  Removing stopped containers..."
docker container prune -f

# Remove unused images (optional - uncomment if needed)
# echo "🖼️  Removing unused images..."
# docker image prune -f

# Remove unused volumes (optional - uncomment if needed)
# echo "💾 Removing unused volumes..."
# docker volume prune -f

echo "✅ Cleanup completed!"
echo ""
echo "📊 Current status:"
docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}"
'''
    
    script_path = Path("cleanup_docker.sh")
    script_path.write_text(script_content)
    script_path.chmod(0o755)
    print(f"✅ Created cleanup script: {script_path}")
    
    # Also create Windows batch file
    batch_content = '''@echo off
REM L-AI-NUX-TOOL Docker Cleanup Script for Windows
echo 🐳 L-AI-NUX-TOOL Docker Cleanup
echo ================================

REM Stop all lai-nux-tool containers
echo 🛑 Stopping lai-nux-tool containers...
for /f "tokens=1" %%i in ('docker ps -q --filter "ancestor=lai-nux-tool:latest"') do docker stop %%i 2>nul

REM Remove stopped containers
echo 🗑️  Removing stopped containers...
docker container prune -f

echo ✅ Cleanup completed!
echo.
echo 📊 Current status:
docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}"
pause
'''
    
    batch_path = Path("cleanup_docker.bat")
    batch_path.write_text(batch_content)
    print(f"✅ Created Windows cleanup script: {batch_path}")

def main():
    parser = argparse.ArgumentParser(description="Docker Cleanup Manager for L-AI-NUX-TOOL")
    parser.add_argument('action', choices=[
        'stop', 'clean', 'prune', 'status', 'full-cleanup', 'create-script'
    ], help='Action to perform')
    parser.add_argument('--force', '-f', action='store_true', help='Skip confirmation prompts')
    
    args = parser.parse_args()
    
    print("🐳 Docker Cleanup Manager")
    print("=" * 50)
    
    if args.action == 'stop':
        stop_all_lai_containers()
    elif args.action == 'clean':
        stop_all_lai_containers()
        remove_stopped_containers()
    elif args.action == 'prune':
        system_prune()
    elif args.action == 'status':
        show_docker_status()
    elif args.action == 'full-cleanup':
        if not args.force:
            response = input("⚠️  This will remove ALL unused Docker resources. Continue? (y/N): ")
            if response.lower() != 'y':
                print("❌ Aborted")
                return
        
        print("🧹 Starting full cleanup...")
        stop_all_lai_containers()
        remove_stopped_containers()
        remove_unused_images()
        remove_unused_volumes()
        print("✅ Full cleanup completed!")
    elif args.action == 'create-script':
        create_cleanup_script()

if __name__ == "__main__":
    main()
