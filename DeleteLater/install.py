"""
Installation script for the Linux AI Command Generator.
"""

#!/usr/bin/env python3

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_requirements():
    """Check if all requirements are met."""
    print("Checking requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        return False
    
    # Check if CUDA is available
    try:
        import torch
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("Warning: CUDA not available, will use CPU")
    except ImportError:
        print("Warning: PyTorch not installed, will install during setup")
    
    # Check if man command is available
    if not shutil.which('man'):
        print("Error: 'man' command not found. Please install man-db package")
        return False
    
    print("Requirements check passed!")
    return True

def install_dependencies():
    """Install Python dependencies."""
    print("Installing dependencies...")
    
    try:
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ], check=True)
        print("Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False

def setup_config():
    """Setup initial configuration."""
    print("Setting up configuration...")
    
    config_dir = Path.home() / '.lai-nux-tool'
    config_dir.mkdir(exist_ok=True)
    
    # Create initial config
    config_file = config_dir / 'config.yaml'
    if not config_file.exists():
        import yaml
        
        default_config = {
            'model': {
                'base_model': 'microsoft/DialoGPT-medium',
                'lora_model': 'lai-nux-tool/lora-command-generator',
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'max_length': 512,
                'temperature': 0.7,
                'top_p': 0.9
            },
            'api': {
                'base_url': 'https://api.openai.com/v1',
                'model': 'gpt-3.5-turbo',
                'max_tokens': 512,
                'temperature': 0.7
            },
            'man': {
                'cache_man_pages': True,
                'cache_duration': 3600,
                'max_depth': 7
            },
            'ui': {
                'theme': 'default',
                'show_confidence': True,
                'auto_execute': False
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        print(f"Configuration created at {config_file}")
    
    return True

def create_symlink():
    """Create symlink for gen-cmd command."""
    print("Creating gen-cmd symlink...")
    
    script_path = Path(__file__).parent / 'gen_cmd.py'
    bin_dir = Path.home() / '.local' / 'bin'
    bin_dir.mkdir(parents=True, exist_ok=True)
    
    symlink_path = bin_dir / 'gen-cmd'
    
    if symlink_path.exists():
        symlink_path.unlink()
    
    symlink_path.symlink_to(script_path)
    
    # Make executable
    script_path.chmod(0o755)
    
    print(f"Symlink created: {symlink_path}")
    print("Add ~/.local/bin to your PATH if not already there")
    
    return True

def main():
    """Main installation function."""
    print("Linux AI Command Generator - Installation")
    print("=" * 50)
    
    if not check_requirements():
        sys.exit(1)
    
    if not install_dependencies():
        sys.exit(1)
    
    if not setup_config():
        sys.exit(1)
    
    if not create_symlink():
        sys.exit(1)
    
    print("\nInstallation completed successfully!")
    print("\nNext steps:")
    print("1. Add ~/.local/bin to your PATH")
    print("2. Run 'gen-cmd --help' to test the installation")
    print("3. Run 'gen-cmd --interactive' to start using the tool")
    print("\nThe first run will download the base model (~350MB)")

if __name__ == '__main__':
    main()
