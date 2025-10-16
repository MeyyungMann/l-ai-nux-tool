# Utility Scripts

This folder contains utility scripts for the L-AI-NUX-TOOL project.

## Available Utilities:

### 🐳 Docker Management
**`docker_cleanup.py`** - Docker cleanup and resource management
```bash
python src/utils/docker_cleanup.py status          # Show Docker status
python src/utils/docker_cleanup.py clean           # Clean stopped containers
python src/utils/docker_cleanup.py full-cleanup    # Complete cleanup
```

### 🔍 Validation
**`validate_linux_commands.py`** - Validate generated Linux commands
```bash
python src/utils/validate_linux_commands.py
```

### 💾 Model Cache Management
**`save_quantized_model.py`** - Save quantized model for faster loading
```bash
python src/utils/save_quantized_model.py
```

**`load_quantized_model.py`** - Load pre-quantized model
```bash
python src/utils/load_quantized_model.py
```

## Usage

### Docker Cleanup
Clean up Docker resources to free disk space:
```bash
# Quick cleanup
python src/utils/docker_cleanup.py clean

# Full cleanup with confirmation
python src/utils/docker_cleanup.py full-cleanup
```

### Model Cache
Create a quantized model cache for 4x faster loading:
```bash
# Save quantized model (run once)
python src/utils/save_quantized_model.py

# Load and test quantized model
python src/utils/load_quantized_model.py
```

### Command Validation
Validate that generated commands are safe and correct:
```bash
python src/utils/validate_linux_commands.py
```

## Integration

These utilities can be imported and used in other scripts:
```python
# Example: Using docker cleanup in a script
from src.utils.docker_cleanup import stop_all_lai_containers, show_docker_status

# Stop containers
stop_all_lai_containers()

# Show status
show_docker_status()
```

