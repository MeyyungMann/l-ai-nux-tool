"""
Pytest configuration and shared fixtures for L-AI-NUX-TOOL tests.
"""

import pytest
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_cache_dir(temp_dir):
    """Create a temporary cache directory."""
    cache_dir = temp_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@pytest.fixture
def mock_config(temp_dir):
    """Create a mock configuration object."""
    from src.config import Config
    
    # Create config in temp directory
    config_file = temp_dir / 'config.yaml'
    config_dir = temp_dir
    
    # Create Config instance and patch instance attributes
    config = Config()
    
    # Patch instance attributes (not class attributes)
    config.config_file = config_file
    config.config_dir = config_dir
    config.models_dir = config_dir / 'models'
    config.cache_dir = config_dir / 'cache'
    
    # Create directories
    config_dir.mkdir(parents=True, exist_ok=True)
    config.models_dir.mkdir(parents=True, exist_ok=True)
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    
    yield config


@pytest.fixture
def mock_llm_engine():
    """Create a mock LLM engine."""
    mock_engine = Mock()
    mock_engine.generate_command = Mock(return_value="ls -la")
    mock_engine.repair_command = Mock(return_value="find . -name '*.txt'")
    mock_engine.switch_mode = Mock(return_value=None)
    return mock_engine


@pytest.fixture
def mock_rag_engine():
    """Create a mock RAG engine."""
    mock_rag = Mock()
    mock_rag.retrieve_relevant_docs = Mock(return_value=[])
    mock_rag.get_context_for_query = Mock(return_value="Mock context")
    mock_rag.get_stats = Mock(return_value={
        'total_documents': 100,
        'indexed_vectors': 100,
        'embedder_model': 'all-MiniLM-L6-v2',
        'cache_dir': '/tmp/cache'
    })
    return mock_rag


@pytest.fixture
def sample_command_docs():
    """Sample command documentation for testing."""
    from src.doc_collector import CommandDoc
    
    return [
        CommandDoc(
            command="find",
            description="Search for files in a directory hierarchy",
            usage="find [path] [options]",
            examples=["find . -name '*.txt'", "find /home -type f -size +100M"],
            options={"-name": "Search by name", "-type": "Search by type"},
            category="file_operations"
        ),
        CommandDoc(
            command="grep",
            description="Search for patterns in files",
            usage="grep [pattern] [file]",
            examples=["grep 'error' log.txt", "grep -r 'pattern' /path"],
            options={"-r": "Recursive search", "-i": "Case insensitive"},
            category="text_processing"
        ),
    ]


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "ls -la"
    mock_client.chat.completions.create = Mock(return_value=mock_response)
    return mock_client


@pytest.fixture
def env_vars(monkeypatch):
    """Fixture to manage environment variables."""
    def _set_env(**kwargs):
        for key, value in kwargs.items():
            monkeypatch.setenv(key, value)
    
    def _unset_env(*keys):
        for key in keys:
            monkeypatch.delenv(key, raising=False)
    
    return type('EnvVars', (), {
        'set': _set_env,
        'unset': _unset_env
    })()


@pytest.fixture(autouse=True)
def reset_imports():
    """Reset module imports between tests."""
    # This ensures clean state between tests
    yield
    # Cleanup if needed


@pytest.fixture
def skip_if_no_api_key(monkeypatch):
    """Skip test if API key is not available."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key or api_key == 'sk-your-api-key-here':
        pytest.skip("API key not configured")


@pytest.fixture
def skip_if_no_gpu():
    """Skip test if GPU is not available."""
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
    except ImportError:
        pytest.skip("PyTorch not installed")


@pytest.fixture
def skip_if_no_docker():
    """Skip test if Docker is not available."""
    import subprocess
    try:
        subprocess.run(['docker', '--version'], 
                      capture_output=True, timeout=5)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pytest.skip("Docker not available")


@pytest.fixture
def test_env_dir():
    """Path to test environment directory."""
    return Path(__file__).parent.parent / "test_env"


@pytest.fixture
def sample_queries():
    """Sample queries for testing command generation."""
    return [
        "list all files in current directory",
        "find all text files recursively",
        "show disk usage",
        "compress all log files",
        "search for error in log files",
    ]


@pytest.fixture
def expected_commands():
    """Expected commands for sample queries."""
    return {
        "list all files in current directory": "ls -la",
        "find all text files recursively": "find . -type f -name '*.txt'",
        "show disk usage": "df -h",
        "compress all log files": "tar -czf logs.tar.gz *.log",
        "search for error in log files": "grep -r 'error' *.log",
    }

