# Use Ubuntu base image (CUDA support will be added via runtime)
FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    curl \
    wget \
    build-essential \
    man-db \
    manpages \
    manpages-dev \
    manpages-posix \
    manpages-posix-dev \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA support (optional - will work with or without GPU)
RUN apt-get update && apt-get install -y \
    nvidia-cuda-toolkit \
    && rm -rf /var/lib/apt/lists/* || echo "CUDA toolkit installation failed - continuing without GPU support"

# Create symbolic link for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements-docker.txt requirements.txt

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create cache directories for models
RUN mkdir -p /root/.cache/huggingface
RUN mkdir -p /root/.lai-nux-tool/model_cache

# Set up man pages
RUN mandb

# Create test environment setup script
RUN echo '#!/bin/bash\n\
echo "🔧 Creating Test Environment..."\n\
echo "================================="\n\
\n\
# Create test directories\n\
mkdir -p /app/test_env/{documents,images,scripts,logs,backups,projects}\n\
mkdir -p /app/test_env/projects/{web_app,data_analysis,scripts}\n\
mkdir -p /app/test_env/documents/{reports,notes,manuals}\n\
mkdir -p /app/test_env/images/{photos,screenshots,diagrams}\n\
\n\
# Create sample files\n\
echo "This is a sample report file." > /app/test_env/documents/reports/sample_report.txt\n\
echo "Meeting notes from today." > /app/test_env/documents/notes/meeting_notes.txt\n\
echo "User manual content here." > /app/test_env/documents/manuals/user_manual.txt\n\
\n\
# Create script files\n\
echo '"'"'#!/bin/bash\necho "Hello from test script!"\nls -la'"'"' > /app/test_env/scripts/hello.sh\n\
chmod +x /app/test_env/scripts/hello.sh\n\
\n\
echo '"'"'#!/bin/bash\necho "System information:"\nuname -a\ndf -h'"'"' > /app/test_env/scripts/system_info.sh\n\
chmod +x /app/test_env/scripts/system_info.sh\n\
\n\
# Create log files\n\
echo "2024-01-01 10:00:00 INFO: Application started" > /app/test_env/logs/app.log\n\
echo "2024-01-01 10:01:00 INFO: User logged in" >> /app/test_env/logs/app.log\n\
echo "2024-01-01 10:02:00 ERROR: Connection failed" >> /app/test_env/logs/app.log\n\
\n\
# Create configuration files\n\
echo "server_port=8080\ndatabase_url=localhost:5432\ndebug_mode=true" > /app/test_env/projects/web_app/config.properties\n\
\n\
echo "import pandas as pd\nimport numpy as np\n\ndef analyze_data():\n    print('"'"'Data analysis complete'"'"')" > /app/test_env/projects/data_analysis/analyze.py\n\
\n\
# Create some empty files with different extensions\n\
touch /app/test_env/documents/reports/{report1.pdf,report2.docx,report3.xlsx}\n\
touch /app/test_env/images/photos/{photo1.jpg,photo2.png,photo3.gif}\n\
touch /app/test_env/logs/{error.log,warning.log,debug.log}\n\
\n\
# Create hidden files\n\
echo "Hidden configuration" > /app/test_env/.hidden_config\n\
echo "Another hidden file" > /app/test_env/documents/.hidden_notes\n\
\n\
# Create symbolic links (remove existing ones first to avoid conflicts)\n\
rm -f /app/test_env/current_reports /app/test_env/quick_hello\n\
ln -s /app/test_env/documents/reports /app/test_env/current_reports\n\
ln -s /app/test_env/scripts/hello.sh /app/test_env/quick_hello\n\
\n\
# Create a large file for testing\n\
dd if=/dev/zero of=/app/test_env/backups/large_file.bin bs=1M count=10 2>/dev/null\n\
\n\
# Create files with special characters\n\
echo "File with spaces" > "/app/test_env/documents/file with spaces.txt"\n\
echo "File-with-dashes" > "/app/test_env/documents/file-with-dashes.txt"\n\
echo "File_with_underscores" > "/app/test_env/documents/file_with_underscores.txt"\n\
\n\
echo "✅ Test environment created successfully!"\n\
echo "📁 Test directory: /app/test_env/"\n\
echo "🎯 Ready for Linux command testing!"' > /create_test_env.sh && chmod +x /create_test_env.sh

# Copy and setup entrypoint script
COPY sh_files/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose port (if needed for future web interface)
EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["interactive"]
