#!/bin/bash

echo "🚀 Setting up Ollama in Docker container..."
echo "============================================="

# Start Ollama service in background
echo "📡 Starting Ollama service..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to start
echo "⏳ Waiting for Ollama to start..."
sleep 5

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "❌ Failed to start Ollama service"
    exit 1
fi

echo "✅ Ollama service started successfully!"

# Pull GPT-OSS model if not already present
echo "📥 Checking for GPT-OSS model..."
if ! ollama list | grep -q "gpt-oss:20b"; then
    echo "📥 Pulling GPT-OSS 20B model..."
    ollama pull gpt-oss:20b
    echo "✅ GPT-OSS model pulled successfully!"
else
    echo "✅ GPT-OSS model already available!"
fi

echo ""
echo "🎯 Ollama setup complete!"
echo "🌐 Ollama API: http://localhost:11434"
echo "🤖 Available models:"
ollama list

echo ""
echo "💡 You can now use Ollama mode in L-AI-NUX-TOOL!"
echo "   python gen_cmd.py --mode ollama 'your command description'"
