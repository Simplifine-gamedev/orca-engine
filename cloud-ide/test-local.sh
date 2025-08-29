#!/bin/bash

echo "🧪 Testing Orca Cloud IDE locally..."

# Build Docker image
echo "🏗️ Building Docker image..."
docker build -f docker/Dockerfile.orca-cloud -t orca-cloud:latest .

# Start backend server in background
echo "🚀 Starting backend server..."
cd cloud-ide/backend
pip3 install -r requirements.txt
python3 server.py &
BACKEND_PID=$!
cd ../..

# Wait for backend to start
sleep 3

# Start frontend
echo "🎨 Starting frontend..."
cd cloud-ide/frontend
npm install
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev &
FRONTEND_PID=$!
cd ../..

echo "✅ Local test environment started!"
echo ""
echo "Frontend: http://localhost:3000"
echo "Backend API: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop..."

# Wait for interrupt
trap "kill $BACKEND_PID $FRONTEND_PID; docker stop $(docker ps -q --filter ancestor=orca-cloud:latest)" INT
wait
