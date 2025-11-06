#!/bin/bash
set -e

# Check if IsaacGymEnvs exists and install it
if [ -d "/workspace/IsaacGymEnvs" ]; then
    echo "Installing IsaacGymEnvs..."
    cd /workspace/IsaacGymEnvs
    pip install -e .
else
    echo "Warning: IsaacGymEnvs directory not found at /workspace/IsaacGymEnvs"
fi

# Check if isaacgym exists and install it
if [ -d "/workspace/isaacgym/python" ]; then
    echo "Installing isaacgym..."
    cd /workspace/isaacgym/python
    pip install -e .
else
    echo "Warning: isaacgym directory not found at /workspace/isaacgym/python"
fi

# Install dexenv (should always exist)
if [ -d "/workspace/dexenv" ]; then
    echo "Installing dexenv..."
    cd /workspace/dexenv
    pip install -e .
else
    echo "Error: dexenv directory not found at /workspace/dexenv"
    exit 1
fi

eval "bash"

exec "$@"