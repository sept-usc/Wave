#!/bin/bash

# Script to prepare the environment for Wave
# Sets up CUDA, Nsight Compute, and uv package manager

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }

# Check if we're in the right directory
[ ! -f "pyproject.toml" ] && { echo "Error: Run from project root"; exit 1; }

# 1. CUDA Installation
print_info "=== CUDA Installation ==="
print_info "CUDA 12.8 required. See: https://developer.nvidia.com/cuda-12-8-0-download-archive"
if command -v nvcc &> /dev/null; then
    print_success "CUDA found: $(nvcc --version | grep release | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')"
else
    print_warning "CUDA not found. Please install CUDA 12.8"
fi
echo ""

# 2. Nsight Compute Installation
print_info "=== Nsight Compute Installation ==="
if command -v ncu &> /dev/null; then
    print_success "Nsight Compute found"
else
    if [[ "$OSTYPE" == "linux-gnu"* ]] && command -v apt-get &> /dev/null; then
        print_info "Installing nsight-compute..."
        sudo apt-get install -y nsight-compute || print_warning "Installation failed"
    else
        print_warning "Please install Nsight Compute manually"
    fi
fi

# Check NVIDIA profiling configuration (Linux only)
if [[ "$OSTYPE" == "linux-gnu"* ]] && [ -f "/proc/driver/nvidia/params" ]; then
    RmProfilingAdminOnly=$(cat /proc/driver/nvidia/params 2>/dev/null | grep RmProfilingAdminOnly || echo "")
    if [ -n "$RmProfilingAdminOnly" ] && echo "$RmProfilingAdminOnly" | grep -q "RmProfilingAdminOnly: 1"; then
        print_info "Configuring NVIDIA profiling..."
        PROFILE_CONF="/etc/modprobe.d/nvidia-profile.conf"
        if [ ! -f "$PROFILE_CONF" ] || ! grep -q "NVreg_RestrictProfilingToAdminUsers=0" "$PROFILE_CONF"; then
            echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" | sudo tee "$PROFILE_CONF" > /dev/null
            sudo update-initramfs -u
            print_warning "REBOOT REQUIRED for profiling changes to take effect"
        fi
    fi
fi
echo ""

# 3. uv Installation
print_info "=== uv Installation ==="
if command -v uv &> /dev/null; then
    print_success "uv found"
else
    print_info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
echo ""

# 4. Install Python dependencies
print_info "=== Installing Python Dependencies ==="
if command -v uv &> /dev/null; then
    uv sync
    print_success "Dependencies installed"
else
    print_warning "uv not available. Please restart shell and run again"
fi
