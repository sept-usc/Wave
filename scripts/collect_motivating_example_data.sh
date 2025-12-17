#!/bin/bash

# Script to collect metrics for various Llama configurations
# This script runs the collect_metrics.py script multiple times with different model parameters

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

gpu_name=$1

if [ -z "$gpu_name" ]; then
    print_error "GPU name is required"
    exit 1
fi

# Function to run a single configuration
run_config() {
    local model_type=$1
    local n_layer=$2
    local hidden_dim=$3
    local ffn_dim=$4
    local n_head=$5
    local n_positions=$6
    local max_tokens=$7
    local batch_size=$8
    local prompt_len=$9
    local target_dir="data/motivating_example/$gpu_name/raw/"
    
    print_info "Running configuration:"
    print_info "  Model: $model_type"
    print_info "  Layers: $n_layer"
    print_info "  Hidden dim: $hidden_dim"
    print_info "  FFN dim: $ffn_dim"
    print_info "  Heads: $n_head"
    print_info "  Positions: $n_positions"
    print_info "  Max tokens: $max_tokens"
    print_info "  Batch size: $batch_size"
    print_info "  Prompt len: $prompt_len"
    print_info "  Target dir: $target_dir"
    
    # Build the command
    cmd="uv run eval/lower_bound/collect_metrics.py"
    cmd="$cmd --model-type $model_type"
    cmd="$cmd --n_layer $n_layer"
    cmd="$cmd --hidden-dim $hidden_dim"
    cmd="$cmd --ffn-dim $ffn_dim"
    cmd="$cmd --n_head $n_head"
    cmd="$cmd --n_positions $n_positions"
    cmd="$cmd --max_tokens $max_tokens"
    cmd="$cmd --batch-size $batch_size"
    cmd="$cmd --prompt-len $prompt_len"
    cmd="$cmd --target-dir $target_dir"

    print_info "Executing: $cmd"
    
    # Run the command
    if eval $cmd; then
        print_success "Configuration completed successfully"
    else
        print_error "Configuration failed"
        return 1
    fi
    
    echo ""
}

# Main execution
main() {
    print_info "Starting metrics collection for multiple configurations"
    print_info "This will run collect_metrics.py for various Llama configurations"
    echo ""
    mkdir -p "data/motivating_example/$gpu_name/raw/"
    mkdir -p "data/motivating_example/$gpu_name/csv/"
    
    # Check if we're in the right directory
    if [ ! -f "eval/lower_bound/collect_metrics.py" ]; then
        print_error "collect_metrics.py not found. Please run this script from the project root directory."
        exit 1
    fi
    
    # Check if NCU is available
    if ! command -v ncu &> /dev/null; then
        print_error "NCU (NVIDIA Compute Profiler) is not installed or not in PATH"
        exit 1
    fi
    
    print_success "NCU found, proceeding with metrics collection"
    echo ""
    
    # Configuration sets
    # Llama with different hidden dimensions
    run_config "llama" 6 512 2048 4 512 2 1 1
    run_config "llama" 6 768 3072 6 512 2 1 1
    run_config "llama" 6 1024 4096 8 512 2 1 1
    run_config "llama" 6 2048 8192 16 512 2 1 1
    run_config "llama" 6 4096 16384 32 512 2 1 1
    
    # Llama with different layers
    run_config "llama" 8 1024 4096 8 512 2 1 1
    run_config "llama" 12 1024 4096 8 512 2 1 1
    run_config "llama" 16 1024 4096 8 512 2 1 1
    run_config "llama" 24 1024 4096 8 512 2 1 1
    run_config "llama" 32 1024 4096 8 512 2 1 1
    
    print_success "All configurations completed successfully!"
    print_info "Raw data saved to: data/motivating_example/$gpu_name/raw/"
    print_info "CSV data saved to: data/motivating_example/$gpu_name/csv/"
}

# Run the main function
main "$@" 