#!/bin/bash

# GPU-PMC-Verifier Overhead Evaluation Script
# This script evaluates inference overhead for different model configurations
# using PyTorch, NVIDIA Nsight Compute (ncu), and hyperfine

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

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
EVAL_SCRIPT="$PROJECT_ROOT/eval/lower_bound/inference.py"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python3"
CREATE_MODEL_SCRIPT="$PROJECT_ROOT/eval/lower_bound/create_model.py"

# Get GPU name and mode from command line arguments
gpu_name=$1
mode=$2

if [ -z "$gpu_name" ]; then
    print_error "GPU name is required"
    print_error "Usage: $0 <gpu_name> <mode: hw|all>"
    print_error "Example: $0 4090 hw"
    exit 1
fi

if [ -z "$mode" ]; then
    print_error "Mode is required (hw|all)"
    print_error "Usage: $0 <gpu_name> <mode: hw|all>"
    exit 1
fi

if [[ "$mode" != "hw" && "$mode" != "all" ]]; then
    print_error "Invalid mode: $mode"
    print_error "Valid modes are: hw, all"
    exit 1
fi

OUTPUT_BASE="$PROJECT_ROOT/data/overhead/$gpu_name/$mode"

# Create output directories
mkdir -p "$OUTPUT_BASE"

# Model configurations to test
declare -a MODEL_CONFIGS=(
    "gpt2:6:1024:4096:16:512:2:4:1"
    "gpt2:8:3072:12288:48:1024:2:4:1"
    "llama:6:1024:4096:8:512:2:4:1"
    "llama:8:4096:16384:32:1024:2:4:1"
    "qwen:6:1024:4096:8:512:2:4:1"
    "qwen:8:4096:16384:32:1024:2:4:1"
)

# NCU metrics to collect (selected by mode)
FULL_NCU_METRICS="l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_miss.sum,smsp__sass_thread_inst_executed_op_dadd_pred_on.sum,smsp__sass_thread_inst_executed_op_dmul_pred_on.sum,smsp__sass_thread_inst_executed_op_dfma_pred_on.sum,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,smsp__sass_thread_inst_executed_op_hadd_pred_on.sum,smsp__sass_thread_inst_executed_op_hmul_pred_on.sum,smsp__sass_thread_inst_executed_op_hfma_pred_on.sum,sm__ops_path_tensor_src_fp8.sum,sm__ops_path_tensor_src_fp8_dst_fp16.sum,sm__ops_path_tensor_src_fp8_dst_fp32.sum,sm__ops_path_tensor_src_bf16_dst_fp32.sum,sm__ops_path_tensor_src_fp16_dst_fp16.sum,sm__ops_path_tensor_src_fp16_dst_fp32.sum,sm__ops_path_tensor_src_tf32_dst_fp32.sum,sm__ops_path_tensor_src_fp64.sum,smsp__sass_inst_executed_op_global_ld.sum,smsp__sass_inst_executed_op_global_st.sum,smsp__sass_inst_executed_op_local_ld.sum,smsp__sass_inst_executed_op_local_st.sum,smsp__sass_inst_executed_op_shared_ld.sum,smsp__sass_inst_executed_op_shared_st.sum"

if [ "$mode" = "all" ]; then
    NCU_METRICS="$FULL_NCU_METRICS"
else
    NCU_METRICS="gpu__time_duration.sum"
fi

# Function to parse model configuration
parse_model_config() {
    local config="$1"
    IFS=':' read -r model_type n_layer hidden_dim ffn_dim n_head n_positions max_tokens batch_size prompt_len <<< "$config"
    echo "$model_type $n_layer $hidden_dim $ffn_dim $n_head $n_positions $max_tokens $batch_size $prompt_len"
}

# Function to generate model identifier
get_model_id() {
    local model_type="$1"
    local n_layer="$2"
    local hidden_dim="$3"
    local ffn_dim="$4"
    local n_head="$5"
    local n_positions="$6"
    local max_tokens="$7"
    local batch_size="$8"
    local prompt_len="$9"
    
    echo "${model_type}_layer${n_layer}_embd${hidden_dim}_ffn${ffn_dim}_head${n_head}_pos${n_positions}_max${max_tokens}_batch${batch_size}_prompt${prompt_len}"
}

# Function to run inference and collect metrics
run_evaluation() {
    local config="$1"
    
    # Parse configuration
    read -r model_type n_layer hidden_dim ffn_dim n_head n_positions max_tokens batch_size prompt_len <<< "$(parse_model_config "$config")"
    
    # Generate model identifier
    local model_id=$(get_model_id "$model_type" "$n_layer" "$hidden_dim" "$ffn_dim" "$n_head" "$n_positions" "$max_tokens" "$batch_size" "$prompt_len")
    
    # Create output directory for this configuration
    local config_dir="$OUTPUT_BASE/$model_id"
    mkdir -p "$config_dir"
    
    print_info "=== Running evaluation for $model_id ==="
    
    # Step 0: Create model first (idempotent; will skip if already exists)
    print_info "Step 0: Creating model (if needed)..."
    local create_args="--model-type $model_type --n_layer $n_layer --hidden-dim $hidden_dim --ffn-dim $ffn_dim --n_head $n_head --n_positions $n_positions"
    $VENV_PYTHON "$CREATE_MODEL_SCRIPT" $create_args > "$config_dir/create_model.log" 2>&1 || true
    
    # Build command arguments
    local cmd_args="--model-type $model_type --n_layer $n_layer --hidden-dim $hidden_dim --ffn-dim $ffn_dim --n_head $n_head --n_positions $n_positions --max_new_tokens $max_tokens --batch-size $batch_size --prompt-len $prompt_len"
    
    # Step 1: Run inference first to ensure model is loaded and working
    print_info "Step 1: Running inference to verify model..."
    $VENV_PYTHON "$EVAL_SCRIPT" $cmd_args > "$config_dir/inference.log" 2>&1
    
    if [ $? -ne 0 ]; then
        print_error "Inference failed for $model_id"
        print_error "Check logs at: $config_dir/inference.log"
        return 1
    fi
    
    # Step 2: Measure direct timing (baseline performance)
    print_info "Step 2: Measuring direct timing (baseline)..."
    local direct_timing_output="$config_dir/direct_timing.json"
    
    hyperfine --warmup 1 --runs 3 --export-json "$direct_timing_output" \
        "$VENV_PYTHON $EVAL_SCRIPT $cmd_args" > "$config_dir/direct_timing.log" 2>&1
    
    if [ $? -ne 0 ]; then
        print_error "Direct timing failed for $model_id"
        print_error "Check logs at: $config_dir/direct_timing.log"
        return 1
    fi
    
    # Step 3: Measure NCU overhead timing
    print_info "Step 3: Measuring NCU overhead timing..."
    local ncu_timing_output="$config_dir/ncu_timing.json"
    local ncu_output="$config_dir/profile.ncu-rep"
    
    # Build the NCU command
    local ncu_cmd="ncu --config-file off --export $ncu_output --force-overwrite --replay-mode application --app-replay-mode relaxed --metrics $NCU_METRICS $VENV_PYTHON $EVAL_SCRIPT $cmd_args"
    
    hyperfine --warmup 1 --runs 3 --export-json "$ncu_timing_output" \
        "$ncu_cmd" > "$config_dir/ncu_timing.log" 2>&1
    
    if [ $? -ne 0 ]; then
        print_error "NCU overhead timing failed for $model_id"
        print_error "Check logs at: $config_dir/ncu_timing.log"
        return 1
    fi
    
    print_success "Completed evaluation for $model_id"
    print_info "Outputs saved to: $config_dir/"
    echo ""
}

# Function to collect and summarize results
summarize_results() {
    print_info "=== Collecting Results Summary ==="
    
    local summary_file="$OUTPUT_BASE/overhead_summary.txt"
    local csv_file="$OUTPUT_BASE/overhead_data.csv"
    
    # Create CSV header
    echo "model_id,direct_run1_ms,direct_run2_ms,direct_run3_ms,direct_mean_ms,direct_std_ms,ncu_run1_ms,ncu_run2_ms,ncu_run3_ms,ncu_mean_ms,ncu_std_ms,overhead_ms,overhead_percent" > "$csv_file"
    
    # Create summary header
    cat > "$summary_file" << EOF
GPU-PMC-Verifier Overhead Evaluation Summary
Generated: $(date)
GPU: $gpu_name
============================================

EOF
    
    # Process each configuration
    for config in "${MODEL_CONFIGS[@]}"; do
        read -r model_type n_layer hidden_dim ffn_dim n_head n_positions max_tokens batch_size prompt_len <<< "$(parse_model_config "$config")"
        local model_id=$(get_model_id "$model_type" "$n_layer" "$hidden_dim" "$ffn_dim" "$n_head" "$n_positions" "$max_tokens" "$batch_size" "$prompt_len")
        local config_dir="$OUTPUT_BASE/$model_id"
        
        if [ -d "$config_dir" ]; then
            echo "Processing results for: $model_id" >> "$summary_file"
            echo "Configuration: $model_type, layers=$n_layer, hidden=$hidden_dim, ffn=$ffn_dim, heads=$n_head, pos=$n_positions" >> "$summary_file"
            echo "Parameters: max_tokens=$max_tokens, batch_size=$batch_size, prompt_len=$prompt_len" >> "$summary_file"
            echo "" >> "$summary_file"
            
            local direct_timing_file="$config_dir/direct_timing.json"
            local ncu_timing_file="$config_dir/ncu_timing.json"
            local ncu_file="$config_dir/profile.ncu-rep"
            
            if [ -f "$direct_timing_file" ] && [ -f "$ncu_timing_file" ] && [ -f "$ncu_file" ]; then
                # Extract direct timing from hyperfine JSON
                local direct_run1_time=$(jq -r '.results[0].times[0] * 1000' "$direct_timing_file" 2>/dev/null || echo "N/A")
                local direct_run2_time=$(jq -r '.results[0].times[1] * 1000' "$direct_timing_file" 2>/dev/null || echo "N/A")
                local direct_run3_time=$(jq -r '.results[0].times[2] * 1000' "$direct_timing_file" 2>/dev/null || echo "N/A")
                local direct_mean_time=$(jq -r '.results[0].mean * 1000' "$direct_timing_file" 2>/dev/null || echo "N/A")
                local direct_std_time=$(jq -r '.results[0].stddev * 1000' "$direct_timing_file" 2>/dev/null || echo "N/A")
                
                # Extract NCU timing from hyperfine JSON
                local ncu_run1_time=$(jq -r '.results[0].times[0] * 1000' "$ncu_timing_file" 2>/dev/null || echo "N/A")
                local ncu_run2_time=$(jq -r '.results[0].times[1] * 1000' "$ncu_timing_file" 2>/dev/null || echo "N/A")
                local ncu_run3_time=$(jq -r '.results[0].times[2] * 1000' "$ncu_timing_file" 2>/dev/null || echo "N/A")
                local ncu_mean_time=$(jq -r '.results[0].mean * 1000' "$ncu_timing_file" 2>/dev/null || echo "N/A")
                local ncu_std_time=$(jq -r '.results[0].stddev * 1000' "$ncu_timing_file" 2>/dev/null || echo "N/A")
                
                # Calculate overhead
                local overhead_ms="N/A"
                local overhead_percent="N/A"
                if [[ "$direct_mean_time" != "N/A" && "$ncu_mean_time" != "N/A" ]]; then
                    overhead_ms=$(echo "$ncu_mean_time - $direct_mean_time" | bc -l 2>/dev/null || echo "N/A")
                    if [[ "$overhead_ms" != "N/A" && "$direct_mean_time" != "0" ]]; then
                        overhead_percent=$(echo "scale=2; ($overhead_ms / $direct_mean_time) * 100" | bc -l 2>/dev/null || echo "N/A")
                    fi
                fi
                
                # Add to CSV
                echo "$model_id,$direct_run1_time,$direct_run2_time,$direct_run3_time,$direct_mean_time,$direct_std_time,$ncu_run1_time,$ncu_run2_time,$ncu_run3_time,$ncu_mean_time,$ncu_std_time,$overhead_ms,$overhead_percent" >> "$csv_file"
                
                echo "  Direct timing (baseline):" >> "$summary_file"
                echo "    Run 1: ${direct_run1_time}ms" >> "$summary_file"
                echo "    Run 2: ${direct_run2_time}ms" >> "$summary_file"
                echo "    Run 3: ${direct_run3_time}ms" >> "$summary_file"
                echo "    Mean: ${direct_mean_time}ms" >> "$summary_file"
                echo "    Std Dev: ${direct_std_time}ms" >> "$summary_file"
                echo "" >> "$summary_file"
                
                echo "  NCU timing (with profiling):" >> "$summary_file"
                echo "    Run 1: ${ncu_run1_time}ms" >> "$summary_file"
                echo "    Run 2: ${ncu_run2_time}ms" >> "$summary_file"
                echo "    Run 3: ${ncu_run3_time}ms" >> "$summary_file"
                echo "    Mean: ${ncu_mean_time}ms" >> "$summary_file"
                echo "    Std Dev: ${ncu_std_time}ms" >> "$summary_file"
                echo "" >> "$summary_file"
                
                echo "  Overhead analysis:" >> "$summary_file"
                echo "    Absolute overhead: ${overhead_ms}ms" >> "$summary_file"
                echo "    Relative overhead: ${overhead_percent}%" >> "$summary_file"
                echo "    NCU profile file: profile.ncu-rep" >> "$summary_file"
                echo "" >> "$summary_file"
            fi
            
            echo "----------------------------------------" >> "$summary_file"
            echo "" >> "$summary_file"
        fi
    done
    
    print_success "Results summary saved to: $summary_file"
    print_success "CSV data saved to: $csv_file"
}

# Main execution
main() {
    print_info "GPU-PMC-Verifier Overhead Evaluation"
    print_info "===================================="
    print_info "GPU: $gpu_name"
    print_info "Mode: $mode"
    print_info "Project root: $PROJECT_ROOT"
    print_info "Output directory: $OUTPUT_BASE"
    print_info "NCU metrics: $NCU_METRICS"
    print_info "Number of configurations: ${#MODEL_CONFIGS[@]}"
    echo ""
    
    # Check dependencies
    if ! command -v ncu &> /dev/null; then
        print_error "ncu (NVIDIA Nsight Compute) not found in PATH"
        print_error "Please install NVIDIA Nsight Compute or add it to your PATH"
        exit 1
    fi
    
    if ! command -v hyperfine &> /dev/null; then
        print_error "hyperfine not found in PATH"
        print_error "Please install hyperfine: https://github.com/sharkdp/hyperfine"
        exit 1
    fi
    
    if ! command -v jq &> /dev/null; then
        print_error "jq not found in PATH"
        print_error "Please install jq for JSON processing"
        exit 1
    fi
    
    if ! command -v bc &> /dev/null; then
        print_error "bc (basic calculator) not found in PATH"
        print_error "Please install bc for mathematical calculations"
        exit 1
    fi
    
    # Check if evaluation script exists
    if [ ! -f "$EVAL_SCRIPT" ]; then
        print_error "Evaluation script not found at: $EVAL_SCRIPT"
        exit 1
    fi
    
    # Check if virtual environment exists
    if [ ! -f "$VENV_PYTHON" ]; then
        print_error "Virtual environment not found at: $VENV_PYTHON"
        exit 1
    fi
    
    print_success "Dependencies check passed."
    echo ""
    
    # Run evaluations for each configuration
    for config in "${MODEL_CONFIGS[@]}"; do
        run_evaluation "$config"
        
        # Small delay between configurations
        sleep 2
    done
    
    # Collect and summarize results
    summarize_results
    
    echo ""
    print_success "=== Evaluation Complete ==="
    print_info "All results saved to: $OUTPUT_BASE"
    print_info "Check the summary file for an overview of all results."
}

# Run main function
main "$@"
