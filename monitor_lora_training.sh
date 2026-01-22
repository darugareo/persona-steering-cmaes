#!/bin/bash

echo "=== LoRA Response-Only Training Monitor ==="
echo "Started: $(date)"
echo ""

while true; do
    clear
    echo "=== LoRA Response-Only Training Progress ===$(date)"
    echo ""
    
    # Count completed models
    completed=$(find lora_models_response_only -name "lora_weights" -type d 2>/dev/null | wc -l)
    echo "✅ Completed personas: $completed / 21"
    echo ""
    
    # Check GPU 0 log
    echo "📊 GPU 0 Progress:"
    if [ -f lora_response_only_gpu0.log ]; then
        grep -E "^\[.*\] episode-" lora_response_only_gpu0.log | tail -1
        grep "Training time:" lora_response_only_gpu0.log | tail -3
    fi
    echo ""
    
    # Check GPU 1 log
    echo "📊 GPU 1 Progress:"
    if [ -f lora_response_only_gpu1.log ]; then
        grep -E "^\[.*\] episode-" lora_response_only_gpu1.log | tail -1
        grep "Training time:" lora_response_only_gpu1.log | tail -3
    fi
    echo ""
    
    # Check processes
    echo "🔄 Running processes:"
    ps aux | grep "train_lora_response_only" | grep -v grep | wc -l
    echo ""
    
    # List completed personas
    echo "📁 Completed personas:"
    ls -1 lora_models_response_only/*/lora_weights 2>/dev/null | sed 's|lora_models_response_only/||' | sed 's|/lora_weights||' | sort
    
    if [ $completed -eq 21 ]; then
        echo ""
        echo "🎉 All 21 personas completed!"
        break
    fi
    
    sleep 60
done
