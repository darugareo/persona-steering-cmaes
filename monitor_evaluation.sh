#!/bin/bash
while true; do
    clear
    echo "=== LoRA Evaluation Progress === $(date)"
    echo ""
    
    # Count completed personas
    completed=$(find results/lora_evaluation_response_only -name "evaluation.json" 2>/dev/null | wc -l)
    echo "✅ Completed: $completed / 21 personas"
    echo ""
    
    # Show latest progress from log
    echo "📊 Latest activity:"
    tail -10 lora_evaluation.log | grep -E "(^\[|Turn|Average score|LoRA weights not found|Error)"
    echo ""
    
    # Check process
    if ps aux | grep -q "[e]valuate_lora.py"; then
        echo "🔄 Process running"
    else
        echo "⏸️  Process stopped"
        break
    fi
    
    if [ $completed -eq 21 ]; then
        echo ""
        echo "🎉 All personas evaluated!"
        break
    fi
    
    sleep 30
done
