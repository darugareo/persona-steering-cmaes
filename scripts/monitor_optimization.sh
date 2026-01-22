#!/bin/bash
# Monitor optimization progress

LOG_FILE="logs/optimization_episode-184019_A_iter10.log"
PID=3393152

echo "Monitoring optimization (PID: $PID)"
echo "Log file: $LOG_FILE"
echo "Press Ctrl+C to stop monitoring"
echo ""

while true; do
    if ! ps -p $PID > /dev/null 2>&1; then
        echo "=== PROCESS COMPLETED ==="
        echo ""
        tail -60 $LOG_FILE
        break
    fi

    clear
    echo "=== Optimization Monitor ($(date +%H:%M:%S)) ==="
    echo "PID: $PID - Status: Running"
    echo ""
    echo "Latest output:"
    echo "----------------------------------------"
    tail -25 $LOG_FILE
    echo "----------------------------------------"
    echo ""
    echo "Checking again in 60 seconds..."

    sleep 60
done
