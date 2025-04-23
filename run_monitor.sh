#!/bin/bash

# ATC Radio Monitor Runner
# This script makes it easy to run the ATC Radio Monitor with different durations

# Default duration in seconds
DURATION=30

# Display help message
show_help() {
    echo "ATC Radio Monitor"
    echo "Usage: ./run_monitor.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -d, --duration SECONDS  Set the sampling duration (default: 30)"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run_monitor.sh                  # Run with default 30s duration"
    echo "  ./run_monitor.sh -d 60            # Run with 60s duration"
    echo "  ./run_monitor.sh --duration 120   # Run with 120s duration"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--duration)
            DURATION="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check if duration is a valid number
if ! [[ "$DURATION" =~ ^[0-9]+$ ]]; then
    echo "Error: Duration must be a positive integer"
    exit 1
fi

# Ensure the script is executable
if [ ! -x "$(command -v python)" ]; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

# Execute the monitor
echo "Starting ATC Radio Monitor with ${DURATION}s duration..."
python atc_monitor.py "$DURATION"