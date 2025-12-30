#!/bin/bash
# Network Impairment Script for macOS
# Uses dnctl (dummynet) and pfctl (packet filter) to simulate network conditions.
#
# REQUIRES: sudo privileges
#
# Usage:
#   ./network_impairment.sh setup <interface> <latency_ms> <loss_percent> <bandwidth_kbps>
#   ./network_impairment.sh teardown
#   ./network_impairment.sh status
#
# Examples:
#   ./network_impairment.sh setup en0 100 5 1000    # 100ms latency, 5% loss, 1Mbps
#   ./network_impairment.sh setup en0 50 0 0        # 50ms latency only
#   ./network_impairment.sh teardown                 # Remove all impairment

set -e

PIPE_NUM=1
PF_ANCHOR="triglav_test"

usage() {
    cat <<EOF
Network Impairment Tool for Triglav Physical Testing

Usage:
    $0 setup <interface> <latency_ms> <loss_percent> [bandwidth_kbps]
    $0 teardown
    $0 status
    $0 list-interfaces

Commands:
    setup       - Apply network impairment to an interface
    teardown    - Remove all network impairment
    status      - Show current impairment status
    list-interfaces - List available network interfaces

Parameters:
    interface     - Network interface (e.g., en0, en1)
    latency_ms    - Additional latency in milliseconds (0 for none)
    loss_percent  - Packet loss percentage (0-100)
    bandwidth_kbps - Bandwidth limit in Kbps (0 for unlimited)

Examples:
    # Add 100ms latency and 5% packet loss to WiFi
    sudo $0 setup en0 100 5

    # Add 50ms latency with 1Mbps bandwidth limit
    sudo $0 setup en0 50 0 1000

    # Simulate poor mobile connection (200ms latency, 10% loss)
    sudo $0 setup en0 200 10

    # Remove all impairment
    sudo $0 teardown
EOF
}

check_root() {
    if [[ $EUID -ne 0 ]]; then
        echo "Error: This script requires root privileges."
        echo "Please run with: sudo $0 $*"
        exit 1
    fi
}

list_interfaces() {
    echo "Available network interfaces:"
    echo
    networksetup -listallhardwareports | grep -A 2 "Hardware Port:" | while read -r line; do
        if [[ "$line" == "Hardware Port:"* ]]; then
            port="${line#Hardware Port: }"
        elif [[ "$line" == "Device:"* ]]; then
            device="${line#Device: }"
            # Get IP if available
            ip=$(ifconfig "$device" 2>/dev/null | grep "inet " | awk '{print $2}' | head -1)
            if [[ -n "$ip" ]]; then
                printf "  %-8s %-25s %s\n" "$device" "$port" "$ip"
            else
                printf "  %-8s %-25s %s\n" "$device" "$port" "(no IP)"
            fi
        fi
    done
}

setup_impairment() {
    local interface=$1
    local latency=$2
    local loss=$3
    local bandwidth=${4:-0}

    check_root

    # Validate interface exists
    if ! ifconfig "$interface" &>/dev/null; then
        echo "Error: Interface '$interface' not found"
        list_interfaces
        exit 1
    fi

    echo "Setting up network impairment on $interface:"
    echo "  Latency:   ${latency}ms"
    echo "  Loss:      ${loss}%"
    echo "  Bandwidth: ${bandwidth:-unlimited} Kbps"
    echo

    # First, clean up any existing rules
    teardown_impairment 2>/dev/null || true

    # Configure dummynet pipe
    local pipe_config="pipe $PIPE_NUM config"

    if [[ $latency -gt 0 ]]; then
        pipe_config="$pipe_config delay ${latency}ms"
    fi

    if [[ $loss -gt 0 ]]; then
        pipe_config="$pipe_config plr $(echo "scale=4; $loss/100" | bc)"
    fi

    if [[ $bandwidth -gt 0 ]]; then
        pipe_config="$pipe_config bw ${bandwidth}Kbit/s"
    fi

    echo "Creating dummynet pipe..."
    dnctl $pipe_config

    # Create pf rules
    echo "Creating packet filter rules..."

    # Create anchor rules
    cat > /tmp/triglav_pf.rules <<EOF
# Triglav test network impairment rules
dummynet-anchor "${PF_ANCHOR}"
anchor "${PF_ANCHOR}"
EOF

    # Create the actual dummynet rules
    cat > /tmp/triglav_pf_anchor.rules <<EOF
# Route traffic through dummynet pipe
dummynet out on $interface all pipe $PIPE_NUM
dummynet in on $interface all pipe $PIPE_NUM
EOF

    # Load the rules
    pfctl -f /tmp/triglav_pf.rules 2>/dev/null || true
    pfctl -a "${PF_ANCHOR}" -f /tmp/triglav_pf_anchor.rules

    # Enable pf if not already enabled
    pfctl -e 2>/dev/null || true

    echo
    echo "Network impairment active on $interface"
    echo "Run '$0 teardown' to remove when done."
}

teardown_impairment() {
    check_root

    echo "Removing network impairment..."

    # Flush the anchor rules
    pfctl -a "${PF_ANCHOR}" -F all 2>/dev/null || true

    # Remove dummynet pipe
    dnctl pipe $PIPE_NUM delete 2>/dev/null || true

    # Clean up temp files
    rm -f /tmp/triglav_pf.rules /tmp/triglav_pf_anchor.rules

    echo "Network impairment removed."
}

show_status() {
    echo "=== Dummynet Pipes ==="
    dnctl list 2>/dev/null || echo "(none)"
    echo
    echo "=== PF Anchor Rules ==="
    pfctl -a "${PF_ANCHOR}" -s rules 2>/dev/null || echo "(none)"
    echo
    echo "=== PF Status ==="
    pfctl -s info 2>/dev/null | head -5
}

# Main
case "${1:-}" in
    setup)
        if [[ $# -lt 4 ]]; then
            echo "Error: setup requires interface, latency, and loss parameters"
            usage
            exit 1
        fi
        setup_impairment "$2" "$3" "$4" "${5:-0}"
        ;;
    teardown)
        teardown_impairment
        ;;
    status)
        show_status
        ;;
    list-interfaces)
        list_interfaces
        ;;
    *)
        usage
        exit 1
        ;;
esac
