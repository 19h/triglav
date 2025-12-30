#!/bin/bash
set -e

echo "=========================================="
echo "Triglav Multi-Path Client Starting"
echo "=========================================="

# Wait for server to be ready
echo "Waiting for server..."
until curl -sf http://10.100.0.10:9090/health >/dev/null 2>&1; do
    sleep 1
done
echo "Server is ready!"

# Configure routing tables for each interface
echo "Configuring multi-path routing..."

# Table 10: WiFi path via eth0 (10.10.1.x network)
ip route add default via 10.10.1.1 table 10 2>/dev/null || true
ip rule add from 10.10.1.100 lookup 10 priority 100 2>/dev/null || true

# Table 20: Ethernet path via eth1 (10.20.1.x network)
ip route add default via 10.20.1.1 table 20 2>/dev/null || true
ip rule add from 10.20.1.100 lookup 20 priority 100 2>/dev/null || true

# Table 30: LTE path via eth2 (10.30.1.x network)
ip route add default via 10.30.1.1 table 30 2>/dev/null || true
ip rule add from 10.30.1.100 lookup 30 priority 100 2>/dev/null || true

# Default route (use WiFi by default for non-bound traffic)
ip route add default via 10.10.1.1 2>/dev/null || true

# Show routing configuration
echo ""
echo "Routing Tables:"
echo "---------------"
echo "Main table:"
ip route show
echo ""
echo "WiFi table (10):"
ip route show table 10
echo ""
echo "Ethernet table (20):"
ip route show table 20
echo ""
echo "LTE table (30):"
ip route show table 30
echo ""

# Show interfaces
echo "Network Interfaces:"
echo "-------------------"
ip addr show | grep -E "^[0-9]+:|inet "
echo ""

# Test connectivity on each path
echo "Testing path connectivity..."
echo ""

test_path() {
    local name=$1
    local src_ip=$2
    local dst_ip=$3
    
    echo -n "  $name ($src_ip): "
    if ping -c 1 -W 2 -I $src_ip $dst_ip >/dev/null 2>&1; then
        local rtt=$(ping -c 3 -W 2 -I $src_ip $dst_ip 2>/dev/null | tail -1 | awk -F'/' '{print $5}')
        echo "OK (RTT: ${rtt}ms)"
    else
        echo "FAILED"
    fi
}

test_path "WiFi" "10.10.1.100" "10.100.0.10"
test_path "Ethernet" "10.20.1.100" "10.100.0.10"
test_path "LTE" "10.30.1.100" "10.100.0.10"
echo ""

# Get server auth key (in production this would be provided)
SERVER_KEY=${TRIGLAV_SERVER_KEY:-""}

# If no key provided, try to fetch from server metrics endpoint
if [ -z "$SERVER_KEY" ]; then
    echo "No server key provided, waiting for key file..."
    # In a real scenario, the key would be shared securely
    # For testing, we'll wait and poll the server's shared volume
    for i in $(seq 1 30); do
        if [ -f /data/server_auth.key ]; then
            SERVER_KEY=$(cat /data/server_auth.key)
            break
        fi
        sleep 1
    done
fi

if [ -z "$SERVER_KEY" ]; then
    echo "WARNING: No server key available. Running in test mode."
    echo "Client will wait for manual key configuration..."
    
    # Keep container running for manual testing
    echo ""
    echo "Container ready for manual testing."
    echo "Interfaces available:"
    echo "  - WiFi:     10.10.1.100 via 10.10.1.1"
    echo "  - Ethernet: 10.20.1.100 via 10.20.1.1"
    echo "  - LTE:      10.30.1.100 via 10.30.1.1"
    echo ""
    echo "To manually test multipath:"
    echo "  triglav connect <server-key> --interface eth0 --interface eth1 --interface eth2"
    echo ""
    
    # Stay alive
    tail -f /dev/null
fi

echo "Connecting to server with key: ${SERVER_KEY:0:20}..."

# Start the client with all three interfaces
exec triglav connect "$SERVER_KEY" \
    --interface eth0 \
    --interface eth1 \
    --interface eth2 \
    --verbose \
    --socks 1080
