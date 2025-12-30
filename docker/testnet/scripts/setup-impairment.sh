#!/bin/bash
# Network Impairment Setup Script
# Uses tc/netem to simulate various network conditions

set -e

# Default interface (typically the one facing "upstream")
IFACE=${INTERFACE:-eth1}

# Get parameters from environment
DELAY=${NETEM_DELAY:-0ms}
JITTER=${NETEM_JITTER:-0ms}
LOSS=${NETEM_LOSS:-0%}
RATE=${NETEM_RATE:-1gbit}
CORRUPT=${NETEM_CORRUPT:-0%}
REORDER=${NETEM_REORDER:-0%}
DUPLICATE=${NETEM_DUPLICATE:-0%}

echo "=========================================="
echo "Network Impairment Setup"
echo "=========================================="
echo "Interface: $IFACE"
echo "Delay: $DELAY (jitter: $JITTER)"
echo "Loss: $LOSS"
echo "Rate: $RATE"
echo "Corruption: $CORRUPT"
echo "Reorder: $REORDER"
echo "Duplicate: $DUPLICATE"
echo "=========================================="

# Enable IP forwarding (may fail on Docker Desktop, that's OK - sysctls handle it)
echo 1 > /proc/sys/net/ipv4/ip_forward 2>/dev/null || echo "Note: IP forwarding set via sysctl in docker-compose"

# Wait for interface to be up
for i in $(seq 1 30); do
    if ip link show $IFACE >/dev/null 2>&1; then
        break
    fi
    echo "Waiting for interface $IFACE..."
    sleep 1
done

# Clear any existing qdisc
tc qdisc del dev $IFACE root 2>/dev/null || true

# Build netem command
NETEM_OPTS=""

# Add delay with jitter (use normal distribution)
if [ "$DELAY" != "0ms" ]; then
    NETEM_OPTS="$NETEM_OPTS delay $DELAY"
    if [ "$JITTER" != "0ms" ]; then
        NETEM_OPTS="$NETEM_OPTS $JITTER distribution normal"
    fi
fi

# Add packet loss
if [ "$LOSS" != "0%" ]; then
    NETEM_OPTS="$NETEM_OPTS loss random $LOSS"
fi

# Add corruption
if [ "$CORRUPT" != "0%" ]; then
    NETEM_OPTS="$NETEM_OPTS corrupt $CORRUPT"
fi

# Add reordering
if [ "$REORDER" != "0%" ]; then
    NETEM_OPTS="$NETEM_OPTS reorder $REORDER gap 5"
fi

# Add duplication
if [ "$DUPLICATE" != "0%" ]; then
    NETEM_OPTS="$NETEM_OPTS duplicate $DUPLICATE"
fi

# Apply netem if we have any options
if [ -n "$NETEM_OPTS" ]; then
    echo "Applying netem: tc qdisc add dev $IFACE root netem $NETEM_OPTS"
    tc qdisc add dev $IFACE root handle 1: netem $NETEM_OPTS
    
    # Add rate limiting as child qdisc
    if [ "$RATE" != "0" ] && [ "$RATE" != "unlimited" ]; then
        echo "Adding rate limit: $RATE"
        tc qdisc add dev $IFACE parent 1: handle 2: tbf rate $RATE burst 32kbit latency 50ms
    fi
else
    # Just rate limiting
    if [ "$RATE" != "0" ] && [ "$RATE" != "unlimited" ]; then
        echo "Applying rate limit only: $RATE"
        tc qdisc add dev $IFACE root tbf rate $RATE burst 32kbit latency 50ms
    fi
fi

# Show final configuration
echo ""
echo "Final qdisc configuration:"
tc qdisc show dev $IFACE
echo ""

# Show routing
echo "Routing table:"
ip route show
echo ""

echo "Impairment setup complete. Forwarding traffic..."

# Keep container running
tail -f /dev/null
