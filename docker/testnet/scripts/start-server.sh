#!/bin/bash
set -e

KEY_FILE="/data/server.key"
AUTH_KEY_FILE="/data/client.auth"

# Generate key if not exists
if [ ! -f "$KEY_FILE" ]; then
    echo "Generating server key..."
    triglav-server --generate-key -k "$KEY_FILE" -l ${TRIGLAV_LISTEN:-0.0.0.0:7443} 2>&1 | tee /tmp/keygen.log &
    sleep 2
    kill %1 2>/dev/null || true
    
    # Extract auth key from output
    grep -E "^tg1_" /tmp/keygen.log > "$AUTH_KEY_FILE" 2>/dev/null || true
fi

# Display auth key for clients
if [ -f "$AUTH_KEY_FILE" ]; then
    echo "============================================="
    echo "CLIENT AUTH KEY:"
    cat "$AUTH_KEY_FILE"
    echo "============================================="
fi

# Configure routing if needed
echo "Setting up routing..."
ip route add 10.10.0.0/16 via 10.100.0.1 2>/dev/null || true
ip route add 10.20.0.0/16 via 10.100.0.1 2>/dev/null || true
ip route add 10.30.0.0/16 via 10.100.0.1 2>/dev/null || true

echo "Starting Triglav server..."
exec triglav-server \
    -k "$KEY_FILE" \
    -l ${TRIGLAV_LISTEN:-0.0.0.0:7443} \
    -m ${TRIGLAV_METRICS:-0.0.0.0:9090}
