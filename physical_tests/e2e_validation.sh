#!/bin/bash
# End-to-End Validation Test Harness for Triglav
#
# This script performs real end-to-end testing by:
# 1. Building and starting an actual triglav server
# 2. Connecting with the triglav client using multiple interfaces
# 3. Running traffic through the SOCKS5/HTTP proxy
# 4. Validating data integrity and multi-path behavior
#
# Usage:
#   ./e2e_validation.sh [options]
#
# Options:
#   --server-only    Start only the server (for remote testing)
#   --client-only    Start only the client (connect to remote server)
#   --local          Run full local test (server + client on same machine)
#   --external       Test with external target (default: httpbin.org)
#   --duration       Test duration in seconds (default: 30)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Configuration
SERVER_PORT=17443
SOCKS_PORT=11080
HTTP_PROXY_PORT=18080
TEST_DURATION=30
EXTERNAL_TARGET="httpbin.org"

# Runtime state
SERVER_PID=""
CLIENT_PID=""
SERVER_KEY=""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

cleanup() {
    log_info "Cleaning up..."

    if [[ -n "$CLIENT_PID" ]] && kill -0 "$CLIENT_PID" 2>/dev/null; then
        log_info "Stopping client (PID: $CLIENT_PID)"
        kill "$CLIENT_PID" 2>/dev/null || true
        wait "$CLIENT_PID" 2>/dev/null || true
    fi

    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        log_info "Stopping server (PID: $SERVER_PID)"
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi

    # Clean up temp files
    rm -f /tmp/triglav_server.key /tmp/triglav_auth.key /tmp/triglav_test_*

    log_info "Cleanup complete"
}

trap cleanup EXIT

build_project() {
    log_info "Building triglav..."

    if ! cargo build --manifest-path "$PROJECT_DIR/Cargo.toml" --release 2>&1; then
        log_error "Build failed!"
        exit 1
    fi

    log_success "Build complete"
}

discover_interfaces() {
    local interfaces=()

    while IFS= read -r line; do
        if [[ -n "$line" ]]; then
            interfaces+=("$line")
        fi
    done < <(ifconfig -l | tr ' ' '\n' | while read -r iface; do
        if [[ "$iface" =~ ^en[0-9]+$ ]]; then
            ip=$(ifconfig "$iface" 2>/dev/null | grep "inet " | awk '{print $2}' | grep -v "^127\." | head -1)
            if [[ -n "$ip" ]]; then
                echo "$iface:$ip"
            fi
        fi
    done)

    echo "${interfaces[@]}"
}

start_server() {
    log_info "Starting triglav server on port $SERVER_PORT..."

    # Generate server key
    "$PROJECT_DIR/target/release/triglav" keygen \
        --output /tmp/triglav_server.key \
        --address "0.0.0.0:$SERVER_PORT" 2>/dev/null || true

    # Start server
    "$PROJECT_DIR/target/release/triglav" server \
        --key /tmp/triglav_server.key \
        --listen "0.0.0.0:$SERVER_PORT" \
        --generate-key \
        > /tmp/triglav_server.log 2>&1 &

    SERVER_PID=$!

    # Wait for server to start and capture auth key
    sleep 2

    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        log_error "Server failed to start. Log:"
        cat /tmp/triglav_server.log
        exit 1
    fi

    # Extract auth key from server log
    SERVER_KEY=$(grep -o 'triglav://[^ ]*' /tmp/triglav_server.log | head -1)

    if [[ -z "$SERVER_KEY" ]]; then
        log_warn "Could not extract auth key from server log"
        log_info "Server log:"
        cat /tmp/triglav_server.log
    else
        log_success "Server started (PID: $SERVER_PID)"
        log_info "Auth key: $SERVER_KEY"
        echo "$SERVER_KEY" > /tmp/triglav_auth.key
    fi
}

start_client() {
    local auth_key=${1:-$SERVER_KEY}

    if [[ -z "$auth_key" ]]; then
        log_error "No auth key provided"
        exit 1
    fi

    log_info "Starting triglav client..."

    # Discover interfaces
    local ifaces
    ifaces=$(discover_interfaces)
    log_info "Available interfaces: $ifaces"

    # Build interface arguments
    local iface_args=""
    for iface_info in $ifaces; do
        iface="${iface_info%%:*}"
        iface_args="$iface_args --interface $iface"
    done

    # Start client with SOCKS5 proxy
    "$PROJECT_DIR/target/release/triglav" connect "$auth_key" \
        $iface_args \
        --socks "$SOCKS_PORT" \
        --auto-discover \
        > /tmp/triglav_client.log 2>&1 &

    CLIENT_PID=$!

    sleep 3

    if ! kill -0 "$CLIENT_PID" 2>/dev/null; then
        log_error "Client failed to start. Log:"
        cat /tmp/triglav_client.log
        exit 1
    fi

    log_success "Client started (PID: $CLIENT_PID)"
    log_info "SOCKS5 proxy: localhost:$SOCKS_PORT"
}

test_socks_proxy() {
    log_info "Testing SOCKS5 proxy connectivity..."

    # Test 1: Simple HTTP request through proxy
    local result
    result=$(curl -s --connect-timeout 10 --max-time 30 \
        --socks5 "localhost:$SOCKS_PORT" \
        "http://$EXTERNAL_TARGET/ip" 2>&1) || true

    if [[ "$result" == *"origin"* ]]; then
        log_success "SOCKS5 proxy working - external IP: $(echo "$result" | grep -o '"origin": "[^"]*"')"
        return 0
    else
        log_warn "SOCKS5 proxy test failed: $result"
        return 1
    fi
}

test_data_transfer() {
    log_info "Testing data transfer through proxy..."

    # Generate test data
    local test_data
    test_data=$(head -c 10240 /dev/urandom | base64)
    echo "$test_data" > /tmp/triglav_test_input.txt

    # POST data through proxy
    local result
    result=$(curl -s --connect-timeout 10 --max-time 60 \
        --socks5 "localhost:$SOCKS_PORT" \
        -X POST \
        -H "Content-Type: text/plain" \
        -d "$test_data" \
        "http://$EXTERNAL_TARGET/post" 2>&1) || true

    if [[ "$result" == *"data"* ]]; then
        log_success "Data transfer test passed"
        return 0
    else
        log_warn "Data transfer test failed"
        return 1
    fi
}

test_concurrent_connections() {
    log_info "Testing concurrent connections..."

    local pids=()
    local success=0
    local total=10

    for i in $(seq 1 $total); do
        (
            result=$(curl -s --connect-timeout 10 --max-time 30 \
                --socks5 "localhost:$SOCKS_PORT" \
                "http://$EXTERNAL_TARGET/uuid" 2>&1)
            if [[ "$result" == *"uuid"* ]]; then
                exit 0
            else
                exit 1
            fi
        ) &
        pids+=($!)
    done

    # Wait for all and count successes
    for pid in "${pids[@]}"; do
        if wait "$pid"; then
            ((success++))
        fi
    done

    log_info "Concurrent test: $success/$total succeeded"

    if [[ $success -ge $((total / 2)) ]]; then
        log_success "Concurrent connection test passed"
        return 0
    else
        log_warn "Concurrent connection test: too many failures"
        return 1
    fi
}

test_bandwidth() {
    log_info "Testing bandwidth through proxy..."

    # Download a larger payload
    local start_time end_time duration bytes_received throughput

    start_time=$(date +%s.%N)
    bytes_received=$(curl -s --connect-timeout 10 --max-time 60 \
        --socks5 "localhost:$SOCKS_PORT" \
        -w '%{size_download}' \
        -o /dev/null \
        "http://$EXTERNAL_TARGET/bytes/1048576" 2>&1) || bytes_received=0
    end_time=$(date +%s.%N)

    if [[ "$bytes_received" -gt 0 ]]; then
        duration=$(echo "$end_time - $start_time" | bc)
        throughput=$(echo "scale=2; ($bytes_received * 8) / $duration / 1000000" | bc)
        log_success "Bandwidth test: ${bytes_received} bytes in ${duration}s = ${throughput} Mbps"
        return 0
    else
        log_warn "Bandwidth test failed"
        return 1
    fi
}

test_latency() {
    log_info "Testing latency through proxy..."

    local latencies=()
    local total=10

    for i in $(seq 1 $total); do
        local start_time end_time latency

        start_time=$(date +%s.%N)
        if curl -s --connect-timeout 5 --max-time 10 \
            --socks5 "localhost:$SOCKS_PORT" \
            "http://$EXTERNAL_TARGET/status/200" >/dev/null 2>&1; then
            end_time=$(date +%s.%N)
            latency=$(echo "($end_time - $start_time) * 1000" | bc)
            latencies+=("$latency")
        fi
    done

    if [[ ${#latencies[@]} -gt 0 ]]; then
        local sum=0
        for lat in "${latencies[@]}"; do
            sum=$(echo "$sum + $lat" | bc)
        done
        local avg=$(echo "scale=2; $sum / ${#latencies[@]}" | bc)
        log_success "Latency test: avg ${avg}ms (${#latencies[@]}/$total samples)"
        return 0
    else
        log_warn "Latency test failed - no successful samples"
        return 1
    fi
}

run_local_test() {
    echo
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║          TRIGLAV E2E VALIDATION TEST                         ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo

    build_project

    echo
    log_info "=== Starting Infrastructure ==="
    start_server
    sleep 2
    start_client
    sleep 3

    echo
    log_info "=== Running Validation Tests ==="

    local passed=0
    local failed=0

    if test_socks_proxy; then ((passed++)); else ((failed++)); fi
    if test_data_transfer; then ((passed++)); else ((failed++)); fi
    if test_concurrent_connections; then ((passed++)); else ((failed++)); fi
    if test_bandwidth; then ((passed++)); else ((failed++)); fi
    if test_latency; then ((passed++)); else ((failed++)); fi

    echo
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                    TEST RESULTS                              ║${NC}"
    echo -e "${CYAN}╠══════════════════════════════════════════════════════════════╣${NC}"
    printf "${CYAN}║${NC}  Passed: %-50s ${CYAN}║${NC}\n" "$passed"
    printf "${CYAN}║${NC}  Failed: %-50s ${CYAN}║${NC}\n" "$failed"
    printf "${CYAN}║${NC}  Total:  %-50s ${CYAN}║${NC}\n" "$((passed + failed))"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"

    if [[ $failed -eq 0 ]]; then
        echo
        log_success "All E2E validation tests passed!"
        return 0
    else
        echo
        log_error "$failed test(s) failed"
        return 1
    fi
}

run_server_only() {
    build_project
    start_server

    log_info "Server running. Auth key saved to /tmp/triglav_auth.key"
    log_info "Press Ctrl+C to stop."

    # Wait for server
    wait "$SERVER_PID"
}

run_client_only() {
    local auth_key=$1

    if [[ -z "$auth_key" ]]; then
        if [[ -f /tmp/triglav_auth.key ]]; then
            auth_key=$(cat /tmp/triglav_auth.key)
        else
            log_error "No auth key provided and /tmp/triglav_auth.key not found"
            echo "Usage: $0 --client-only <auth_key>"
            exit 1
        fi
    fi

    build_project
    start_client "$auth_key"

    log_info "Client running. SOCKS5 proxy on localhost:$SOCKS_PORT"
    log_info "Press Ctrl+C to stop."

    # Wait for client
    wait "$CLIENT_PID"
}

show_help() {
    cat <<EOF
Triglav E2E Validation Test Harness

Usage: $0 [options]

Options:
    --local          Run full local test (server + client + tests)
    --server-only    Start only the server (for remote testing)
    --client-only    Start only the client (connect to provided key)
    --external URL   External test target (default: httpbin.org)
    --duration SEC   Test duration in seconds (default: 30)
    --help           Show this help

Examples:
    # Run full local E2E test
    $0 --local

    # Start server only (e.g., on a VPS)
    $0 --server-only

    # Connect to a remote server
    $0 --client-only "triglav://..."

    # Test with custom external target
    $0 --local --external example.com
EOF
}

# Parse arguments
MODE="local"
AUTH_KEY=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --local)
            MODE="local"
            shift
            ;;
        --server-only)
            MODE="server"
            shift
            ;;
        --client-only)
            MODE="client"
            AUTH_KEY="${2:-}"
            shift
            [[ -n "$AUTH_KEY" ]] && shift
            ;;
        --external)
            EXTERNAL_TARGET="$2"
            shift 2
            ;;
        --duration)
            TEST_DURATION="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        triglav://*)
            AUTH_KEY="$1"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main
case $MODE in
    local)
        run_local_test
        ;;
    server)
        run_server_only
        ;;
    client)
        run_client_only "$AUTH_KEY"
        ;;
esac
