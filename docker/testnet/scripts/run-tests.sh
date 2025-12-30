#!/bin/bash
# Triglav Multi-Path Network Test Suite
# Validates multipath behavior under various network conditions

set -e

RESULTS_DIR="${RESULTS_DIR:-/results}"
SERVER_ADDR="${SERVER_ADDR:-10.100.0.10:7443}"
SERVER_METRICS="${SERVER_METRICS:-http://10.100.0.10:9090}"
CLIENT_ADDR="${CLIENT_ADDR:-client}"
VERBOSE="${VERBOSE:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

# Ensure results directory exists
mkdir -p "$RESULTS_DIR"

# Logging
log() {
    echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((TESTS_PASSED++))
}

log_failure() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((TESTS_FAILED++))
}

log_skip() {
    echo -e "${YELLOW}[SKIP]${NC} $1"
    ((TESTS_SKIPPED++))
}

log_section() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE} $1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

# Test helper functions
wait_for_service() {
    local url=$1
    local timeout=${2:-30}
    local start=$(date +%s)
    
    while true; do
        if curl -sf "$url" >/dev/null 2>&1; then
            return 0
        fi
        
        local now=$(date +%s)
        if [ $((now - start)) -ge $timeout ]; then
            return 1
        fi
        sleep 1
    done
}

measure_latency() {
    local host=$1
    local count=${2:-10}
    
    ping -c "$count" -q "$host" 2>/dev/null | tail -1 | awk -F'/' '{print $5}'
}

get_metric() {
    local metric=$1
    curl -sf "$SERVER_METRICS/metrics" 2>/dev/null | grep "^$metric " | awk '{print $2}' | head -1
}

# ============================================================
# INFRASTRUCTURE TESTS
# ============================================================

test_server_health() {
    log "Testing server health..."
    
    if curl -sf "$SERVER_METRICS/health" >/dev/null; then
        log_success "Server health check passed"
        return 0
    else
        log_failure "Server health check failed"
        return 1
    fi
}

test_server_metrics() {
    log "Testing server metrics endpoint..."
    
    local metrics=$(curl -sf "$SERVER_METRICS/metrics" 2>/dev/null)
    if [ -n "$metrics" ]; then
        if echo "$metrics" | grep -q "triglav_"; then
            log_success "Server metrics endpoint responding with Triglav metrics"
            return 0
        else
            log_failure "Server metrics endpoint missing Triglav metrics"
            return 1
        fi
    else
        log_failure "Server metrics endpoint not responding"
        return 1
    fi
}

# ============================================================
# CONNECTIVITY TESTS
# ============================================================

test_path_connectivity() {
    log_section "Path Connectivity Tests"
    
    local server_ip="10.100.0.10"
    local paths=("wifi_router:10.10.1.1" "ethernet_router:10.20.1.1" "lte_gateway:10.30.1.1")
    
    for path_info in "${paths[@]}"; do
        local name="${path_info%%:*}"
        local gateway="${path_info##*:}"
        
        log "Testing $name path via $gateway..."
        
        if ping -c 3 -W 2 "$gateway" >/dev/null 2>&1; then
            local latency=$(measure_latency "$gateway" 5)
            log_success "$name path: reachable (avg latency: ${latency}ms)"
        else
            log_failure "$name path: unreachable"
        fi
    done
}

test_end_to_end_connectivity() {
    log "Testing end-to-end connectivity to server..."
    
    if ping -c 5 -W 5 "10.100.0.10" >/dev/null 2>&1; then
        local latency=$(measure_latency "10.100.0.10" 10)
        log_success "Server reachable (avg latency: ${latency}ms)"
        return 0
    else
        log_failure "Server unreachable"
        return 1
    fi
}

# ============================================================
# LATENCY CHARACTERIZATION TESTS
# ============================================================

test_path_latency_characteristics() {
    log_section "Path Latency Characterization"
    
    # WiFi path - expect ~25ms (20ms router + 5ms ISP)
    log "Measuring WiFi path latency..."
    local wifi_latency=$(measure_latency "10.10.1.1" 20)
    if [ -n "$wifi_latency" ]; then
        local wifi_check=$(echo "$wifi_latency > 15 && $wifi_latency < 50" | bc -l)
        if [ "$wifi_check" = "1" ]; then
            log_success "WiFi latency in expected range: ${wifi_latency}ms (expected 15-50ms)"
        else
            log_failure "WiFi latency out of range: ${wifi_latency}ms (expected 15-50ms)"
        fi
    fi
    
    # Ethernet path - expect ~5ms (2ms router + 3ms ISP)
    log "Measuring Ethernet path latency..."
    local eth_latency=$(measure_latency "10.20.1.1" 20)
    if [ -n "$eth_latency" ]; then
        local eth_check=$(echo "$eth_latency < 20" | bc -l)
        if [ "$eth_check" = "1" ]; then
            log_success "Ethernet latency in expected range: ${eth_latency}ms (expected <20ms)"
        else
            log_failure "Ethernet latency out of range: ${eth_latency}ms (expected <20ms)"
        fi
    fi
    
    # LTE path - expect ~70ms (30ms gateway + 20ms core + overhead)
    log "Measuring LTE path latency..."
    local lte_latency=$(measure_latency "10.30.1.1" 20)
    if [ -n "$lte_latency" ]; then
        local lte_check=$(echo "$lte_latency > 25 && $lte_latency < 150" | bc -l)
        if [ "$lte_check" = "1" ]; then
            log_success "LTE latency in expected range: ${lte_latency}ms (expected 25-150ms)"
        else
            log_failure "LTE latency out of range: ${lte_latency}ms (expected 25-150ms)"
        fi
    fi
    
    echo ""
    echo "Latency Summary:"
    echo "  WiFi:     ${wifi_latency:-N/A}ms"
    echo "  Ethernet: ${eth_latency:-N/A}ms"
    echo "  LTE:      ${lte_latency:-N/A}ms"
    
    # Save to results
    cat > "$RESULTS_DIR/latency.json" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "wifi_ms": ${wifi_latency:-null},
    "ethernet_ms": ${eth_latency:-null},
    "lte_ms": ${lte_latency:-null}
}
EOF
}

# ============================================================
# PACKET LOSS TESTS
# ============================================================

test_packet_loss() {
    log_section "Packet Loss Tests"
    
    local count=100
    
    for target in "10.10.1.1:WiFi" "10.20.1.1:Ethernet" "10.30.1.1:LTE"; do
        local ip="${target%%:*}"
        local name="${target##*:}"
        
        log "Testing $name packet loss ($count packets)..."
        
        local result=$(ping -c $count -q "$ip" 2>&1)
        local loss=$(echo "$result" | grep -oE '[0-9]+% packet loss' | grep -oE '[0-9]+')
        
        if [ -n "$loss" ]; then
            echo "  $name: ${loss}% packet loss"
            
            case "$name" in
                "WiFi")
                    if [ "$loss" -le 5 ]; then
                        log_success "$name packet loss acceptable: ${loss}%"
                    else
                        log_failure "$name packet loss too high: ${loss}%"
                    fi
                    ;;
                "Ethernet")
                    if [ "$loss" -le 1 ]; then
                        log_success "$name packet loss acceptable: ${loss}%"
                    else
                        log_failure "$name packet loss too high: ${loss}%"
                    fi
                    ;;
                "LTE")
                    if [ "$loss" -le 10 ]; then
                        log_success "$name packet loss acceptable: ${loss}%"
                    else
                        log_failure "$name packet loss too high: ${loss}%"
                    fi
                    ;;
            esac
        else
            log_failure "Could not measure $name packet loss"
        fi
    done
}

# ============================================================
# BANDWIDTH TESTS
# ============================================================

test_bandwidth() {
    log_section "Bandwidth Tests"
    
    # Check if iperf3 server is available
    if ! nc -z 10.100.0.10 5201 2>/dev/null; then
        log_skip "iperf3 server not available (port 5201)"
        return 0
    fi
    
    log "Running bandwidth test..."
    
    local result=$(iperf3 -c 10.100.0.10 -t 10 -J 2>/dev/null)
    
    if [ -n "$result" ]; then
        local bps=$(echo "$result" | jq '.end.sum_sent.bits_per_second // 0')
        local mbps=$(echo "scale=2; $bps / 1000000" | bc)
        
        log_success "Bandwidth test completed: ${mbps} Mbps"
        
        # Save result
        echo "$result" > "$RESULTS_DIR/bandwidth.json"
    else
        log_failure "Bandwidth test failed"
    fi
}

# ============================================================
# MULTIPATH BEHAVIOR TESTS
# ============================================================

test_multipath_detection() {
    log_section "Multipath Detection Tests"
    
    log "Checking if multiple paths are detected..."
    
    local uplinks=$(curl -sf "$SERVER_METRICS/status" 2>/dev/null | jq '.uplinks | length' 2>/dev/null)
    
    if [ -n "$uplinks" ] && [ "$uplinks" -gt 1 ]; then
        log_success "Multiple uplinks detected: $uplinks paths"
    elif [ "$uplinks" = "1" ]; then
        log_failure "Only single uplink detected"
    else
        log_skip "Could not query uplink status"
    fi
}

test_path_failover() {
    log_section "Path Failover Simulation"
    
    log "This test requires manual chaos agent intervention"
    log "Run: docker exec chaos-agent /scripts/chaos-agent.sh single_link_failure"
    
    log_skip "Manual failover test - see instructions above"
}

# ============================================================
# STRESS TESTS
# ============================================================

test_concurrent_connections() {
    log_section "Concurrent Connection Test"
    
    log "Testing with 10 concurrent connections..."
    
    local success=0
    local failed=0
    
    for i in $(seq 1 10); do
        if curl -sf -o /dev/null -m 5 "$SERVER_METRICS/health" 2>/dev/null; then
            ((success++))
        else
            ((failed++))
        fi &
    done
    wait
    
    if [ $success -ge 8 ]; then
        log_success "Concurrent connections: $success/10 succeeded"
    else
        log_failure "Concurrent connections: only $success/10 succeeded"
    fi
}

# ============================================================
# METRICS VALIDATION
# ============================================================

test_metrics_collection() {
    log_section "Metrics Collection Tests"
    
    local metrics=("triglav_sessions_active" "triglav_uplinks_active" "triglav_bytes_received_total" "triglav_bytes_sent_total")
    
    for metric in "${metrics[@]}"; do
        local value=$(get_metric "$metric")
        if [ -n "$value" ]; then
            log_success "Metric $metric = $value"
        else
            log_failure "Metric $metric not found"
        fi
    done
}

# ============================================================
# REPORT GENERATION
# ============================================================

generate_report() {
    log_section "Test Results Summary"
    
    local total=$((TESTS_PASSED + TESTS_FAILED + TESTS_SKIPPED))
    local pass_rate=0
    if [ $total -gt 0 ]; then
        pass_rate=$(echo "scale=1; $TESTS_PASSED * 100 / $total" | bc)
    fi
    
    echo ""
    echo "======================================"
    echo "       TRIGLAV TEST RESULTS"
    echo "======================================"
    echo ""
    printf "  ${GREEN}Passed:${NC}  %3d\n" $TESTS_PASSED
    printf "  ${RED}Failed:${NC}  %3d\n" $TESTS_FAILED
    printf "  ${YELLOW}Skipped:${NC} %3d\n" $TESTS_SKIPPED
    echo "  ----------------------"
    printf "  Total:   %3d\n" $total
    echo ""
    echo "  Pass Rate: ${pass_rate}%"
    echo ""
    echo "======================================"
    
    # Save JSON report
    cat > "$RESULTS_DIR/report.json" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "tests_passed": $TESTS_PASSED,
    "tests_failed": $TESTS_FAILED,
    "tests_skipped": $TESTS_SKIPPED,
    "total": $total,
    "pass_rate": $pass_rate
}
EOF
    
    echo ""
    echo "Full report saved to: $RESULTS_DIR/report.json"
    
    # Return non-zero if any tests failed
    if [ $TESTS_FAILED -gt 0 ]; then
        return 1
    fi
    return 0
}

# ============================================================
# MAIN
# ============================================================

main() {
    log_section "Triglav Multi-Path Network Test Suite"
    echo "Started at: $(date)"
    echo "Server: $SERVER_ADDR"
    echo "Metrics: $SERVER_METRICS"
    echo ""
    
    # Wait for infrastructure
    log "Waiting for server to be ready..."
    if ! wait_for_service "$SERVER_METRICS/health" 60; then
        log_failure "Server did not become ready within 60 seconds"
        exit 1
    fi
    log_success "Server is ready"
    
    # Run test suites
    test_server_health
    test_server_metrics
    
    test_path_connectivity
    test_end_to_end_connectivity
    
    test_path_latency_characteristics
    test_packet_loss
    
    # These require specific setup
    test_bandwidth
    test_multipath_detection
    test_path_failover
    
    test_concurrent_connections
    test_metrics_collection
    
    # Generate final report
    generate_report
}

# Allow running individual tests
case "${1:-all}" in
    all)
        main
        ;;
    health)
        test_server_health
        ;;
    connectivity)
        test_path_connectivity
        test_end_to_end_connectivity
        ;;
    latency)
        test_path_latency_characteristics
        ;;
    loss)
        test_packet_loss
        ;;
    bandwidth)
        test_bandwidth
        ;;
    multipath)
        test_multipath_detection
        ;;
    failover)
        test_path_failover
        ;;
    stress)
        test_concurrent_connections
        ;;
    metrics)
        test_metrics_collection
        ;;
    *)
        echo "Usage: $0 {all|health|connectivity|latency|loss|bandwidth|multipath|failover|stress|metrics}"
        exit 1
        ;;
esac
