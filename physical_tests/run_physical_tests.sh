#!/bin/bash
# Physical Test Runner for Triglav
#
# This script orchestrates comprehensive physical testing of triglav's
# multi-path networking capabilities using real network interfaces.
#
# Usage:
#   ./run_physical_tests.sh [options] [test_name]
#
# Options:
#   --all           Run all physical tests
#   --quick         Run quick connectivity tests only
#   --stress        Run stress/performance tests
#   --failover      Run failover tests (may require manual intervention)
#   --impairment    Run tests with network impairment (requires sudo)
#   --external      Run tests against external endpoints
#   --verbose       Enable verbose output
#   --help          Show this help

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Test configuration
VERBOSE=false
TEST_TIMEOUT=120

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $*"
}

log_header() {
    echo
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC} $*"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo
}

show_help() {
    cat <<EOF
Triglav Physical Test Runner

Comprehensive testing of multi-path networking using real network interfaces.

Usage:
    $0 [options] [test_name]

Options:
    --all           Run all physical tests
    --quick         Run quick connectivity tests only
    --stress        Run stress/performance tests
    --failover      Run failover tests (may require manual intervention)
    --impairment    Run tests with network impairment (requires sudo)
    --external      Run tests against external endpoints
    --verbose       Enable verbose output
    --list          List available tests
    --help          Show this help

Individual Tests:
    interface_discovery      - Discover available network interfaces
    multi_interface          - Test connectivity on multiple interfaces
    multipath_manager        - Test MultipathManager with real interfaces
    bandwidth                - Measure bandwidth capabilities
    latency                  - Measure latency distribution
    concurrent               - Test concurrent traffic from multiple interfaces
    external                 - Test connectivity to external endpoints
    failover                 - Manual failover test

Examples:
    # Run all quick tests
    $0 --quick

    # Run specific test with verbose output
    $0 --verbose interface_discovery

    # Run stress tests
    $0 --stress

    # Run tests with network impairment (requires sudo)
    sudo $0 --impairment
EOF
}

list_tests() {
    log_header "Available Physical Tests"
    echo "Basic Tests:"
    echo "  interface_discovery     - Discover available network interfaces"
    echo "  multi_interface         - Test connectivity on multiple interfaces"
    echo "  multipath_manager       - Test MultipathManager with real interfaces"
    echo
    echo "Performance Tests:"
    echo "  bandwidth               - Measure bandwidth capabilities"
    echo "  latency                 - Measure latency distribution"
    echo "  concurrent              - Test concurrent traffic"
    echo
    echo "Advanced Tests:"
    echo "  external                - Test external endpoint connectivity"
    echo "  failover                - Manual failover test"
    echo
    echo "Test Suites:"
    echo "  --quick                 - Run: interface_discovery, multi_interface"
    echo "  --stress                - Run: bandwidth, latency, concurrent"
    echo "  --all                   - Run all tests"
}

check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check Rust toolchain
    if ! command -v cargo &>/dev/null; then
        log_error "Rust/Cargo not found. Please install Rust."
        exit 1
    fi

    # Check if project builds
    if ! cargo check --manifest-path "$PROJECT_DIR/Cargo.toml" &>/dev/null; then
        log_warn "Project has compilation issues. Running 'cargo build' first..."
        cargo build --manifest-path "$PROJECT_DIR/Cargo.toml"
    fi

    # Check for multiple interfaces
    local iface_count
    iface_count=$(ifconfig -l | tr ' ' '\n' | grep -E '^en[0-9]+$' | wc -l | tr -d ' ')

    if [[ $iface_count -lt 2 ]]; then
        log_warn "Only $iface_count network interface(s) found. Multi-path tests may be limited."
    else
        log_info "Found $iface_count network interfaces"
    fi

    log_success "Prerequisites check passed"
}

discover_interfaces() {
    log_info "Discovering network interfaces..."
    echo

    networksetup -listallhardwareports | while IFS= read -r line; do
        if [[ "$line" == "Hardware Port:"* ]]; then
            port="${line#Hardware Port: }"
        elif [[ "$line" == "Device:"* ]]; then
            device="${line#Device: }"
            ip=$(ifconfig "$device" 2>/dev/null | grep "inet " | awk '{print $2}' | head -1)
            status=$(ifconfig "$device" 2>/dev/null | grep "status:" | awk '{print $2}')

            if [[ -n "$ip" ]]; then
                echo -e "  ${GREEN}●${NC} $device ($port): $ip [$status]"
            elif [[ "$status" == "active" ]]; then
                echo -e "  ${YELLOW}○${NC} $device ($port): no IP [$status]"
            fi
        fi
    done
    echo
}

run_cargo_test() {
    local test_name=$1
    local extra_args=${2:-}

    log_info "Running test: $test_name"

    local cmd="cargo test --manifest-path $PROJECT_DIR/Cargo.toml --test physical_multipath $test_name -- --ignored --nocapture $extra_args"

    if $VERBOSE; then
        echo "Command: $cmd"
    fi

    if timeout "$TEST_TIMEOUT" bash -c "$cmd"; then
        log_success "Test passed: $test_name"
        return 0
    else
        log_error "Test failed: $test_name"
        return 1
    fi
}

run_quick_tests() {
    log_header "Running Quick Tests"

    local passed=0
    local failed=0

    discover_interfaces

    if run_cargo_test "test_interface_discovery"; then
        ((passed++))
    else
        ((failed++))
    fi

    if run_cargo_test "test_multi_interface_connectivity"; then
        ((passed++))
    else
        ((failed++))
    fi

    echo
    log_info "Quick tests complete: $passed passed, $failed failed"
}

run_stress_tests() {
    log_header "Running Stress Tests"

    local passed=0
    local failed=0

    if run_cargo_test "test_bandwidth_measurement"; then
        ((passed++))
    else
        ((failed++))
    fi

    if run_cargo_test "test_latency_distribution"; then
        ((passed++))
    else
        ((failed++))
    fi

    if run_cargo_test "test_concurrent_traffic"; then
        ((passed++))
    else
        ((failed++))
    fi

    echo
    log_info "Stress tests complete: $passed passed, $failed failed"
}

run_all_tests() {
    log_header "Running All Physical Tests"

    local passed=0
    local failed=0
    local tests=(
        "test_interface_discovery"
        "test_multi_interface_connectivity"
        "test_multipath_manager_real_interfaces"
        "test_bandwidth_measurement"
        "test_latency_distribution"
        "test_concurrent_traffic"
        "test_external_connectivity"
    )

    discover_interfaces

    for test in "${tests[@]}"; do
        if run_cargo_test "$test"; then
            ((passed++))
        else
            ((failed++))
        fi
        echo
    done

    log_header "Test Summary"
    echo "  Passed: $passed"
    echo "  Failed: $failed"
    echo "  Total:  ${#tests[@]}"

    if [[ $failed -eq 0 ]]; then
        log_success "All tests passed!"
    else
        log_error "$failed test(s) failed"
        return 1
    fi
}

run_impairment_tests() {
    log_header "Running Network Impairment Tests"

    # Check for sudo
    if [[ $EUID -ne 0 ]]; then
        log_error "Network impairment tests require root privileges."
        log_info "Please run: sudo $0 --impairment"
        exit 1
    fi

    discover_interfaces

    # Get primary interface
    local primary_iface
    primary_iface=$(route -n get default 2>/dev/null | grep "interface:" | awk '{print $2}')

    if [[ -z "$primary_iface" ]]; then
        log_error "Could not determine primary interface"
        exit 1
    fi

    log_info "Primary interface: $primary_iface"
    echo

    # Test 1: High latency scenario
    log_info "Test: High latency (100ms)"
    "$SCRIPT_DIR/network_impairment.sh" setup "$primary_iface" 100 0 0
    sleep 1
    run_cargo_test "test_latency_distribution" || true
    "$SCRIPT_DIR/network_impairment.sh" teardown
    sleep 1

    # Test 2: Packet loss scenario
    log_info "Test: Packet loss (10%)"
    "$SCRIPT_DIR/network_impairment.sh" setup "$primary_iface" 0 10 0
    sleep 1
    run_cargo_test "test_concurrent_traffic" || true
    "$SCRIPT_DIR/network_impairment.sh" teardown
    sleep 1

    # Test 3: Combined impairment
    log_info "Test: Combined (50ms latency, 5% loss)"
    "$SCRIPT_DIR/network_impairment.sh" setup "$primary_iface" 50 5 0
    sleep 1
    run_cargo_test "test_multi_interface_connectivity" || true
    "$SCRIPT_DIR/network_impairment.sh" teardown

    log_success "Network impairment tests complete"
}

run_failover_test() {
    log_header "Running Failover Test"

    log_warn "This test requires manual intervention!"
    echo "You will need to disconnect one network interface during the test."
    echo
    read -p "Press Enter to continue or Ctrl+C to cancel..."

    run_cargo_test "test_manual_failover"
}

run_external_test() {
    log_header "Running External Connectivity Test"

    log_info "Testing connectivity to external endpoints..."
    run_cargo_test "test_external_connectivity"
}

# Parse arguments
TESTS_TO_RUN=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            TESTS_TO_RUN+=("all")
            shift
            ;;
        --quick)
            TESTS_TO_RUN+=("quick")
            shift
            ;;
        --stress)
            TESTS_TO_RUN+=("stress")
            shift
            ;;
        --failover)
            TESTS_TO_RUN+=("failover")
            shift
            ;;
        --impairment)
            TESTS_TO_RUN+=("impairment")
            shift
            ;;
        --external)
            TESTS_TO_RUN+=("external")
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --list)
            list_tests
            exit 0
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        -*)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
        *)
            TESTS_TO_RUN+=("single:$1")
            shift
            ;;
    esac
done

# Main execution
log_header "Triglav Physical Test Suite"

check_prerequisites

if [[ ${#TESTS_TO_RUN[@]} -eq 0 ]]; then
    # Default to quick tests
    run_quick_tests
else
    for test in "${TESTS_TO_RUN[@]}"; do
        case $test in
            all)
                run_all_tests
                ;;
            quick)
                run_quick_tests
                ;;
            stress)
                run_stress_tests
                ;;
            failover)
                run_failover_test
                ;;
            impairment)
                run_impairment_tests
                ;;
            external)
                run_external_test
                ;;
            single:*)
                test_name="${test#single:}"
                run_cargo_test "test_$test_name" || run_cargo_test "$test_name"
                ;;
        esac
    done
fi

log_header "Testing Complete"
