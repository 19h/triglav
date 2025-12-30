#!/bin/bash
# Chaos Agent - Dynamically modifies network conditions to simulate real-world scenarios

set -e

SCENARIO=${SCENARIO:-realistic_internet}
SCENARIO_FILE="/scenarios/${SCENARIO}.sh"

echo "=========================================="
echo "Triglav Chaos Agent"
echo "=========================================="
echo "Scenario: $SCENARIO"
echo "=========================================="

# Function to modify impairment on a container
modify_impairment() {
    local container=$1
    local interface=$2
    local delay=$3
    local jitter=$4
    local loss=$5
    
    echo "[$container] Setting: delay=$delay jitter=$jitter loss=$loss"
    
    docker exec $container sh -c "
        tc qdisc del dev $interface root 2>/dev/null || true
        tc qdisc add dev $interface root netem delay $delay $jitter distribution normal loss random $loss
    " 2>/dev/null || true
}

# Function to simulate complete link failure
link_down() {
    local container=$1
    local interface=$2
    local duration=$3
    
    echo "[$container] Link DOWN for ${duration}s"
    docker exec $container ip link set $interface down
    sleep $duration
    docker exec $container ip link set $interface up
    echo "[$container] Link UP"
}

# Function to simulate congestion
simulate_congestion() {
    local container=$1
    local interface=$2
    local rate=$3
    local duration=$4
    
    echo "[$container] Congestion: limiting to $rate for ${duration}s"
    docker exec $container sh -c "
        tc qdisc del dev $interface root 2>/dev/null || true
        tc qdisc add dev $interface root tbf rate $rate burst 32kbit latency 100ms
    "
    sleep $duration
    docker exec $container sh -c "tc qdisc del dev $interface root" 2>/dev/null || true
}

# Realistic internet scenario
run_realistic_internet() {
    echo "Running realistic internet scenario..."
    
    while true; do
        # Random event selection
        event=$((RANDOM % 100))
        
        if [ $event -lt 5 ]; then
            # 5% chance: WiFi interference (high jitter spike)
            echo "EVENT: WiFi interference"
            modify_impairment "wifi-router" "eth1" "50ms" "30ms" "5%"
            sleep $((10 + RANDOM % 20))
            modify_impairment "wifi-router" "eth1" "20ms" "5ms" "1%"
            
        elif [ $event -lt 10 ]; then
            # 5% chance: LTE cell handover (brief packet loss)
            echo "EVENT: LTE handover"
            modify_impairment "lte-gateway" "eth1" "100ms" "50ms" "10%"
            sleep 2
            modify_impairment "lte-gateway" "eth1" "30ms" "15ms" "2%"
            
        elif [ $event -lt 12 ]; then
            # 2% chance: ISP congestion
            echo "EVENT: ISP-A congestion"
            simulate_congestion "isp-a" "eth1" "10mbit" $((30 + RANDOM % 60))
            
        elif [ $event -lt 13 ]; then
            # 1% chance: WiFi dropout (brief disconnection)
            echo "EVENT: WiFi dropout"
            link_down "wifi-router" "eth1" $((2 + RANDOM % 5))
            
        elif [ $event -lt 14 ]; then
            # 1% chance: Mobile network brownout
            echo "EVENT: Mobile network brownout"
            modify_impairment "mobile-core" "eth1" "200ms" "100ms" "20%"
            sleep $((5 + RANDOM % 10))
            modify_impairment "mobile-core" "eth1" "20ms" "5ms" "0.5%"
            
        elif [ $event -lt 20 ]; then
            # 6% chance: Background traffic on Ethernet (slight congestion)
            echo "EVENT: Ethernet background traffic"
            modify_impairment "ethernet-router" "eth1" "5ms" "2ms" "0.1%"
            sleep $((5 + RANDOM % 15))
            modify_impairment "ethernet-router" "eth1" "2ms" "0.5ms" "0.01%"
            
        else
            # 80% chance: Normal conditions
            echo "STATUS: Normal conditions"
        fi
        
        # Wait before next event (5-30 seconds)
        sleep $((5 + RANDOM % 25))
    done
}

# Heavy impairment scenario (stress test)
run_stress_test() {
    echo "Running stress test scenario..."
    
    while true; do
        event=$((RANDOM % 10))
        
        case $event in
            0|1)
                # Ethernet failure
                echo "EVENT: Ethernet failure"
                link_down "ethernet-router" "eth1" $((10 + RANDOM % 20))
                ;;
            2|3)
                # WiFi failure
                echo "EVENT: WiFi failure"
                link_down "wifi-router" "eth1" $((10 + RANDOM % 20))
                ;;
            4|5)
                # LTE failure
                echo "EVENT: LTE failure"
                link_down "lte-gateway" "eth1" $((10 + RANDOM % 20))
                ;;
            6)
                # All ISPs congested
                echo "EVENT: Internet congestion"
                simulate_congestion "isp-a" "eth1" "1mbit" 30 &
                simulate_congestion "isp-b" "eth1" "1mbit" 30 &
                simulate_congestion "mobile-core" "eth1" "1mbit" 30 &
                wait
                ;;
            7|8|9)
                # Extreme jitter
                echo "EVENT: Network instability"
                modify_impairment "wifi-router" "eth1" "100ms" "80ms" "10%"
                modify_impairment "lte-gateway" "eth1" "200ms" "150ms" "15%"
                sleep $((10 + RANDOM % 20))
                modify_impairment "wifi-router" "eth1" "20ms" "5ms" "1%"
                modify_impairment "lte-gateway" "eth1" "30ms" "15ms" "2%"
                ;;
        esac
        
        sleep $((10 + RANDOM % 20))
    done
}

# Load and run scenario
case $SCENARIO in
    realistic_internet)
        run_realistic_internet
        ;;
    stress_test)
        run_stress_test
        ;;
    none)
        echo "No chaos scenario running (static network)"
        tail -f /dev/null
        ;;
    *)
        if [ -f "$SCENARIO_FILE" ]; then
            source "$SCENARIO_FILE"
        else
            echo "Unknown scenario: $SCENARIO"
            echo "Available: realistic_internet, stress_test, none"
            exit 1
        fi
        ;;
esac
