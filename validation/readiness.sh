#!/bin/bash
# Wait for the paths an assertion actually requires, beyond first-path startup.
wait_client_paths() {
  local readiness_ns=$1 readiness_port=$2 readiness_count=$3 readiness_out=$4
  local readiness_deadline=$((SECONDS + 10))
  while [[ $SECONDS -lt $readiness_deadline ]]; do
    if ip netns exec "$readiness_ns" curl --max-time 1 --silent --fail \
      "http://127.0.0.1:$readiness_port/status" >"$readiness_out" && \
      python3 -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x["state"] == "connected"; assert x["quality"]["usable_uplinks"] == int(sys.argv[2])' \
      "$readiness_out" "$readiness_count" >/dev/null 2>&1; then
      return 0
    fi
    sleep .05
  done
  echo "Client did not establish $readiness_count usable paths" >&2
  cat "$readiness_out" >&2 || true
  return 1
}
