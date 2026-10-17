#!/bin/bash
# Reproducible root-only network tests in a disposable Linux container.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:-"$ROOT/validation/artifacts/$(date -u +%Y%m%dT%H%M%SZ)"}
if [[ -d "$OUT" && -n $(ls -A "$OUT") ]]; then
  echo "Evidence directory must be empty: $OUT" >&2
  exit 1
fi
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
name="triglav-proof-$(date +%s)-$$"
cleanup() { docker rm -f "$name" >/dev/null 2>&1 || true; }
trap cleanup EXIT
# No repository build context is sent to Docker (in particular, no target/).
docker build -t triglav-validation:local - < "$ROOT/validation/Dockerfile" >"$OUT/image-build.log" 2>&1
docker image inspect triglav-validation:local >"$OUT/image.json"
docker run --name "$name" --privileged \
  --mount "type=bind,src=$ROOT,dst=/source,readonly" \
  --mount "type=bind,src=$OUT,dst=/evidence" \
  --mount type=volume,src=triglav-validation-cargo,dst=/usr/local/cargo \
  --mount type=volume,src=triglav-validation-build,dst=/build \
  -e TRIGLAV_EVIDENCE_DIR=/evidence triglav-validation:local bash -c '
    set -euo pipefail
    date -u +%Y-%m-%dT%H:%M:%SZ > /evidence/started-at.txt
    uname -a > /evidence/platform.txt
    rustc -Vv > /evidence/rustc.txt
    python3 - <<"PY"
import pathlib
paths = {pathlib.Path("Cargo.toml"), pathlib.Path("Cargo.lock"), pathlib.Path("README.md"), pathlib.Path(".github/workflows/validate.yml")}
for directory, suffixes in [("src", {".rs"}), ("tests", {".rs"}), ("physical_tests", {".sh"})]:
    paths.update(p for p in pathlib.Path(directory).rglob("*") if p.suffix in suffixes)
paths.update(p for p in pathlib.Path("validation").iterdir() if p.is_file() and p.name != "REPORT.md")
pathlib.Path("/evidence/source-files.txt").write_text("\n".join(str(p) for p in sorted(paths))+"\n")
PY
    xargs -d "\n" sha256sum < /evidence/source-files.txt > /evidence/source-sha256.txt
    tar -czf /evidence/source.tar.gz -T /evidence/source-files.txt
    cargo build --locked --bins > /evidence/build.log 2>&1
    # Stress tests enforce throughput floors; measure optimized code on CI.
    cargo test --locked --release --all-targets -- --test-threads=1 > /evidence/tests.log 2>&1
    cargo build --locked --release --bins > /evidence/release-build.log 2>&1
    mkdir /evidence/bin
    cp /build/release/triglav /build/release/triglav-server /evidence/bin/
    # An existing namespace is not owned by the harness and must survive rejection.
    ip netns add tgproof-client
    if TRIGLAV_EVIDENCE_DIR=/evidence/preexisting timeout 30 bash validation/netns.sh > /evidence/preexisting.log 2>&1; then exit 1; fi
    grep -q "Namespace already exists: tgproof-client" /evidence/preexisting.log
    ip netns list | grep -q "^tgproof-client\( \|$\)"
    ip netns del tgproof-client
    sha256sum /build/release/triglav /build/release/triglav-server > /evidence/binary-sha256.txt
    TRIGLAV_BIN_DIR=/build/release timeout 600 bash validation/netns.sh > /evidence/network.log 2>&1
    TRIGLAV_BIN_DIR=/build/release timeout 180 bash validation/outer_ipv6.sh > /evidence/outer-ipv6.log 2>&1
    TRIGLAV_BIN_DIR=/build/release timeout 300 bash validation/recovery.sh > /evidence/recovery.log 2>&1
    sha256sum -c /evidence/source-sha256.txt > /evidence/source-verification.log
    sha256sum /evidence/source.tar.gz /evidence/bin/* > /evidence/archive-sha256.txt
    python3 validation/report.py /evidence > /evidence/report.log
    date -u +%Y-%m-%dT%H:%M:%SZ > /evidence/completed-at.txt
  '
cat "$OUT/network.log"
printf 'Evidence: %s\n' "$OUT"
