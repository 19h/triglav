FROM rust:1-trixie
RUN apt-get update && apt-get install -y --no-install-recommends \
    iproute2 iptables conntrack tcpdump python3 curl iputils-ping ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ENV CARGO_TARGET_DIR=/build
WORKDIR /source
