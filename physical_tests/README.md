# Physical interfaces and production validation

The production acceptance command is:

```sh
./physical_tests/e2e_validation.sh --local
# Equivalent:
./validation/run.sh
```

It builds and runs the shipped client and exit binaries, creates two isolated
Linux uplinks with separate routers and source NATs, and checks packet integrity,
single exit identity, failover, rejection controls and concurrent clients.
The [validation contract](../validation/README.md) defines its assertions,
assumptions, artifacts and deployment requirements. Failed gates return a
nonzero exit status. The old remote-server, external-httpbin and duration flags
are rejected; those scripts did not establish the production forwarding path.

`run_physical_tests.sh` and the ignored tests in `tests/physical_multipath.rs`
remain diagnostic tools for local interface/socket behavior. Several of those
tests return early when hardware or privileges are absent. Their green result
is not evidence that two physical uplinks or the production exit were tested.

## Additional proof on physical hardware

A namespace result exercises real kernel networking and production code, but
cannot establish independent physical NIC, modem or ISP behavior. A physical
validation record needs all of the following:

- Client interface names, source addresses, routes and interface binding.
- The exact tested source/binary hashes and server configuration.
- Packet captures or per-path data counters establishing payload traffic on
  each interface, beyond handshake and keepalive traffic.
- An independent destination's observed peer address and exact bidirectional
  payload comparison, including concurrent flows.
- Observations while each physical path is independently unavailable, followed
  by recovery; existing inner TCP connections must be distinguished from newly
  opened connections.
- Cleanup and restoration of the test routes/interfaces.

The server and client must share `TRIGLAV_ACCESS_TOKEN` when admission control is
enabled. The printed `tg1_...` key identifies the server and its address; it does
not include that separate token. Follow the deployment commands in the
validation contract. Use a reachable server address when generating the key;
`0.0.0.0` is a bind wildcard, not a remote endpoint.
