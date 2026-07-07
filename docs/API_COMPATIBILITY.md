# kenkui HTTP API compatibility policy

kenkui exposes an HTTP API (FastAPI, see `kenkui.server.api`) consumed by
clients such as kengui. Clients do not bundle a kenkui sidecar on every
platform (mobile packaging talks only to remote servers of uncontrolled
versions), so the client and server versions can differ at runtime. This
document defines the compatibility contract that makes that safe.

## Versioning

The API carries two version signals, both returned by the `/health`
(and `/v1/health`) endpoint:

- **`api_version`** — the API contract major version, currently `v1`. This is
  the value clients use for the compatibility handshake. It changes only on a
  breaking change (see below).
- **`server_version`** — the kenkui package version (e.g. `2.2.1`), for
  diagnostics/telemetry. It is not used for compatibility decisions.

The build-time contract (the OpenAPI schema dumped by
`python -m kenkui.server.openapi`) is the source of truth for client type
generation. Clients generate their API types against a pinned kenkui version
and record the `api_version` they were built for.

## Compatibility rules

Within a single `api_version` major, changes are **additive-only**:

- Allowed without a major bump: adding new endpoints; adding new **optional**
  request fields; adding new response fields; adding new enum members in
  response-only positions; adding new capabilities to the `capabilities` list.
- Requires an `api_version` major bump: removing or renaming an endpoint,
  request field, or response field; making an optional request field required;
  changing a field's type or semantics; removing a capability.

A client built for `api_version` major *N* is expected to work against any
server advertising major *N*. Servers must not ship breaking changes without
incrementing the major.

## Runtime handshake

On connect, a client fetches `/health`, reads `api_version`, and compares its
major against the version it was built for:

- **Equal major** → compatible; no action.
- **Server major higher** → the server speaks a newer API; the client may not
  understand some responses. Warn the user (non-blocking) and suggest updating
  the client.
- **Server major lower** → the server is older than the client expects; some
  features may be unavailable. Warn the user (non-blocking) and suggest
  updating the server.

The handshake is a warning, not a hard gate: additive-only evolution means a
same-major mismatch of *minor* details is always safe, and cross-major use
degrades rather than fails outright.

## Status as of this writing (2026-07)

`api_version` is `v1` and no API versions exist in the wild outside
development. Breaking changes are therefore currently free; this policy takes
effect from the first published release, at which point `v1` becomes frozen
under the additive-only rule and any breaking change moves to `v2`.
