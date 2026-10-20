# rustyred-thg-query

Shared OpenCypher parser, read-gate, executor, dialect lists, and query receipts
for Theorem's HTTP and MCP query surfaces. One pest grammar is the gate; MCP
handlers and the server HTTP Cypher door both consume this crate so the parser
cannot drift.
