# Security policy

## Supported versions

Security fixes go into the latest release only.

## Reporting a vulnerability

Please don't report security problems in public issues. Use GitHub's
[private vulnerability reporting](https://github.com/thorstenkranz/cnnsim/security/advisories/new)
instead (*Security → Report a vulnerability*). You'll get an answer within a few days.

cnnsim is a numerical library without network access or file parsing of its own; the most likely
issues are in its dependencies (PyTorch, Triton), which should be reported to those projects.
