# Security policy

## Supported versions

Security fixes are applied to the `main` branch and to the latest release on PyPI. Older releases do not receive backports.

## Reporting a vulnerability

Please do not open a public issue for security problems.

Use GitHub private vulnerability reporting instead:
<https://github.com/DiogoRibeiro7/subspaceknn/security/advisories/new>

You can expect an acknowledgement within seven days. Once the report is confirmed, a fix and a coordinated disclosure date are agreed with the reporter before any advisory is published.

## Scope

`subspaceknn` is a pure-Python library depending on numpy and scikit-learn. Its public API validates every input and raises typed errors rather than failing unpredictably.

The following are treated as security-relevant and welcome through the private channel:

- unbounded resource use reachable through the public API with attacker-controlled input beyond what `max_candidates` documents;
- vulnerabilities in the dependency tree that affect this package.

Accuracy problems, surprising predictions or explanation semantics are correctness bugs, not vulnerabilities. Please report them as ordinary issues with a minimal reproduction.
