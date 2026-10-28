# Contributing to Domyn Swarm

Contributions should keep Domyn Swarm predictable for users and maintainable for
contributors. Keep each pull request focused and explain the behavior it changes
or preserves.

## Principles

- Start from the user-facing behavior and make invalid states difficult to
  express or submit.
- Preserve existing behavior and compatibility unless the change explicitly
  introduces a migration.
- Prefer direct designs, existing abstractions, and domain-specific names.
  Generalize code only when current callers share a real rule.
- Keep code, tests, documentation, and configuration descriptions consistent.
- Confirm that copied or adapted code is compatible with the Apache-2.0 license,
  retain required notices, and credit its source.

## Development setup

Domyn Swarm supports Python `>=3.11,<3.15` on Linux. Install
[uv](https://docs.astral.sh/uv/), clone the repository, and synchronize the
development environment:

```bash
uv sync --all-extras
```

Documentation work also needs the docs dependency group:

```bash
uv sync --all-extras --group docs
```

Run project tools through `uv run`. When dependencies change intentionally,
update and commit `uv.lock`; otherwise leave it unchanged.

## Code and prose style

Ruff owns formatting and linting, and Pyright owns static type checking. The
repository-specific conventions are:

- Preserve public imports, serialized forms, CLI output, templates, and
  persistence formats unless the task explicitly changes them.
- Add Google-style docstrings to supported public APIs and non-obvious internal
  contracts. Do not add boilerplate docstrings to obvious private helpers,
  overrides, callbacks, validators, properties, or special methods.
- Comments are exceptional. Use one only when unusual code, a hidden invariant,
  an external constraint, a non-local side effect, or a non-obvious tradeoff
  would otherwise surprise a careful reader. Do not narrate straightforward
  code.
- Use concise active voice, US English, and sentence-case headings. Use “Domyn
  Swarm” for the project, `domyn-swarm` for the CLI and distribution, and
  `domyn_swarm` for the Python package.

Run the style checks with:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pyright
```

## Tests and documentation

Run the closest tests while iterating, then the full suite before review:

```bash
uv run pytest tests/path/to/test_file.py -q --no-cov
uv run pytest
```

Changes to user-facing behavior require corresponding documentation. Build the
documentation with warnings treated as errors:

```bash
docs_output=$(mktemp -d)
DOCS_VERSION=latest uv run sphinx-build -W --keep-going -b html docs "$docs_output"
```

Run every configured hook before submitting:

```bash
uv run pre-commit run --all-files
```

## Pull request guidelines

Open pull requests against `main` and keep each one centered on a single change.
Describe the motivation, observable effect, verification performed, and any
compatibility or migration considerations. Update tests and documentation where
the behavior requires them.

Use conventional commit messages. Every commit must be signed and signed off;
create commits with:

```bash
git commit -sS -m "type(scope): concise description"
```

See GitHub's documentation for
[configuring commit signing](https://docs.github.com/en/authentication/managing-commit-signature-verification/signing-commits).

## Developer Certificate of Origin

Signing off a commit certifies the following Developer Certificate of Origin.

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
1 Letterman Drive
Suite D4700
San Francisco, CA, 94129

Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

(b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

(c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

(d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
```
