# Security Policy

TinyAgent can inspect files, write files, execute shell commands, and optionally use network tools. Treat it like any local coding agent with access to your workspace.

## Supported versions

Security fixes target the current `main` branch until formal releases begin. Once versioned releases exist, this policy will list supported release lines.

## Reporting a vulnerability

Do not open public issues for security vulnerabilities.

Until a project security contact is published, report privately to the repository owner through GitHub. If GitHub Security Advisories are enabled for the repository, use "Report a vulnerability" from the Security tab.

Please include:

- Affected commit or version.
- Reproduction steps.
- Impact and affected files, commands, or model inputs.
- Whether the issue requires TinyAgent tool approval, `--yes`, network access, or a malicious repository/model.

## Security boundaries

TinyEngine is a local inference runtime. TinyAgent is an agent layer that can operate on the local filesystem and shell when approved.

Important boundaries:

- Do not run TinyAgent with `--yes` in untrusted repositories.
- Prefer `--dry-run` or the default approval mode when evaluating unknown prompts.
- Use `--no-network` when a task should stay offline.
- Review generated file edits before committing.
- Do not point TinyEngine at untrusted GGUF files unless you are comfortable testing native model parsers against that input.

## Vulnerability classes of interest

Reports are especially useful for:

- Unsafe parsing of malformed GGUF metadata or tensor data.
- Command injection or path traversal in TinyAgent tools.
- Tool approval bypasses.
- Accidental network access when `--no-network` is set.
- Secret exposure through logs, sessions, benchmark artifacts, or tool output.
- Memory-safety issues in the C/Metal runtime.
