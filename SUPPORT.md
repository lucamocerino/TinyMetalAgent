# Support

TinyMetalAgent is early-stage software. The best support channel depends on the request:

- **Bug reports**: open a GitHub issue with the bug report template.
- **Feature requests**: open a GitHub issue with the feature request template.
- **Security reports**: follow `SECURITY.md`; do not file public issues.
- **Questions and experiments**: use GitHub Discussions if enabled, otherwise open a question issue.

When asking for help, include:

- macOS version and Apple Silicon chip.
- `git rev-parse --short HEAD`.
- The command you ran.
- Whether you built with `make -C c all` and ran `make -C c test`.
- Model filename, quantization, and source if the issue needs a model.
- Any relevant `TINYENGINE_*` or `TINYAGENT_*` environment variables.
