# Release Process

TinyMetalAgent does not have formal published releases yet. Use this process when cutting the first release.

## Versioning

- Use semantic versioning for Python package versions and source tags.
- Keep `python/tinyagent/__init__.py` and `pyproject.toml` aligned.
- Bump `TE_ABI_VERSION` only when the public C ABI changes.

## Pre-release checklist

1. Update `CHANGELOG.md`.
2. Run:

   ```bash
   make -C c clean all
   make -C c test
   python3 -m compileall -q python
   python3 -m pip install -e .
   TINYENGINE_LIBRARY=$PWD/c/build/libtinyengine.dylib python3 -m tinyagent --help
   ```

3. Run model-backed oracle/benchmark checks for any runtime, tokenizer, or kernel release.
4. Confirm no GGUF models, logs, local benchmark scratch files, or secrets are staged.
5. Tag the release:

   ```bash
   git tag -a v0.1.0 -m "TinyMetalAgent v0.1.0"
   git push origin v0.1.0
   ```

6. Publish release notes with supported platform, model setup, known limitations, and benchmark evidence.
