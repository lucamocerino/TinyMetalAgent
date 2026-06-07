# Changelog

All notable user-facing changes should be recorded here.

This project follows semantic versioning once formal releases begin. Before `1.0.0`, minor versions may include breaking changes.

## Unreleased

- Added open-source project scaffolding: contribution, support, security, release, model setup, benchmark policy, CI, and packaging metadata.
- Standardized TinyAgent setup/docs on Qwen2.5-Coder 3B Q4_0 as the release target.
- Reduced TinyAgent prompt-budgeting overhead by avoiding duplicate token counts and skipping tool schemas in chat-only mode.
- Added an optional local model daemon (`--daemon`) to reuse a loaded TinyEngine model across CLI invocations.
- Reduced ACT-mode prompt size with compact tool schemas and cached stable tool/system prompt prefixes.
- Defaulted TinyAgent context sizing to the detected hardware recommendation and made history compaction respect available context.
- Added clearer terminal progress UI for model loading, prompt rendering, agent steps, tool calls, results, and timings.
- Fixed REPL greetings/questions being routed through ACT tool prompts, which could overflow small context windows and crash.
- Added dynamic ACT tool windows so small-context runs include only task-relevant tools and preserve the original task across tool-protocol retries.
- Added REPL slash-command autocomplete and moved detailed status/timing/tool output behind `--verbose`.
- Added modern terminal-agent REPL affordances: `?` quick help, `!` shell escape, `@file` mentions, `/context`, and `/diff`.
- Added a deterministic Matrix integration test that drives real agent tools to create a pure-Python Matrix class, write unittest coverage, and run it.
- Added an original SVG repository logo focused on local AI learning for students.
- Added an SVG terminal demo screen to the README.
- Added a README disclosure that the repository was developed with substantial GPT-5.5 assistance.
- Added a PNG fallback for the README logo to improve GitHub rendering reliability.
- Added `install.sh` for one-command local build, test, editable install, CLI check, and optional model download.
