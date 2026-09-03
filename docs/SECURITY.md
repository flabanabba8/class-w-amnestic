# Security

This runtime executes model-chosen actions. It is designed to be *inspectable and
bounded*, not to be safe for unattended execution with elevated privileges. Do not run it
as root, on machines holding secrets you cannot afford to expose, or with
`MNESTIC_SHELL_MODE=unrestricted` outside a disposable sandbox.

## Boundaries implemented

| concern | mitigation |
|---|---|
| workspace confinement | `ToolContext.resolve_path` resolves symlinks and rejects any path outside the workspace root (`allow_workspace_escape=False` by default) |
| path traversal | `..`, absolute paths and symlinks are resolved before the check; shell arguments that look like paths are checked too |
| shell execution | `disabled` \| `allowlist` (default: argv prefix allowlist, no shell interpretation, metacharacters rejected, workspace cwd, timeout, output cap) \| `unrestricted` (explicit opt-in) |
| environment secrets | subprocesses receive a minimal environment (`PATH, HOME, LANG, LC_ALL, TERM, TZ`) unless `pass_environment=True` |
| model output | every decision is Pydantic-validated with `extra="forbid"`; unknown ops, wrong types, oversized fields, unknown evidence ids, forbidden statuses are rejected before anything is executed |
| state destruction | patches are atomic; no op replaces the whole state; every version is recoverable |
| resource use | step budgets, per-invocation `--max-steps`, consecutive-failure and continue-loop guards, state byte/count limits, bounded observations, bounded retrieval, tool timeouts, output caps, file-size caps |
| SQL | all queries are parameterised; FTS terms are tokenised and quoted |
| deserialisation | JSON only (no pickle); all rows are re-validated through Pydantic models on read; skills are re-hashed on load |
| large outputs | never inlined into state; archived and excerpted |

## Known gaps (do not overstate)

- **Prompt injection**: file contents, tool output, human input and retrieved archive
  events are untrusted text placed inside the model prompt. The runtime labels them by
  section, but a model can still be steered into bad-but-allowed actions (e.g. writing a
  misleading report, calling allowlisted commands). Structural limits — not the prompt —
  are the real boundary.
- **Archive poisoning**: whatever a tool returned is what retrieval returns later. The
  archive is faithful, not sanitised.
- `allowlist` shell mode checks argv prefixes; an allowlisted binary with dangerous flags
  (e.g. `find -exec`) is still allowed. Tighten `allowed_commands` for your environment.
- `unrestricted` shell mode is a full shell as the current user.
- `write_workspace_file` can overwrite any file inside the workspace, including skills or
  the repo itself if the workspace is the repo.
- No network sandboxing; tools that reach the network are not included in v1.
- No secrets scanning on tool output before archival.
- The SQLite file is unencrypted and contains every observation.

## Recommendations

Run in a container or VM with a throwaway workspace; keep the default allowlist shell or
disable it; keep the SQLite database outside the workspace (the default
`.mnestic/` directory is skipped by `search_text`, but a workspace-level tool can still
read it); review `events`/`inspect-context` before trusting a run's report.
