# AI PR Review

InfiniOps uses `/ai-pr-review` as an opt-in GitHub Actions slash command. The review follows the repository root `REVIEW.md`, posts advisory findings on the pull request, and does not replace maintainer review, required approvals, or CI.

## How it works

A repository owner, member, or collaborator posts this comment on a pull request:

```text
/ai-pr-review
```

The command can also be rerun manually:

```text
/ai-pr-review --recheck
```

For every invocation, the workflow:

1. Adds an `eyes` reaction to the triggering comment.
2. Collects PR metadata, changed files, the diff, commits, issue comments, review comments, reviews, and check summaries through the GitHub REST API.
3. Stores an immutable context artifact and records the PR head SHA observed at trigger time.
4. Runs a read-only Codex review using the trusted review policy from the repository default branch and `REVIEW.md`.
5. Posts **Blocking findings**, **Non-blocking findings**, and **Evidence gaps** as a PR comment.
6. Changes the triggering comment's reaction to `rocket` after successfully publishing the review.

If the workflow itself fails, it posts an infrastructure-failure comment. That failure is not a code-review conclusion.

## Required repository configuration

### 1. Enable GitHub Actions

In the repository settings:

```text
Settings → Actions → General
```

make sure Actions are enabled.

### 2. Add the InfiniPlan API key

Add an Actions secret named:

```text
INFINIPLAN_API_KEY
```

in:

```text
Settings → Secrets and variables → Actions → Actions secrets
```

This secret is required. The reviewer cannot start without it.

### 3. Check the Actions allowlist

If the organization or repository permits only selected actions, allow at least:

```text
actions/checkout
actions/upload-artifact
actions/download-artifact
actions/setup-node
openai/codex-action
```

No allowlist changes are needed when the repository permits all GitHub Actions.

### 3. Configure InfiniPlan Responses API settings

Add a repository variable named:

```text
INFINIPLAN_RESPONSES_ENDPOINT
```

with the full InfiniPlan Responses API URL, for example:

```text
https://infiniplan.example.com/v1/responses
```

Add another repository variable for the Codex-compatible model name:

```text
INFINIPLAN_MODEL
```

The exact value depends on the models exposed by InfiniPlan. The endpoint must implement the OpenAI-compatible Responses API; a Chat Completions-only endpoint is not sufficient for Codex. These values may be repository variables, or masked secrets if the endpoint/model configuration is private.

Optionally add a repository variable named:

```text
INFINIPLAN_EFFORT
```

to configure Codex reasoning effort. For broad compatibility, use one of `none`, `minimal`, `low`, `medium`, `high`, or `xhigh`; the selected model and InfiniPlan endpoint must also support the value. Leave the variable unset to use the workflow default, `high`.

### 4. Workflow permissions

If an organization policy caps the `GITHUB_TOKEN` at read-only access and prevents job-level write grants, grant this workflow permission to write pull-request and issue comments. The repository-wide default can otherwise remain read-only.

The repository-wide default can remain read-only. The workflow requests only the permissions needed by each job:

```text
contents: read
pull-requests: write
issues: write
checks: read
```

for context capture, and:

```text
pull-requests: write
issues: write
```

for publishing the review comment.

The setting **Allow GitHub Actions to create and approve pull requests** is not needed. The workflow only creates comments and reactions; it does not open or approve pull requests.

## Temporary-branch testing limitation

An `issue_comment` workflow is loaded from the repository default branch, not from the PR branch.

Therefore, after putting this workflow on a temporary feature branch:

- a comment of `/ai-pr-review` on that feature-branch PR normally does **not** run the new workflow version;
- the slash command becomes available only after the workflow is merged into the default branch;
- after merging, test it on an existing open PR.

If needed, the temporary-branch PR can still validate Python formatting, workflow lint, YAML syntax, and the normal repository CI before merge.

## Security boundary

- The workflow, collector, and prompt come from the trusted default branch.
- Pull-request text and code are treated as untrusted review data.
- PR-controlled text is never interpolated into a shell command.
- PR files are checked out separately under `pr-head/` for reading only.
- The reviewer does not run contributor-suggested commands, repository scripts, builds, or tests.
- The InfiniPlan API key is stored only as a GitHub Actions secret and is not printed.
- Each job receives the minimum GitHub token permissions required.
- The result is advisory; the workflow never submits a request-changes review.

## Phase 1 limitations

- Reviews are posted as one summary comment, not GitHub inline review comments.
- New pushes do not automatically trigger another review; rerun the command when needed.
- `--recheck` performs a full review of the current head. It does not yet remember human false-positive corrections.
- REST API context cannot determine exact review-thread resolution; comments are provided as supporting context.
- Inline findings, a false-positive feedback loop, and incremental re-reviews are planned for later phases.
