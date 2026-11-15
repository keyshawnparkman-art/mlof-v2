# Protected CI proposal

Offline CI runs on pull requests targeting main and pushes to main. The
offline-regression job runs the full standard-library suite with PYTHONPATH=src
on Ubuntu 24.04 and Python 3.14. The project declares Python 3.11 or newer.
The workflow needs no dependency install, repository secrets, or broker access.
It requests read-only repository contents and disables persisted checkout
credentials. Both actions are pinned to full release commit SHAs.

## Main ruleset

After separate owner authorization, publish the infrastructure branch and open
a pull request. Verify that GitHub runs offline-regression successfully and
select the exact reported check from GitHub Actions in the ruleset. Target main.
Require pull requests, passing offline-regression with the branch up to date,
and resolved review conversations. Block deletion and force pushes. Keep
auto-merge off and grant no automation bypass or merge authority.

Require one independent human approval and dismiss stale approvals where
another eligible reviewer is available. The pull request author cannot approve
their own change. The owner may independently review a worker-authored pull
request if GitHub recognizes them as a separate eligible reviewer. A worker
operating under the owner's GitHub identity does not provide that separation.
If no independent eligible reviewer is available, the owner must choose between
waiting for one and using zero required approvals with a manual owner review
and merge. The latter does not enforce independent review.

Optional CODEOWNERS entries for a later owner decision:

    /.github/ @keyshawnparkman-art
    /AGENTS.md @keyshawnparkman-art

Those entries request review when the owner is eligible. Require code-owner
approval only when an eligible code owner other than the author is available.
Only the owner accepts milestones. CI success and an automated review are not
milestone acceptance.

## Deferred email reporting

A later authorized change should deliver the daily report at 10 AM
America/Chicago, accounting for daylight saving time, and send an immediate
milestone-completion alert. Reports should distinguish completed,
pending-review, and owner-accepted milestones. Keep mail credentials in GitHub
Secrets. This proposal includes no notification workflow or credentials.
