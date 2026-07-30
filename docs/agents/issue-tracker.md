# Issue tracker: GitHub

Issues and PRDs for this repo live as GitHub issues in `LCJD99/eec-sched`. Use the `gh` CLI for all operations.

## Conventions

- **Create an issue**: `gh issue create --repo LCJD99/eec-sched --title "..." --body "..."`. Use a heredoc for multi-line bodies.
- **Read an issue**: `gh issue view <number> --repo LCJD99/eec-sched --comments`, filtering comments by `jq` and also fetching labels.
- **List issues**: `gh issue list --repo LCJD99/eec-sched --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'` with appropriate `--label` and `--state` filters.
- **Comment on an issue**: `gh issue comment <number> --repo LCJD99/eec-sched --body "..."`
- **Apply / remove labels**: `gh issue edit <number> --repo LCJD99/eec-sched --add-label "..."` / `--remove-label "..."`
- **Close**: `gh issue close <number> --repo LCJD99/eec-sched --comment "..."`

## Pull requests as a triage surface

**PRs as a request surface: no.** _(Set to `yes` if this repo treats external PRs as feature requests; `/triage` reads this flag.)_

When set to `yes`, PRs run through the same labels and states as issues, using the `gh pr` equivalents:

- **Read a PR**: `gh pr view <number> --repo LCJD99/eec-sched --comments` and `gh pr diff <number> --repo LCJD99/eec-sched`.
- **List external PRs for triage**: use `gh pr list --repo LCJD99/eec-sched --state open --json number,title,body,labels,author,authorAssociation,comments`, then retain only `CONTRIBUTOR`, `FIRST_TIME_CONTRIBUTOR`, or `NONE` author associations.
- **Comment / label / close**: use `gh pr comment`, `gh pr edit`, and `gh pr close` with `--repo LCJD99/eec-sched`.

GitHub shares one number space across issues and PRs. Resolve an ambiguous `#42` with `gh pr view 42 --repo LCJD99/eec-sched`, falling back to `gh issue view 42 --repo LCJD99/eec-sched`.

## When a skill says “publish to the issue tracker”

Create a GitHub issue in `LCJD99/eec-sched`.

## When a skill says “fetch the relevant ticket”

Run `gh issue view <number> --repo LCJD99/eec-sched --comments`.

## Wayfinding operations

Used by `/wayfinder`. The **map** is a single issue with **child** issues as tickets.

- **Map**: a single issue labelled `wayfinder:map`, holding the Notes / Decisions-so-far / Fog body.
- **Child ticket**: an issue linked to the map as a GitHub sub-issue. Where sub-issues are unavailable, add it to a task list in the map body and put `Part of #<map>` at the top of the child body. Labels use `wayfinder:<type>` (`research`, `prototype`, `grilling`, or `task`).
- **Blocking**: use GitHub’s native issue dependencies. Where unavailable, put a `Blocked by: #<n>, #<n>` line at the top of the child body.
- **Frontier query**: list the map’s open children, dropping assigned tickets and tickets with open blockers; the first in map order wins.
- **Claim**: `gh issue edit <n> --repo LCJD99/eec-sched --add-assignee @me`.
- **Resolve**: comment with the answer, close the ticket, and append a context pointer to the map’s Decisions-so-far.
