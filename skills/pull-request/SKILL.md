---
name: pull-request
description: Take a change to speech.cpp from a branch to main. Covers the worktree and branch, the checks a change runs, commit messages, the pull request, one Codex review, CI, stacked pull requests, the squash merge and its order, cleaning up, and taking back a subagent's branch. Use when starting a change, committing, pushing, opening, reviewing or merging a pull request, stacking one on another, or waiting for CI (PRを出す、レビュー、CIを待つ、マージ、PR を積む、片づけ). Not for tagging a release (release) or for checks on a Windows machine (windows-check).
---

# From a branch to main

Every change reaches main through a pull request; nobody commits on main. A pull request is one coherent unit, and
may be large: a model's options, a feature with its page and docs, a set of fixes in one area.

## 1. Start

- Work in a worktree of its own, on a branch from `origin/main`:
  `git worktree add -b <branch> ../speech.cpp-wt/<name> origin/main`. Link the shared `models/` folder and, if the
  change runs dumps, `reference/<family>/out` and `.venv`; never commit the links.
- Read the ADRs whose names cover the behaviour you change before changing it.

## 2. Check and commit

- Build and run the checks the change touches; AGENTS.md's Workflow names them (stage checks, `speech-api-check`, the
  smoke scripts with a model of each family). A defect fix comes with a check that fails before it.
- A defect or a small fix found on the way goes into this pull request when it touches the same place, or into a small
  pull request of its own; it is not left for later.
- A rebase onto `origin/main` that changes no file a check reads does not call for running that check again; a long
  check runs again only when what it reads moved.
- Commit as each part works. The message is one English sentence in the imperative without a prefix, then what
  changed and why, then the checks run and what they gave ("Checked: …"), as earlier commits write them.
- A decision the code cannot show gets a record in `docs/adr/` in the same pull request (adr skill).

## 3. Open the pull request

`git push -u origin <branch>` and `gh pr create --base main --title "<one sentence>" --body-file <file>`. The body says
what changed and why, any behaviour that changes for existing users, what was checked and on what, and what was not.
A branch whose Windows or Linux code was not compiled locally can open as a draft first, so that CI compiles it. A pull
request that moves the ggml submodule pushes the fork's branch first, as the ggml-fork skill says.

## 4. One review

Run Codex once on the branch: `codex review --base origin/main -c model='"gpt-6-astra"'`. A run that stops without a
review, such as one flagged as a possible risk, is run again; that is not a second review. Read every finding; fix the
real ones where their cause is, each with a check that fails without the fix where one can be written. Do not run the
review again after fixing. Record anything checked outside CI (a Windows run, a page tried in a browser) as a comment on
the pull request.

## 5. CI and merge

- `gh pr checks <n> --watch` until every job ends, started in the background. Right after a push it may report no
  checks yet; wait until they appear (`until gh pr checks <n> | grep -q .; do sleep 10; done`) before watching. A
  failure is read from its log (`gh run view <run> --log-failed`) and fixed on the branch.
- A pull request that changes only Markdown files, `docs/` or `skills/` runs no CI (`paths-ignore` in
  `.github/workflows/build.yml`), so `gh pr checks` never lists a check for it: do not wait for one. It is ready to merge
  once the review's findings are fixed.
- Merge with a squash once CI passes, the review's findings are fixed and any Windows check is recorded on the pull
  request, when the maintainer has said you may merge: `gh pr merge <n> --squash --delete-branch`, the pull requests
  others depend on first. Otherwise leave it for the maintainer.

## 6. Clean up

`git worktree remove --force <path>`, `git branch -D <branch>`, and confirm with `git worktree list`. Remove temporary
branches pushed for CI runs (`git push origin --delete <branch>`).

## Stacked pull requests

- A pull request may build on another's branch when it needs that change: open it with `--base <that branch>` and say
  in its body which pull request it builds on.
- Before merging the base, retarget every pull request on it to main (`gh pr edit <n> --base main`): `--delete-branch`
  deletes the merged branch, and that closes every pull request whose base it is.
- Note each dependent's own base when it is opened, the commit it was built on:
  `git merge-base <dependent> <base branch>`, taken before the base branch changes. In a stack A, B, C, C's base is
  the commit of B it was built on, which is no longer B's tip once B is rebased.
- After the base merges with a squash, its commits are not on main. `git fetch origin` and check that `origin/main`
  holds the squash merge (`git log origin/main` lists it); a rebase onto an `origin/main` from before the merge drops
  the base's changes. Then bring the dependent onto main with `git rebase --onto origin/main <its base>` and push with
  `--force-with-lease`. Whenever a branch is rebased, rebase every branch stacked on it the same way, onto its new tip
  from the base noted for it, so that no merged commit is replayed.

## A subagent's branch

A subagent commits on its own branch and never pushes or merges; the coordinator pushes. Read its report, then its diff
yourself (`git diff origin/main...<branch>`), and check that the diff is what the report says before you push it. Send
fixes back to the same subagent rather than finishing its work in its worktree. When main moves under it, have it
rebase onto `origin/main`.
