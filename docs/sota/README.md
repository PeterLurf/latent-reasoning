# Current-architecture page

`sota_arch.json` is the source of truth for the live page
https://claude.ai/artifact/Hz4pVzYnuSrZFbNzgMZuTn (owned by the Audit session).

When a scored result changes the best model or its scorecard (a new SOTA, or a C-status change on the current one):
1. The scoring session (Theory) edits `sota_arch.json` in the same commit as the scoring Note.
2. It messages Audit "SOTA page: <section>".
3. Audit reproduces the score, then republishes the page with the new json.

Keep the fields; add new criteria rows rather than renaming keys. Write with newline="\n" (repo is LF).
