# Project instructions

- Build a provider-neutral Pi package for proactive semantic memory.
- Preserve global/project filtering and full provenance.
- Keep memories terse; deduplicate every write and prefer update/no-op.
- Assistant memories require ≥60s non-trivial investigation, independent validation, capped confidence, and lower priority than user memories.
- Keep background recall bounded, coalesced, clearly labelled, and non-mandatory.
- Run `npm run check` after changes.
