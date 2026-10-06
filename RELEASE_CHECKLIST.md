# Release checklist

## Repository
- [ ] Review the final diff.
- [ ] README, changelog, architecture, and qualification records match reality.
- [ ] Package/runtime versions agree.
- [ ] No generated reports, venvs, caches, or model files are staged.

## Automated validation
```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m compileall -q src tests
PYTHONPATH=src python -m devbits_bench --help
PYTHONPATH=src python -m devbits_bench --version
```

## Live smoke
- [ ] Ollama discovery works on a configured machine.
- [ ] MLX-LM discovery works from a valid environment.
- [ ] Engine and model selection work.
- [ ] Ctrl-C exits cleanly before inference.
- [ ] Previously qualified real benchmark evidence is retained.

Expensive Stress qualification does not need to be repeated unless methodology
or the engine execution path changes.

## Publish
- [ ] Commit the release candidate.
- [ ] Push and verify the remote commit.
- [ ] Tag `v0.2.0` only after the remote commit is verified.
- [ ] Use `CHANGELOG.md` as the basis for release notes.
