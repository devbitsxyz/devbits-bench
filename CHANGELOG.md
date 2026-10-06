# Changelog

## 0.2.0 — 2026-10-06

### Added
- Native **MLX-LM** engine alongside Ollama.
- Portable, non-mutating MLX runtime discovery and `DEVBITS_MLX_PYTHON`.
- MLX support across Quick, Practical, Stress, and Custom.
- Native MLX measurement provenance, reasoning controls, fresh-cache isolation,
  peak-memory reporting, and engine-aware request-context semantics.
- Stress memory-pressure advisories and cancellation-safe partial reporting.
- MLX-specific synthetic UI demos.

### Improved
- Engine-neutral benchmark architecture and lifecycle contract.
- Practical, Stress, and Custom progress rendering.
- Markdown/JSON provenance and memory reporting.
- Runtime diagnostics and public usage/troubleshooting documentation.

### Compatibility
Benchmark protocols remain versioned. Cross-engine results preserve qualified
engine-specific measurement and lifecycle semantics; equal labels do not imply
identical internal implementation.
