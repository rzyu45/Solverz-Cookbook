# Solverz Cookbook Development Guidelines

## Project Overview

Solverz Cookbook is the example collection for [Solverz](https://github.com/smallbunnies/Solverz). Each example demonstrates how to model and solve a specific problem using Solverz.

Documentation is hosted at https://cook.solverz.org/

## Project Structure

Each example follows this layout:

```
docs/source/{category}/{example}/
├── {example}.md           # Documentation page (MyST markdown)
├── __init__.py
├── src/
│   ├── __init__.py
│   ├── {example}_mdl.py   # Modelling script (shown via literalinclude)
│   ├── test_{example}.py  # Tests (pytest, uses datadir fixture)
│   ├── plot_{example}.py  # Figure generation (optional)
│   └── {data_dir}/        # Test data (.mat files, etc.)
```

Categories: `ae/` (algebraic), `dae/` (differential-algebraic), `fdae/` (finite-difference DAE)

## Development Conventions

- Run tests with: `pytest docs/source/`
- Tests use the `datadir` pytest fixture for test data paths
- Documentation uses MyST markdown with `{literalinclude}` for code blocks
- Each modelling script should be self-contained and runnable
- Test scripts cross-validate results against known benchmarks
- Use conventional commit format: `feat:`, `fix:`, `docs:`, `test:`

## Dependencies

- [Solverz](https://github.com/smallbunnies/Solverz) — core simulation language
- [SolMuseum](https://github.com/smallbunnies/SolMuseum) — model library (required by some examples)
- [SolUtil](https://github.com/smallbunnies/SolUtil) — utility functions (required by some examples)
