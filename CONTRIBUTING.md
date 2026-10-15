# Contributing to seizure-pipeline

Thanks for considering a contribution. This project is small and
actively maintained by its two authors; contributions of any size are
welcome, from a typo fix to a new feature.

## Reporting a bug or requesting a feature

Open an issue on GitHub. For a bug report, please include:

- Python version and OS
- The exact command or code that triggered the problem
- The full error traceback
- What you expected to happen instead

## Submitting a change

1. Fork the repository and create a branch from `main`.
2. Make your change. Keep commits focused: one logical change per commit.
3. Add or update tests in `test_seizure_pipeline.py` for any behavior
   change. Run the test suite locally before opening a pull request:
```bash
   pytest test_seizure_pipeline.py -v
   pytest test_seizure_pipeline.py --runslow   # include the larger, slower checks
```
4. Update `README.md` if your change affects usage.
5. Open a pull request against `main`, describing what changed and why.

## Code style

- Follow the existing style in the file you're editing (type hints,
  docstrings on public functions, no unexplained magic numbers).
- Prefer small, well-documented functions over large ones.

## Support expectations

This is a research-software project maintained alongside the authors'
own work, not a funded, full-time effort. We aim to respond to issues
and pull requests within a couple of weeks, but response time may vary
around teaching and research deadlines. If something is urgent, say so
in the issue and we'll try to prioritize it.

## Code of conduct

Be respectful and constructive. Disagreements about technical
approach are fine and expected; personal attacks are not.
