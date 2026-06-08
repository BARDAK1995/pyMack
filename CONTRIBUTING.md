# Contributing to pyMack

Thank you for your interest in pyMack! Contributions, bug reports, and suggestions are welcome.

## Setting up a development environment

```bash
git clone https://github.com/BARDAK1995/pyMack.git
cd pyMack

# Install in editable mode with development dependencies
pip install -e ".[dev]"
```

## Running the tests

The main test suite lives in the `validation/` directory:

```bash
# Run the full validation suite (quiet mode)
pytest validation/ -q

# Run with more output
pytest validation/ --tb=short
```

Individual diagnostic scripts can also be run directly (e.g. `python validation/diagnose_low_mid_table_10_1_shooting.py`).

## How to contribute

1. Fork the repository and create a feature branch from `master`.
2. Make your changes.
3. Add or update tests if applicable.
4. Run the test suite and make sure it passes.
5. Open a Pull Request.

We use a relatively relaxed style — focus on correctness and clarity. If you're unsure about something, open an issue or draft PR and ask.

## Reporting bugs

Please use the GitHub issue tracker and include:
- Python version and OS
- Exact commands you ran
- Full error traceback (if any)
- A minimal example that reproduces the problem

## Citation

If you use pyMack in work that leads to a publication, please cite it (see the main README for details).

## Questions?

Feel free to open an issue with the "question" label or start a discussion.
