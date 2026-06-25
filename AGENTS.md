# fusellm AGENTS.md

## Development commands
- Run all tests: `pytest tests/ -v`
- Run quick tests (skip GPU): `pytest tests/ -v -m "not slow"`
- Run benchmarks: `cd benchmarks && python run_benchmarks.py`
- Install editable: `pip install -e .`
- Test import: `python -c "from fusellm import merge_prod, utils"`
- Build package: `python -m build`
- Publish to PyPI: `python -m twine upload dist/*`

## GPU notes
- RTX 3050 4GB: use torch.float16, batch size 1, max sequence length 64-128
