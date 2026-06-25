# fusellm AGENTS.md

## Development commands
- Run benchmarks: `cd benchmarks && python run_benchmarks.py`
- Run a single scenario: edit `for scenario in range(1, 2):` at the bottom of run_benchmarks.py
- Install: `pip install -r requirements.txt`
- Test import: `python -c "from fusellm import merge_v2, merge_prod, utils"`

## GPU notes
- RTX 3050 4GB: use torch.float16, batch size 1, max sequence length 64-128
- Differential evolution (fusellm.llm_merge_solver) will OOM on 4GB
- All other methods fit in 4GB VRAM
