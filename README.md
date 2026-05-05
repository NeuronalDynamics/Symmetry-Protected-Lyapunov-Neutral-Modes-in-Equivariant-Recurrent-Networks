# Draft Asset Generation Code

This folder contains the single entry point for regenerating the figures and
tables used in the current Goldstone-Lyapunov draft.

## Quick Check

```powershell
python draft_asset_generation_code/run_all.py --check-only
```

## Generate Assets

```powershell
python draft_asset_generation_code/run_all.py
```

Generated files are written to:

```text
draft_asset_generation_code/generated_assets/
```

The default path regenerates the manuscript figures and tables from frozen raw
CSV/JSON outputs already present under `results/`. It does not rerun the
heaviest Lyapunov/GRU sweeps by default. The heavy rerun commands are recorded
in `rerun_commands.ps1`, and the relevant source scripts are copied into
`source_scripts/`.

## Main Outputs

- `generated_assets/figures/`: main and appendix figure PDFs/PNGs.
- `generated_assets/tables/`: table CSV and LaTeX files.
- `generated_assets/asset_manifest.csv`: figure/table-to-code/data mapping.

## Notes

- Main Figures 1-4 are generated from `results/journal_full` via the clean
  conference-fix plotting functions.
- Main Figure 5 is the learned equivariant path-integration figure generated
  from `results/learned_equivariant_pi`.
- Appendix Figures A9-A12 are copied from archived full-run PNGs by default
  because rerunning those experiments can be slow. Their rerun scripts are
  copied into `source_scripts/` and listed in `rerun_commands.ps1`.
"# Symmetry-Protected-Lyapunov-Neutral-Modes-in-Equivariant-Recurrent-Networks" 
