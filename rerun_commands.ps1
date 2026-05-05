# Heavy/Full Rerun Commands

# Main clean figures from frozen journal tables:
python draft_asset_generation_code/run_all.py

# Learned experiment full rerun:
python -m goldstone_lyapunov.experiments.exp31_learned_equivariant_path_integration --full

# Stronger baseline check full rerun:
python -m goldstone_lyapunov.experiments.exp32_stronger_path_integration_baselines --full

# Heavy appendix figures, if raw full figures need to be regenerated:
python -m goldstone_lyapunov.experiments.exp25_finite_time_chaos_diagnostics
python -m goldstone_lyapunov.experiments.exp26_large_chaotic_spectra
python -m goldstone_lyapunov.experiments.exp21_path_integration_heatmaps
python -m goldstone_lyapunov.experiments.exp29_gru_path_integration_sweep
