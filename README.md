# PULSE-UCB

This repository is for reproducing the paper "Learning with Incomplete Context: Linear Contextual Bandits with Pretrained Imputation", published at AISTATS 2026. The method proposed in the paper is PULSE-UCB.

## Contents
- [Project Layout](#project-layout)
- [Environment Setup](#environment-setup)
- [Data Paths](#data-paths)
- [Reproduction (Paper Order)](#reproduction-paper-order)
- [Quick Run Scripts](#quick-run-scripts)
- [Output Locations](#output-locations)

## Project Layout
- `experiments/synthetic`: synthetic experiments
- `experiments/real`: real-data pipeline (preprocess -> embedding -> pretrain -> evaluation)
- `outputs`: generated plots and result files
- `configs/real_taobao.yaml`: default config template

## Environment Setup
```bash
pip install torch numpy pandas scikit-learn statsmodels matplotlib seaborn numba tqdm
```

## Real Data Paths
Real dataset (taobao): https://www.kaggle.com/datasets/pavansanagapati/ad-displayclick-data-on-taobaocom

Default (repository-relative):
- raw data: `data/taobao/`
- processed data: `data/taobao/preprocess/`

Optional environment overrides:
- `NCB_DATASET` (default: `taobao`)
- `NCB_DATASET_PATH` (default: `data/<dataset>`)
- `NCB_BASE_DATA_PATH` (default: `data/<dataset>/preprocess`)

## Reproduction (Paper Order)
| # | Target | Script / Pipeline | Main Output | Notes |
|---|---|---|---|---|
| 1 | Figure 1: Synthetic comparison (3-agent default) | `python experiments/synthetic/compare_b_o.py` | `outputs/synthetic/Comparison_linear.png`, `outputs/synthetic/Comparison_nonlinear.png` | Top linear, bottom nonlinear, 30 trials |
| 1* | Figure 1 variant: 4-agent + CLBBF | `python experiments/synthetic/compare4UCB.py` | same comparison figures | Kept as comparison extension |
| 2 | Figure 2: Taobao comparison | see full real pipeline below | `outputs/real/final_comparison_plot.png` | 5 seeds with error bands |
| 3 | Figure 3: Different linearity (`rho=0.1,1,10`) | `python experiments/synthetic/diff_eta.py` | `outputs/synthetic/eta_performance_analysis.png` | Interactive prompt: choose `2` for full analysis |
| 4 | Tables 1A/1B: sensitivity under varying `N`, `T0` | `python experiments/synthetic/compare_b_o+.py` | `outputs/synthetic/Sensitivity_N_T_Analysis.png` | Tables printed in terminal |
| 5 | Tables 1C/1D: 4-agent comparison (linear/nonlinear) | `python experiments/synthetic/compare4UCB.py` | comparison figures reused | Regret tables printed in terminal |
| 6 | Table 2: random feature masking robustness | `python experiments/synthetic/mask.py` | terminal results | `p` from `0.0` to `0.8` |
| 7 | Table 3: covariate variance shift (4-agent) | `python experiments/synthetic/dist_shift4.py` | `outputs/synthetic/Covariate_Shift_Analysis.png` | Table printed in terminal |
| 8 | Table 4: structural shift (rotation) | `python experiments/synthetic/dist_shift_high.py` | `outputs/synthetic/Structural_Shift_Analysis.png` | Table printed in terminal |
| 8* | Table 5/6: mean shift and geometric shift | `python experiments/synthetic/dist_shift_all.py` | terminal results | Translation and deformation |

### Real Pipeline (for Figure 2)
```bash
python experiments/real/preprocess.py
python experiments/real/autoencoder_train.py --seed 0
python experiments/real/generate_embeddings.py
python experiments/real/prepare_final_data.py
python experiments/real/pretrain_inference_model.py
python experiments/real/run_online_evaluation.py
```

## Quick Run Scripts
Synthetic:
```powershell
.\scripts\run_synthetic.ps1 -Mode single
.\scripts\run_synthetic.ps1 -Mode single -Script compare4UCB.py
.\scripts\run_synthetic.ps1 -Mode core
```

Real:
```powershell
.\scripts\run_real.ps1 -Dataset taobao -DataRoot data -Seed 0
```

## Output Locations
- synthetic outputs: `outputs/synthetic/`
- real-data outputs: `outputs/real/`

