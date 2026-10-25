# MIFS-BDA

Official repository for **"Multimodal Information Fusion Strategies for Biomedical Data Analysis"** (SoMeT 2026).

![Architecture](figures/Figure_1.png)

## Taxonomy

| Level | Strategy | Fusion Point | Missing Modality Robustness |
|---|---|---|---|
| 1 | Early | Input concatenation | Poor |
| 2 | Intermediate | Feature-level | Moderate |
| 3 | Late | Decision-level | Good |
| 4 | Attention | Cross-modal weighting | Moderate |
| 5 | Adaptive | Dynamic routing | Excellent |

## Results

Evaluation on [VQA-RAD](https://osf.io/89kps/) (closed-ended subset):

| Strategy | Model | F1 | BalAcc | Acc |
|---|---|---|---|---|
| Baseline | ImageOnly | 0.561 | 0.561 | 0.561 |
| Baseline | TextOnly | 0.628 | 0.630 | 0.631 |
| Early | EarlyFusion | 0.672 | 0.672 | 0.673 |
| Intermediate | DeepFusion | 0.654 | 0.654 | 0.654 |
| Intermediate | LMF | 0.643 | 0.646 | 0.645 |
| Late | LateFusion | 0.672 | 0.674 | 0.673 |
| Attention | CrossAttn | 0.628 | 0.642 | 0.640 |
| Attention | GAF | 0.640 | 0.640 | 0.640 |
| Adaptive | **HybridFusion** | **0.691** | **0.692** | **0.692** |

Robustness under image patch masking (HybridFusion):

| Mask Ratio | F1 | BalAcc | Acc |
|---|---|---|---|
| 0.0 | 0.655 | 0.658 | 0.659 |
| 0.1 | **0.679** | **0.681** | **0.682** |
| 0.5 | 0.636 | 0.636 | 0.636 |
| 0.9 | 0.650 | 0.650 | 0.650 |

## Repository Structure

```
├── requirements.txt
├── LICENSE
├── README.md
├── src/
│   ├── models/
│   │   ├── encoders.py
│   │   ├── fusion.py
│   │   └── __init__.py
│   ├── dataset.py
│   └── run_experiment.py
├── figures/
│   ├── Figure_1.png
│   ├── draw_radar.py
│   ├── draw_performance.py
│   ├── radar_scores.csv
│   └── performance_data.csv
└── data/
    ├── table_vqarad.csv
    └── table_masking.csv
```

## Reproduction

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download [VQA-RAD](https://osf.io/89kps/) to `data/VQA_RAD/`:
```
data/VQA_RAD/
├── VQA_RAD Dataset Public.json
└── VQA_RAD_Image_Folder/
```

3. Run experiments:
```bash
python src/run_experiment.py --data_dir data/VQA_RAD --mask_experiment
```

4. Generate figures:
```bash
python figures/draw_radar.py
python figures/draw_performance.py
```

## Citation

```bibtex
@inproceedings{nguyen2026multimodal,
  title     = {Multimodal Information Fusion Strategies for Biomedical Data Analysis},
  author    = {Nguyen, Ba-Duy and Hoang, Van-Dung and Vu, Sang and Huynh, Anh T},
  booktitle = {New Trends in Intelligent Software Methodologies, Tools and Techniques:
               Proceedings of the 25th International Conference on New Trends in
               Intelligent Software Methodologies, Tools and Techniques (SoMeT\_26)},
  pages     = {130--139},
  year      = {2026},
  organization = {SAGE Publications},
  doi       = {10.3233/FAIA260732}
}
```

## License

MIT License. See [LICENSE](LICENSE).
