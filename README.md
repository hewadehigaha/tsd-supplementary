# Temporal Saliency Distillation (TSD)

Official PyTorch implementation and supplementary code for:

**Learning to Reason: Temporal Saliency Distillation for Interpretable Knowledge Transfer**

Nilushika Udayangani Hewa Dehigahawattage, Kishor Nandakishor, and Marimuthu Palaniswami

**ECAI 2025 — 28th European Conference on Artificial Intelligence**

## Overview

Knowledge distillation (KD) typically transfers knowledge from a large teacher model to a compact student model by matching logits or intermediate features. However, these forms of knowledge provide limited insight into *why* the teacher makes a particular prediction.

We propose **Temporal Saliency Distillation (TSD)**, a knowledge distillation approach for time-series classification that transfers not only the teacher's predictions but also its reasoning.

TSD derives **temporal saliency** from the teacher's predictive distribution by measuring its sensitivity to perturbations at individual time steps. The student is then trained to match the temporal saliency of the teacher, encouraging it to base its predictions on similar temporal features.

TSD:

- Transfers interpretable knowledge from teacher to student.
- Encourages the student to focus on the same temporal features as the teacher.
- Requires no additional network parameters.
- Makes no architecture-specific assumptions.
- Can be combined with existing knowledge distillation methods.
- Improves teacher-student fidelity and interpretability while maintaining competitive predictive performance.

## Method

For each input time series, TSD estimates the importance of individual time steps by measuring changes in the teacher's predictive distribution after perturbing the corresponding input regions.

The student is trained using the conventional task/distillation objective together with a temporal saliency matching objective:

```text
Teacher
   │
   ├── Prediction ───────────────► Knowledge Distillation
   │
   └── Temporal Saliency
              │
              ▼
Student ──► Match Teacher Saliency
```

This encourages the student to learn not only the teacher's output but also the temporal evidence underlying its predictions.

## Repository

This repository contains the implementation and supplementary material used for the experiments reported in the paper.

```text
tsd-supplementary/
├── ...
└── README.md
```

Please refer to the source files and experiment scripts in this repository for the corresponding experimental configurations.

## Datasets

The experiments use time-series classification datasets from the **UCR Time Series Classification Archive**.

Please download the required datasets from the UCR archive and configure the dataset path according to the experiment scripts.

## Requirements

The implementation is based on **Python** and **PyTorch**.

Install the required dependencies according to your Python environment. A `requirements.txt` file can be used to reproduce the environment:

```bash
pip install -r requirements.txt
```

## Running the Experiments

Clone the repository:

```bash
git clone https://github.com/hewadehigaha/tsd-supplementary.git
cd tsd-supplementary
```

Please refer to the provided experiment scripts for training the teacher and student models and reproducing the experiments reported in the paper.

## Paper

**Learning to Reason: Temporal Saliency Distillation for Interpretable Knowledge Transfer**

Proceedings of the 28th European Conference on Artificial Intelligence (ECAI 2025), pp. 2866–2873.

DOI: `10.3233/FAIA251144`

arXiv: `2601.04263`

## Citation

If you find this work useful in your research, please cite:

```bibtex
@inproceedings{dehigahawattage2025learning,
  title     = {Learning to Reason: Temporal Saliency Distillation for Interpretable Knowledge Transfer},
  author    = {Hewa Dehigahawattage, Nilushika Udayangani and Nandakishor, Kishor and Palaniswami, Marimuthu},
  booktitle = {Proceedings of the 28th European Conference on Artificial Intelligence (ECAI 2025)},
  pages     = {2866--2873},
  year      = {2025},
  doi       = {10.3233/FAIA251144}
}
```

## Authors

**Nilushika Udayangani Hewa Dehigahawattage**  
**Kishor Nandakishor**  
**Marimuthu Palaniswami**

Department of Electrical and Electronic Engineering  
The University of Melbourne

## Acknowledgements

This research was supported by The University of Melbourne's Research Computing Services and the Petascale Campus Initiative.

We also thank the authors and maintainers of the UCR Time Series Archive for making the datasets used in this research publicly available.