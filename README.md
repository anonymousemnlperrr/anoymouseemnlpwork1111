# anonymous-contact-local-policy

Anonymous architecture-focused release for the main contact-local policy and its data/evaluation pipeline.

Included:
- core tactile encoder
- language-guided bottleneck and tactile-language alignment modules
- contact-local residual routing logic
- mainline contact-local policy implementation
- dataset loading and split definitions
- dataset preprocessing and annotation scripts
- core evaluation scripts for representation, instruction sensitivity, and summary tables

Excluded on purpose:
- checkpoints and trained weights
- configs and machine-specific paths
- comparison studies and ablations
- external baseline bridges and result artifacts
- paper drafts, logs, and dataset contents

## Notes

This release contains the main model, the dataset-processing path, and the core evaluation code.
It still excludes trained weights, raw data, configs, and comparison-only experiment packaging.

## Minimal dependencies

- numpy
- pandas
- pillow
- pyyaml
- pyarrow
- torch
- transformers
- huggingface_hub

## Optional dependencies

- av
- scikit-learn
- matplotlib
