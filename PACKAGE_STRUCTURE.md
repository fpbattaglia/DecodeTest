# Package Structure

## Created Python Package: `replay_decoder`

This document describes the structure of the replay_decoder Python package.

## Directory Structure

```
DecodeTest/
├── README.md                              # Main documentation
├── LICENSE                                # MIT License
├── MANIFEST.in                            # Package manifest
├── setup.py                               # Setup script (legacy)
├── pyproject.toml                         # Modern Python package config
├── .gitignore                             # Git ignore rules
│
├── replay_decoder/                        # Main package directory
│   ├── __init__.py                        # Package initialization
│   ├── decoder.py                         # Core MultivariateReplayDecoder class
│   ├── visualization.py                   # Plotting functions
│   └── utils.py                           # Utility functions
│
├── examples/                              # Example scripts
│   └── basic_usage.py                     # Basic usage demonstration
│
├── multivariate_replay_analysis.ipynb     # Jupyter notebook with examples
│
└── Liu et al. - 2019 - Human replay...pdf # Original paper
```

## Package Modules

### `replay_decoder.decoder`

Main decoder class:
- `MultivariateReplayDecoder` - Core class for replay analysis

### `replay_decoder.visualization`

Plotting functions:
- `plot_sequenceness()` - Plot sequenceness vs time lag
- `plot_reactivations()` - Plot state reactivation time series
- `plot_transition_matrix()` - Visualize transition matrices
- `plot_comparison()` - Compare multiple sequenceness curves

### `replay_decoder.utils`

Helper functions:
- `create_linear_transition_matrix()` - Create A→B→C→D matrix
- `create_circular_transition_matrix()` - Create circular transitions
- `simulate_replay_data()` - Generate simulated data with replay
- `find_replay_events()` - Identify replay events
- `validate_training_data()` - Data format validation
- `validate_rest_data()` - Rest data validation

## Installation

### From source (development mode):

```bash
cd DecodeTest
pip install -e .
```

### With optional dependencies:

```bash
# For development (includes testing tools)
pip install -e ".[dev]"

# For Jupyter notebooks
pip install -e ".[notebook]"
```

## Usage Example

```python
from replay_decoder import (
    MultivariateReplayDecoder,
    create_linear_transition_matrix,
    plot_sequenceness
)
import numpy as np

# Initialize decoder
decoder = MultivariateReplayDecoder(n_states=4, sampling_rate=100)

# Train on functional localizer data
decoder.train_classifiers(X_train, y_train, time_point_ms=200)

# Decode rest data
reactivations = decoder.decode_states(X_rest)

# Define sequence structure
transition_matrix = create_linear_transition_matrix(n_states=4)

# Compute sequenceness
sequenceness, time_lags = decoder.compute_sequenceness(
    reactivations, transition_matrix
)

# Statistical testing
p_values, threshold, _ = decoder.permutation_test(
    reactivations, transition_matrix, n_permutations=1000
)

# Visualize
plot_sequenceness(sequenceness, time_lags, threshold)
```

## Running Examples

### Basic Python script:

```bash
python examples/basic_usage.py
```

### Jupyter notebook:

```bash
jupyter notebook multivariate_replay_analysis.ipynb
```

## Testing

(Tests not yet implemented, but structure is ready)

```bash
pip install -e ".[dev]"
pytest tests/
```

## Key Features

1. **Modular Design**: Separate modules for decoding, visualization, and utilities
2. **Well-Documented**: Comprehensive docstrings in NumPy format
3. **Type Hints**: (Can be added in future versions)
4. **Easy Installation**: Standard Python packaging with setup.py and pyproject.toml
5. **Examples Included**: Both script and notebook examples
6. **Extensible**: Easy to add new features or custom transition matrices

## Package Metadata

- **Name**: replay_decoder
- **Version**: 0.1.0
- **License**: MIT
- **Python**: ≥3.7
- **Dependencies**: numpy, scipy, scikit-learn, matplotlib

## Citation

If you use this package, please cite the original paper:

Liu, Y., Dolan, R. J., Kurth-Nelson, Z., & Behrens, T. E. (2019).
Human replay spontaneously reorganizes experience. Cell, 178(3), 640-652.
https://doi.org/10.1016/j.cell.2019.06.012
