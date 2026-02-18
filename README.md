# Multivariate Decoding Analysis for Neural Replay

Python implementation of the multivariate decoding algorithm from:

**Liu, Y., Dolan, R. J., Kurth-Nelson, Z., & Behrens, T. E. (2019). Human replay spontaneously reorganizes experience. *Cell*, 178(3), 640-652.**

[https://doi.org/10.1016/j.cell.2019.06.012](https://doi.org/10.1016/j.cell.2019.06.012)

## Overview

This implementation detects sequential replay of neural representations during rest periods using MEG/EEG data. The algorithm can identify:

- **Forward replay**: Sequences playing in the experienced order (A→B→C→D)
- **Reverse replay**: Sequences playing backward, typically after reward (D→C→B→A)
- **Sequence-specific replay**: Detection of which particular sequence is replaying
- **Time-compressed replay**: Sequences that occur faster than real-time experience

## Key Features

- **Lasso-regularized logistic regression** for state decoding
- **Time-lagged regression** to compute sequenceness measure
- **Alpha oscillation control** (10Hz nuisance regressors)
- **Permutation testing** for statistical validation
- **Comprehensive visualization** tools

## Installation

### Requirements

```bash
pip install numpy scipy scikit-learn matplotlib seaborn jupyter
```

### Python Version
Python 3.7 or higher

## Quick Start

### 1. Basic Usage

```python
from multivariate_replay_decoder import MultivariateReplayDecoder
import numpy as np

# Initialize decoder
decoder = MultivariateReplayDecoder(n_states=4, max_lag_ms=600, sampling_rate=100)

# Train classifiers on functional localizer data
# X_train: (n_trials, n_timepoints, n_sensors)
# y_train: (n_trials,) with labels 0 to n_states-1
decoder.train_classifiers(X_train, y_train, time_point_ms=200)

# Decode resting state data
# X_rest: (n_timepoints, n_sensors)
reactivation_probs = decoder.decode_states(X_rest)

# Define transition matrix (A→B→C→D)
transition_matrix = np.array([
    [0, 1, 0, 0],  # A → B
    [0, 0, 1, 0],  # B → C
    [0, 0, 0, 1],  # C → D
    [0, 0, 0, 0]   # D → nothing
])

# Compute sequenceness
sequenceness, time_lags = decoder.compute_sequenceness(
    reactivation_probs, transition_matrix
)

# Statistical testing
p_values, threshold, true_seq = decoder.permutation_test(
    reactivation_probs, transition_matrix, n_permutations=1000
)
```

### 2. Interactive Examples

See the comprehensive Jupyter notebook with multiple examples:

```bash
jupyter notebook multivariate_replay_analysis.ipynb
```

The notebook includes:
- **Example 1**: Forward replay detection
- **Example 2**: Reverse replay after reward
- **Example 3**: Multiple sequence detection

## Algorithm Details

### Step 1: Train State Decoders

Train binary classifiers to recognize neural patterns associated with each state/stimulus:

```python
decoder.train_classifiers(X_train, y_train, time_point_ms=200, C=1.0)
```

- Uses L1-regularized (lasso) logistic regression
- One binary classifier per state
- Trains on specific time point (typically 200ms post-stimulus)
- Feature standardization applied automatically

### Step 2: Decode Resting State

Apply trained classifiers to rest periods:

```python
reactivation_probs = decoder.decode_states(X_rest)
```

Returns probability time series for each state (n_timepoints × n_states).

### Step 3: Compute Sequenceness

Quantify sequential structure using time-lagged regression:

```python
sequenceness, time_lags = decoder.compute_sequenceness(
    reactivation_probs, transition_matrix, alpha_control=True
)
```

For each time lag Δt:
1. Regress time-shifted activations X(Δt) onto current activations Y
2. Obtain regression coefficient matrix β(Δt)
3. Project onto hypothesized transition matrix P
4. Include nuisance regressors for 10Hz oscillations (if alpha_control=True)

**Sequenceness = Forward - Backward**

### Step 4: Statistical Testing

```python
p_values, threshold, true_seq = decoder.permutation_test(
    reactivation_probs, transition_matrix, n_permutations=1000
)
```

- Permutes stimulus labels to create null distribution
- Corrects for multiple comparisons across time lags
- Returns significance threshold at p < 0.05

## File Structure

```
DecodeTest/
├── README.md                              # This file
├── multivariate_replay_analysis.ipynb     # Interactive examples
├── multivariate_replay_decoder.py         # Main implementation (optional)
├── Liu et al. - 2019 - Human replay...pdf # Original paper
└── .gitignore                             # Git ignore rules
```

## Data Format

### Training Data (Functional Localizer)

```python
X_train: ndarray, shape (n_trials, n_timepoints, n_sensors)
    Neural activity during stimulus presentations

y_train: ndarray, shape (n_trials,)
    Stimulus labels (integers from 0 to n_states-1)
```

### Resting State Data

```python
X_rest: ndarray, shape (n_timepoints, n_sensors)
    Continuous neural activity during rest periods
```

### Transition Matrix

```python
transition_matrix: ndarray, shape (n_states, n_states)
    Binary matrix where element [i,j] = 1 if state i → state j

Example for A→B→C→D:
[[0, 1, 0, 0],
 [0, 0, 1, 0],
 [0, 0, 0, 1],
 [0, 0, 0, 0]]
```

## Key Findings from the Paper

1. **Forward Replay**: Sequences play forward during learning/exploration
2. **Reverse Replay**: Sequences reverse direction after reward
3. **Structural Reorganization**: Replay follows inferred (not just experienced) order
4. **Factorized Representations**: Abstract position/sequence codes precede stimulus codes by ~50ms
5. **Hippocampal Ripples**: Replay coincides with sharp-wave ripples (120-150 Hz)

## Parameters

### MultivariateReplayDecoder

```python
MultivariateReplayDecoder(
    n_states=8,           # Number of distinct states/stimuli
    max_lag_ms=600,       # Maximum time lag to test (ms)
    sampling_rate=100     # Sampling rate in Hz
)
```

### train_classifiers

```python
train_classifiers(
    X_train,              # Training data
    y_train,              # Labels
    time_point_ms=200,    # Time point for training (ms post-stimulus)
    C=1.0                 # Inverse regularization strength
)
```

### compute_sequenceness

```python
compute_sequenceness(
    reactivation_probs,   # Decoded state probabilities
    transition_matrix,    # Hypothesized transitions
    time_lags_ms=None,    # Specific lags to test (default: 10-600ms)
    alpha_control=True    # Include 10Hz nuisance regressors
)
```

### permutation_test

```python
permutation_test(
    reactivation_probs,   # Decoded state probabilities
    transition_matrix,    # Hypothesized transitions
    n_permutations=1000,  # Number of permutations
    time_lags_ms=None     # Specific lags to test
)
```

## Visualization

The implementation includes visualization functions:

```python
# Plot sequenceness with significance threshold
fig = decoder.plot_sequenceness(sequenceness, time_lags, threshold)
plt.show()
```

## Expected Results

With properly simulated or real data containing replay at 40ms lag:

- **Peak sequenceness**: Positive for forward, negative for reverse
- **Peak timing**: Around 40-50ms (as reported in the paper)
- **Significance**: Should exceed permutation threshold if replay is present

## Tips for Real Data

1. **Preprocessing**:
   - High-pass filter at 0.5 Hz
   - Reject artifacts
   - Standardize sensor data

2. **Classifier Training**:
   - Use sufficient trials (>20 per state)
   - Cross-validate regularization parameter
   - Check decoding accuracy (should be above chance)

3. **Rest Period**:
   - Minimum 2-5 minutes recommended
   - Eyes closed typically better (less artifacts)
   - Check for sufficient replay events

4. **Interpretation**:
   - Consider task structure
   - Look at both forward and reverse
   - Compare before/after reward or learning

## Limitations

- Requires good signal-to-noise ratio in training data
- Sensitive to classifier performance
- Assumes discrete states/stimuli
- May miss very fast or very slow sequences

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{liu2019human,
  title={Human replay spontaneously reorganizes experience},
  author={Liu, Yunzhe and Dolan, Raymond J and Kurth-Nelson, Zeb and Behrens, Timothy EJ},
  journal={Cell},
  volume={178},
  number={3},
  pages={640--652},
  year={2019},
  publisher={Elsevier}
}
```

## License

This implementation is provided for research and educational purposes.

## Contact

For questions about this implementation, please open an issue on GitHub.

For questions about the original method, see the paper or contact the authors.

## Acknowledgments

Implementation based on the methods described in:
- Liu et al. (2019) Cell
- Original MATLAB code concepts from the paper's supplementary materials

## Related Work

- **Kurth-Nelson et al. (2016)** - Fast sequences of non-spatial state representations in humans
- **Foster & Wilson (2006)** - Reverse replay of behavioral sequences in hippocampal place cells
- **Ambrose et al. (2016)** - Reverse replay uniquely modulated by changing reward

---

**Note**: This is a research implementation. For production use, additional validation and optimization may be required.
