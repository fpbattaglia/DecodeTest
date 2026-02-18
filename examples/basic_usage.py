#!/usr/bin/env python
"""
Basic usage example for replay_decoder package.

This script demonstrates:
1. Creating simulated data
2. Training the decoder
3. Detecting replay
4. Visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from replay_decoder import (
    MultivariateReplayDecoder,
    create_linear_transition_matrix,
    simulate_replay_data,
    plot_sequenceness
)

# Set random seed for reproducibility
np.random.seed(42)

print("="*60)
print("Replay Decoder - Basic Usage Example")
print("="*60)

# 1. Create simulated training data
print("\n1. Creating simulated training data...")
n_trials = 200
n_timepoints_per_trial = 50  # 500ms at 100Hz
n_sensors = 100
n_states = 4
sampling_rate = 100

X_train = np.random.randn(n_trials, n_timepoints_per_trial, n_sensors) * 0.5
y_train = np.random.randint(0, n_states, n_trials)

# Add state-specific signal at 200ms
time_idx = 20
for trial in range(n_trials):
    state = y_train[trial]
    X_train[trial, time_idx, state*10:(state+1)*10] += 2.0

print(f"   Training data shape: {X_train.shape}")
print(f"   Number of states: {n_states}")

# 2. Initialize and train decoder
print("\n2. Training decoder...")
decoder = MultivariateReplayDecoder(
    n_states=n_states,
    max_lag_ms=600,
    sampling_rate=sampling_rate
)

decoder.train_classifiers(X_train, y_train, time_point_ms=200)
print("   Decoder trained successfully!")

# 3. Simulate resting state with replay
print("\n3. Simulating resting state with embedded replay...")
X_rest, replay_times = simulate_replay_data(
    n_states=n_states,
    n_sensors=n_sensors,
    n_rest_timepoints=3000,
    lag_samples=4,  # 40ms lag
    sequence=[0, 1, 2, 3],  # A→B→C→D
    sampling_rate=sampling_rate
)

print(f"   Rest data shape: {X_rest.shape}")
print(f"   Embedded {len(replay_times)} replay events")

# 4. Decode states
print("\n4. Decoding states from rest data...")
reactivation_probs = decoder.decode_states(X_rest)
print(f"   Reactivation matrix shape: {reactivation_probs.shape}")

# 5. Define transition matrix
print("\n5. Creating transition matrix...")
transition_matrix = create_linear_transition_matrix(n_states)
print("   Transition matrix (A→B→C→D):")
print(transition_matrix)

# 6. Compute sequenceness
print("\n6. Computing sequenceness...")
sequenceness, time_lags = decoder.compute_sequenceness(
    reactivation_probs,
    transition_matrix,
    alpha_control=True
)

# Find peak
peak_idx = np.argmax(np.abs(sequenceness))
peak_lag = time_lags[peak_idx]
peak_value = sequenceness[peak_idx]

print(f"   Peak sequenceness: {peak_value:.4f}")
print(f"   Peak lag: {peak_lag}ms")
print(f"   Expected lag: 40ms")

# 7. Statistical testing
print("\n7. Running permutation test...")
p_values, threshold, true_seq = decoder.permutation_test(
    reactivation_probs,
    transition_matrix,
    n_permutations=500
)

significant_lags = time_lags[p_values < 0.05]
print(f"   Significance threshold: {threshold:.4f}")
print(f"   Number of significant lags: {len(significant_lags)}")
if len(significant_lags) > 0:
    print(f"   Significant lags: {significant_lags}ms")

# 8. Visualize results
print("\n8. Creating visualization...")
fig, ax = plot_sequenceness(sequenceness, time_lags, threshold)
plt.axvline(x=40, color='green', linestyle=':', linewidth=2,
           alpha=0.7, label='True lag (40ms)')
ax.legend()
plt.savefig('basic_usage_result.png', dpi=150, bbox_inches='tight')
print("   Saved visualization to 'basic_usage_result.png'")

print("\n" + "="*60)
print("Analysis complete!")
print("="*60)

# Show plot
plt.show()
