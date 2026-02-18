"""
Utility functions for replay analysis.

This module provides helper functions for data validation,
preprocessing, and common operations.
"""

import numpy as np


def validate_training_data(X_train, y_train, n_states):
    """
    Validate training data format and dimensions.

    Parameters
    ----------
    X_train : ndarray
        Training data, expected shape (n_trials, n_timepoints, n_sensors)
    y_train : ndarray
        Labels, expected shape (n_trials,)
    n_states : int
        Number of expected states

    Raises
    ------
    ValueError
        If data format is incorrect

    Returns
    -------
    bool
        True if validation passes
    """
    if X_train.ndim != 3:
        raise ValueError(f"X_train must be 3D (trials, timepoints, sensors), got {X_train.ndim}D")

    if y_train.ndim != 1:
        raise ValueError(f"y_train must be 1D, got {y_train.ndim}D")

    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(f"Number of trials mismatch: X_train {X_train.shape[0]}, y_train {y_train.shape[0]}")

    if not np.all((y_train >= 0) & (y_train < n_states)):
        raise ValueError(f"y_train contains invalid labels (should be 0 to {n_states-1})")

    return True


def validate_rest_data(X_rest, n_sensors):
    """
    Validate resting state data format.

    Parameters
    ----------
    X_rest : ndarray
        Resting data, expected shape (n_timepoints, n_sensors)
    n_sensors : int
        Expected number of sensors

    Raises
    ------
    ValueError
        If data format is incorrect

    Returns
    -------
    bool
        True if validation passes
    """
    if X_rest.ndim != 2:
        raise ValueError(f"X_rest must be 2D (timepoints, sensors), got {X_rest.ndim}D")

    if X_rest.shape[1] != n_sensors:
        raise ValueError(f"Sensor count mismatch: expected {n_sensors}, got {X_rest.shape[1]}")

    return True


def create_linear_transition_matrix(n_states):
    """
    Create a simple linear transition matrix (0→1→2→...→n-1).

    Parameters
    ----------
    n_states : int
        Number of states

    Returns
    -------
    transition_matrix : ndarray, shape (n_states, n_states)
        Binary transition matrix
    """
    transition_matrix = np.zeros((n_states, n_states))
    for i in range(n_states - 1):
        transition_matrix[i, i + 1] = 1
    return transition_matrix


def create_circular_transition_matrix(n_states):
    """
    Create a circular transition matrix (0→1→2→...→n-1→0).

    Parameters
    ----------
    n_states : int
        Number of states

    Returns
    -------
    transition_matrix : ndarray, shape (n_states, n_states)
        Binary transition matrix
    """
    transition_matrix = np.zeros((n_states, n_states))
    for i in range(n_states):
        transition_matrix[i, (i + 1) % n_states] = 1
    return transition_matrix


def find_replay_events(reactivation_probs, threshold_percentile=95):
    """
    Identify putative replay events based on high reactivation probability.

    Parameters
    ----------
    reactivation_probs : ndarray, shape (n_timepoints, n_states)
        State reactivation probabilities
    threshold_percentile : float, default=95
        Percentile threshold for event detection

    Returns
    -------
    event_times : ndarray
        Timepoints where replay events were detected
    event_states : ndarray
        States reactivated at each event time
    """
    max_probs = np.max(reactivation_probs, axis=1)
    threshold = np.percentile(max_probs, threshold_percentile)

    event_mask = max_probs > threshold
    event_times = np.where(event_mask)[0]
    event_states = np.argmax(reactivation_probs[event_mask], axis=1)

    return event_times, event_states


def compute_cross_validated_accuracy(decoder, X_train, y_train, time_point_ms=200, cv=5):
    """
    Compute cross-validated decoding accuracy.

    Parameters
    ----------
    decoder : MultivariateReplayDecoder
        Decoder instance
    X_train : ndarray
        Training data
    y_train : ndarray
        Labels
    time_point_ms : int
        Time point for decoding
    cv : int
        Number of cross-validation folds

    Returns
    -------
    accuracies : dict
        Dictionary with accuracy for each state
    """
    from sklearn.model_selection import cross_val_score

    time_idx = int(time_point_ms * decoder.sampling_rate / 1000)
    X_at_time = X_train[:, time_idx, :]

    accuracies = {}

    for state in range(decoder.n_states):
        y_binary = (y_train == state).astype(int)

        if len(decoder.classifiers) > state and len(decoder.scalers) > state:
            X_scaled = decoder.scalers[state].transform(X_at_time)
            scores = cross_val_score(decoder.classifiers[state], X_scaled, y_binary,
                                    cv=cv, scoring='roc_auc')
            accuracies[f'state_{state}'] = {
                'mean': scores.mean(),
                'std': scores.std()
            }

    return accuracies


def simulate_replay_data(n_states=4, n_sensors=100, n_rest_timepoints=3000,
                        lag_samples=4, sequence=None, noise_level=0.3,
                        signal_strength=1.5, sampling_rate=100):
    """
    Simulate resting state data with embedded replay sequences.

    Parameters
    ----------
    n_states : int
        Number of states
    n_sensors : int
        Number of sensors
    n_rest_timepoints : int
        Length of rest period in samples
    lag_samples : int
        Lag between states in replay (samples)
    sequence : list, optional
        Custom sequence to embed. If None, uses 0→1→2→...
    noise_level : float
        Standard deviation of background noise
    signal_strength : float
        Strength of embedded signal
    sampling_rate : int
        Sampling rate in Hz

    Returns
    -------
    X_rest : ndarray, shape (n_rest_timepoints, n_sensors)
        Simulated resting state data with embedded replay
    replay_times : list
        Times when replay events were embedded
    """
    X_rest = np.random.randn(n_rest_timepoints, n_sensors) * noise_level

    if sequence is None:
        sequence = list(range(n_states))

    replay_times = []

    for start_time in range(100, n_rest_timepoints - 100, 200):
        replay_times.append(start_time)
        for step, state in enumerate(sequence):
            time_point = start_time + step * lag_samples
            if time_point < n_rest_timepoints:
                # Add signal to state-specific sensors
                sensor_start = state * (n_sensors // n_states)
                sensor_end = (state + 1) * (n_sensors // n_states)
                X_rest[time_point, sensor_start:sensor_end] += signal_strength

    return X_rest, replay_times
