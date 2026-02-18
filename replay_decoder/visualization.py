"""
Visualization tools for replay analysis.

This module provides plotting functions for sequenceness results,
reactivation time series, and transition matrices.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_sequenceness(sequenceness, time_lags, threshold=None, ax=None, **kwargs):
    """
    Plot sequenceness as a function of time lag.

    Parameters
    ----------
    sequenceness : ndarray
        Sequenceness values at each time lag
    time_lags : ndarray
        Time lags in milliseconds
    threshold : float, optional
        Significance threshold to plot as horizontal line
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    **kwargs : dict
        Additional keyword arguments passed to plot()

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    # Plot sequenceness
    ax.plot(time_lags, sequenceness, linewidth=2.5, label='Sequenceness', **kwargs)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    # Plot threshold if provided
    if threshold is not None:
        ax.axhline(y=threshold, color='r', linestyle='--', linewidth=2,
                   label=f'Threshold (p<0.05)')
        ax.axhline(y=-threshold, color='r', linestyle='--', linewidth=2)

        # Fill significant regions
        ax.fill_between(time_lags, 0, sequenceness,
                       where=(sequenceness > threshold),
                       alpha=0.3, color='blue', label='Significant (forward)')
        ax.fill_between(time_lags, 0, sequenceness,
                       where=(sequenceness < -threshold),
                       alpha=0.3, color='red', label='Significant (backward)')

    ax.set_xlabel('Time Lag (ms)', fontsize=12)
    ax.set_ylabel('Sequenceness\n(Forward - Backward)', fontsize=12)
    ax.set_title('Neural Replay Sequenceness Analysis', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig, ax


def plot_reactivations(reactivation_probs, time_window=None, state_names=None,
                      sampling_rate=100, ax=None):
    """
    Plot state reactivation probabilities over time.

    Parameters
    ----------
    reactivation_probs : ndarray, shape (n_timepoints, n_states)
        State reactivation probabilities
    time_window : tuple of int, optional
        (start, end) indices to plot. If None, plots all data.
    state_names : list of str, optional
        Names for each state. If None, uses numeric labels.
    sampling_rate : int, default=100
        Sampling rate in Hz
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 5))
    else:
        fig = ax.get_figure()

    n_timepoints, n_states = reactivation_probs.shape

    if time_window is None:
        time_window = (0, n_timepoints)

    start, end = time_window
    time_ms = np.arange(start, end) * (1000 / sampling_rate)

    if state_names is None:
        state_names = [f'State {i}' for i in range(n_states)]

    for state in range(n_states):
        ax.plot(time_ms, reactivation_probs[start:end, state],
               label=state_names[state], linewidth=1.5, alpha=0.8)

    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Reactivation Probability', fontsize=12)
    ax.set_title('State Reactivations During Rest', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', ncol=min(4, n_states))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax


def plot_transition_matrix(transition_matrix, state_names=None, ax=None):
    """
    Visualize transition matrix as a heatmap.

    Parameters
    ----------
    transition_matrix : ndarray, shape (n_states, n_states)
        Transition matrix to visualize
    state_names : list of str, optional
        Names for each state. If None, uses numeric labels.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    n_states = transition_matrix.shape[0]

    if state_names is None:
        state_names = [f'{i}' for i in range(n_states)]

    im = ax.imshow(transition_matrix, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(n_states))
    ax.set_yticks(range(n_states))
    ax.set_xticklabels(state_names)
    ax.set_yticklabels(state_names)
    ax.set_xlabel('To State', fontsize=12)
    ax.set_ylabel('From State', fontsize=12)
    ax.set_title('Transition Matrix', fontsize=14, fontweight='bold')

    # Add text annotations
    for i in range(n_states):
        for j in range(n_states):
            value = transition_matrix[i, j]
            if value > 0:
                text = ax.text(j, i, f'{value:.0f}',
                             ha="center", va="center",
                             color="white" if value > 0.5 else "black",
                             fontsize=12, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Transition')
    plt.tight_layout()
    return fig, ax


def plot_comparison(sequenceness_list, labels, time_lags, ax=None):
    """
    Plot multiple sequenceness curves for comparison.

    Parameters
    ----------
    sequenceness_list : list of ndarray
        List of sequenceness arrays to compare
    labels : list of str
        Labels for each curve
    time_lags : ndarray
        Time lags in milliseconds
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()

    for seq, label in zip(sequenceness_list, labels):
        ax.plot(time_lags, seq, linewidth=2.5, label=label, alpha=0.8)

    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time Lag (ms)', fontsize=12)
    ax.set_ylabel('Sequenceness', fontsize=12)
    ax.set_title('Sequenceness Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig, ax
