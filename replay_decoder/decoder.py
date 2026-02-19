"""
Core decoder class for multivariate replay analysis.

This module implements the main decoding algorithm from Liu et al. (2019).
"""

import numpy as np
from pygments.util import docstring_headline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class MultivariateReplayDecoder:
    """
    Multivariate decoding analysis for detecting neural replay sequences.

    This class implements the algorithm from Liu et al. (2019) Cell paper:
    "Human replay spontaneously reorganizes experience"

    The algorithm detects sequential replay of neural representations during
    rest periods by:
    1. Training classifiers on stimulus-evoked activity
    2. Decoding spontaneous reactivations during rest
    3. Computing "sequenceness" via time-lagged regression
    4. Statistical testing via permutation

    Parameters
    ----------
    n_states : int, default=8
        Number of distinct states/stimuli
    max_lag_ms : int, default=600
        Maximum time lag to test in milliseconds
    sampling_rate : int, default=100
        Sampling rate in Hz

    Attributes
    ----------
    classifiers : list
        Trained logistic regression classifiers (one per state)
    scalers : list
        Feature scalers (one per state)

    Examples
    --------
    >>> decoder = MultivariateReplayDecoder(n_states=4, sampling_rate=100)
    >>> decoder.train_classifiers(X_train, y_train, time_point_ms=200)
    >>> reactivations = decoder.decode_states(X_rest)
    >>> sequenceness, lags = decoder.compute_sequenceness(reactivations, transitions)
    """

    def __init__(self, n_states=8, max_lag_ms=600, sampling_rate=100):
        self.n_states = n_states
        self.max_lag_ms = max_lag_ms
        self.sampling_rate = sampling_rate
        self.max_lag_samples = int(max_lag_ms * sampling_rate / 1000)
        self.classifiers = []
        self.scalers = []

    def train_classifiers(self, X_train, y_train, time_point_ms=200, C=1.0):
        """
        Train binary classifiers for each state using lasso logistic regression.

        Parameters
        ----------
        X_train : ndarray, shape (n_trials, n_timepoints, n_sensors)
            Training data from functional localizer task
        y_train : ndarray, shape (n_trials,)
            Labels for each trial (integers from 0 to n_states-1)
        time_point_ms : int, default=200
            Time point relative to stimulus onset to use for training (ms)
        C : float, default=1.0
            Inverse regularization strength for logistic regression

        Returns
        -------
        self : MultivariateReplayDecoder
            Returns self for method chaining
        """
        time_idx = int(time_point_ms * self.sampling_rate / 1000)
        X_at_time = X_train[:, time_idx, :]

        self.classifiers = []
        self.scalers = []

        for state in range(self.n_states):
            y_binary = (y_train == state).astype(int)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_at_time)

            clf = LogisticRegression(
                penalty='l1',
                C=C,
                solver='liblinear',
                max_iter=1000,
                random_state=42
            )
            clf.fit(X_scaled, y_binary)

            self.classifiers.append(clf)
            self.scalers.append(scaler)

        return self

    def decode_states(self, X_rest):
        """
        Apply trained classifiers to resting-state data.

        Parameters
        ----------
        X_rest : ndarray, shape (n_timepoints, n_sensors)
            Resting state neural data (MEG/EEG)

        Returns
        -------
        probabilities : ndarray, shape (n_timepoints, n_states)
            Reactivation probabilities for each state at each time point
        """
        if not self.classifiers:
            raise ValueError("Classifiers not trained. Call train_classifiers() first.")

        n_timepoints = X_rest.shape[0]
        probabilities = np.zeros((n_timepoints, self.n_states))

        for state in range(self.n_states):
            X_scaled = self.scalers[state].transform(X_rest)
            probabilities[:, state] = self.classifiers[state].predict_proba(X_scaled)[:, 1]

        return probabilities

    def compute_sequenceness(self, reactivation_probs, transition_matrix,
                            time_lags_ms=None, alpha_control=True):
        """
        Compute sequenceness measure using time-lagged regression.

        The sequenceness measure quantifies the degree to which state
        reactivations follow a hypothesized sequential structure.

        Parameters
        ----------
        reactivation_probs : ndarray, shape (n_timepoints, n_states)
            State reactivation probabilities from decode_states()
        transition_matrix : ndarray, shape (n_states, n_states)
            Hypothesized transition structure (1 for transitions, 0 otherwise)
        time_lags_ms : array-like, optional
            Specific time lags to test in milliseconds.
            If None, tests lags from 10ms to max_lag_ms in 10ms steps.
        alpha_control : bool, default=True
            Whether to include nuisance regressors for 10Hz oscillations

        Returns
        -------
        sequenceness : ndarray
            Sequenceness measure at each time lag (forward - backward)
        time_lags : ndarray
            Time lags tested in milliseconds

        Notes
        -----
        The algorithm:
        1. For each time lag Δt, regress time-shifted activations X(Δt) onto Y
        2. Obtain regression coefficient matrix β(Δt)
        3. Project β(Δt) onto transition matrix P (Frobenius inner product)
        4. Compute forward and backward sequenceness
        5. Return difference (forward - backward)
        """
        if time_lags_ms is None:
            time_lags_ms = np.arange(10, self.max_lag_ms + 1, 10)

        time_lags_samples = (time_lags_ms * self.sampling_rate / 1000).astype(int)
        sequenceness_forward = np.zeros(len(time_lags_ms))
        sequenceness_backward = np.zeros(len(time_lags_ms))

        Y = reactivation_probs
        n_timepoints, n_states = Y.shape

        # Normalize transition matrix
        P = transition_matrix / (np.sum(transition_matrix) + 1e-10)

        for lag_idx, lag in enumerate(time_lags_samples):
            X_lag = self._create_lagged_matrix(reactivation_probs, lag, alpha_control)

            beta_matrix = np.zeros((n_states, n_states))

            for state_i in range(n_states):
                if lag > 0:
                    y_i = Y[lag:, state_i]
                    X_reg = X_lag[:len(y_i)]
                else:
                    y_i = Y[:lag, state_i]
                    X_reg = X_lag[:len(y_i)]

                try:
                    beta = np.linalg.lstsq(X_reg, y_i, rcond=None)[0]
                    beta_matrix[:, state_i] = beta[:n_states]
                except:
                    continue

            # Project onto transition matrix (Frobenius inner product)
            sequenceness_forward[lag_idx] = np.sum(beta_matrix * P)
            sequenceness_backward[lag_idx] = np.sum(beta_matrix * P.T)

        sequenceness = sequenceness_forward - sequenceness_backward

        return sequenceness, time_lags_ms

    def _create_lagged_matrix(self, reactivation_probs, lag, alpha_control=True):
        """
        Create time-lagged predictor matrix with optional alpha confounds.

        Parameters
        ----------
        reactivation_probs : ndarray
            State reactivation probabilities
        lag : int
            Time lag in samples
        alpha_control : bool
            Whether to include 10Hz confound regressors

        Returns
        -------
        X_lag : ndarray
            Lagged design matrix
        """
        n_timepoints, n_states = reactivation_probs.shape

        if lag > 0:
            X_lag = reactivation_probs[:-lag, :]
        else:
            X_lag = reactivation_probs[-lag:, :]

        if alpha_control:
            confounds = []
            for extra_lag_ms in range(100, 700, 100):
                extra_lag_samples = int(extra_lag_ms * self.sampling_rate / 1000)
                total_lag = lag + extra_lag_samples

                if total_lag < n_timepoints:
                    if total_lag > 0:
                        confound = reactivation_probs[:-total_lag, :]
                    else:
                        confound = reactivation_probs[-total_lag:, :]

                    min_len = min(X_lag.shape[0], confound.shape[0])
                    confounds.append(confound[:min_len, :])

            if confounds:
                min_len = min([X_lag.shape[0]] + [c.shape[0] for c in confounds])
                X_lag = X_lag[:min_len, :]
                confounds = [c[:min_len, :] for c in confounds]
                X_lag = np.hstack([X_lag] + confounds)

        # Add constant term
        X_lag = np.hstack([X_lag, np.ones((X_lag.shape[0], 1))])

        return X_lag

    def permutation_test_decoding(self, X_test, y_test, n_permutations=1000):
        """
        Permutation test for basic decoding accuracy (Figure S3c, S3d).

        This tests whether the classifiers can decode stimuli better than chance
        by randomly permuting the labels and recomputing accuracy.

        Parameters
        ----------
        X_test : ndarray, shape (n_trials, n_timepoints, n_sensors) or (n_trials, n_sensors)
            Test data. If 3D, should specify time_point_ms to extract timepoint.
        y_test : ndarray, shape (n_trials,)
            True labels for test data
        n_permutations : int, default=1000
            Number of random permutations

        Returns
        -------
        true_accuracy : float
            True decoding accuracy
        perm_accuracies : ndarray, shape (n_permutations,)
            Null distribution of accuracies from permuted labels
        p_value : float
            P-value (proportion of permutations >= true accuracy)
        threshold : float
            95th percentile of null distribution

        Notes
        -----
        This implements the permutation test shown in Figure S3 panels c and d,
        where the dotted line shows the permutation threshold for classifier
        performance during the functional localizer task.
        """
        if not self.classifiers:
            raise ValueError("Classifiers not trained. Call train_classifiers() first.")

        # If 3D data, we need a timepoint - use the same as training (200ms default)
        if X_test.ndim == 3:
            # Assume 200ms timepoint (can be made parameter if needed)
            time_idx = int(200 * self.sampling_rate / 1000)
            X_test = X_test[:, time_idx, :]

        # Compute true accuracy
        true_accuracy = self._compute_accuracy(X_test, y_test)

        # Permutation test
        perm_accuracies = np.zeros(n_permutations)
        for perm in range(n_permutations):
            # Shuffle labels
            y_perm = np.random.permutation(y_test)
            perm_accuracies[perm] = self._compute_accuracy(X_test, y_perm)

        # Compute p-value and threshold
        p_value = np.mean(perm_accuracies >= true_accuracy)
        threshold = np.percentile(perm_accuracies, 95)

        return true_accuracy, perm_accuracies, p_value, threshold

    def _compute_accuracy(self, X, y_true):
        """
        Compute classification accuracy.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_sensors)
            Test data (2D)
        y_true : ndarray, shape (n_trials,)
            True labels

        Returns
        -------
        accuracy : float
            Proportion of correctly classified trials
        """
        n_trials = X.shape[0]
        predictions = np.zeros(n_trials, dtype=int)

        for trial in range(n_trials):
            # Get probability from each classifier
            probs = np.zeros(self.n_states)
            for state in range(self.n_states):
                X_scaled = self.scalers[state].transform(X[trial:trial+1, :])
                probs[state] = self.classifiers[state].predict_proba(X_scaled)[0, 1]

            # Predict the state with highest probability
            predictions[trial] = np.argmax(probs)

        accuracy = np.mean(predictions == y_true)
        return accuracy

    def permutation_test(self, reactivation_probs, transition_matrix,
                        n_permutations=1000, time_lags_ms=None ,alpha_control=False):
        """
        Statistical testing via permutation of stimulus labels.

        Parameters
        ----------
        reactivation_probs : ndarray
            State reactivation probabilities
        transition_matrix : ndarray
            Hypothesized transition structure
        n_permutations : int, default=1000
            Number of permutations for null distribution
        time_lags_ms : array-like, optional
            Time lags to test
        alpha_control : bool, default=False
            Whether to control for alpha oscillations with the Liu et al. (2019) method

        Returns
        -------
        p_values : ndarray
            P-values at each time lag
        threshold : float
            Significance threshold (95th percentile, corrected for multiple comparisons)
        true_seq : ndarray
            True sequenceness values
        """
        true_seq, time_lags = self.compute_sequenceness(
            reactivation_probs, transition_matrix, time_lags_ms, alpha_control=alpha_control
        )

        perm_max_abs = np.zeros(n_permutations)

        for perm in range(n_permutations):
            # Permute transition matrix (shuffle rows and columns together)
            perm_indices = np.random.permutation(self.n_states)
            P_perm = transition_matrix[perm_indices, :][:, perm_indices]

            perm_seq, _ = self.compute_sequenceness(
                reactivation_probs, P_perm, time_lags_ms
            )

            perm_max_abs[perm] = np.max(np.abs(perm_seq))

        threshold = np.percentile(perm_max_abs, 95)

        p_values = np.array([
            np.mean(perm_max_abs >= np.abs(seq_val))
            for seq_val in true_seq
        ])

        return p_values, threshold, true_seq
