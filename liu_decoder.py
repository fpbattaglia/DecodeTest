import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt


class MultivariatReplayDecoder:
    """
    Multivariate decoding analysis for detecting neural replay sequences
    based on Liu et al. (2019) Cell paper.
    """

    def __init__(self, n_states=8, max_lag_ms=600, sampling_rate=100):
        """
        Parameters:
        -----------
        n_states : int
            Number of distinct states/stimuli
        max_lag_ms : int
            Maximum time lag to test in milliseconds
        sampling_rate : int
            Sampling rate in Hz
        """
        self.n_states = n_states
        self.max_lag_ms = max_lag_ms
        self.sampling_rate = sampling_rate
        self.max_lag_samples = int(max_lag_ms * sampling_rate / 1000)
        self.classifiers = []
        self.scalers = []

    def train_classifiers(self, X_train, y_train, time_point_ms=200, C=1.0):
        """
        Train binary classifiers for each state using lasso logistic regression.

        Parameters:
        -----------
        X_train : ndarray, shape (n_trials, n_timepoints, n_sensors)
            Training data from functional localizer
        y_train : ndarray, shape (n_trials,)
            Labels for each trial (0 to n_states-1)
        time_point_ms : int
            Time point relative to stimulus onset to use for training
        C : float
            Inverse regularization strength
        """
        time_idx = int(time_point_ms * self.sampling_rate / 1000)
        X_at_time = X_train[:, time_idx, :]  # Extract specific time point

        self.classifiers = []
        self.scalers = []

        # Train one binary classifier per state
        for state in range(self.n_states):
            # Create binary labels
            y_binary = (y_train == state).astype(int)

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_at_time)

            # Train L1-regularized logistic regression
            clf = LogisticRegression(penalty='l1', C=C, solver='liblinear',
                                     max_iter=1000, random_state=42)
            clf.fit(X_scaled, y_binary)

            self.classifiers.append(clf)
            self.scalers.append(scaler)

        print(f"Trained {self.n_states} classifiers")

    def decode_states(self, X_rest):
        """
        Apply trained classifiers to resting-state data.

        Parameters:
        -----------
        X_rest : ndarray, shape (n_timepoints, n_sensors)
            Resting state MEG data

        Returns:
        --------
        probabilities : ndarray, shape (n_timepoints, n_states)
            Reactivation probabilities for each state at each time point
        """
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

        Parameters:
        -----------
        reactivation_probs : ndarray, shape (n_timepoints, n_states)
            State reactivation probabilities from decode_states()
        transition_matrix : ndarray, shape (n_states, n_states)
            Hypothesized transition structure (1 for transitions, 0 otherwise)
        time_lags_ms : array-like, optional
            Specific time lags to test in milliseconds
        alpha_control : bool
            Whether to include nuisance regressors for 10Hz oscillations

        Returns:
        --------
        sequenceness : ndarray
            Sequenceness measure at each time lag
        time_lags : ndarray
            Time lags tested in milliseconds
        """
        if time_lags_ms is None:
            time_lags_ms = np.arange(10, self.max_lag_ms + 1, 10)

        time_lags_samples = (time_lags_ms * self.sampling_rate / 1000).astype(int)
        sequenceness_forward = np.zeros(len(time_lags_ms))
        sequenceness_backward = np.zeros(len(time_lags_ms))

        Y = reactivation_probs  # Current activations
        n_timepoints, n_states = Y.shape

        # Normalize transition matrix
        P = transition_matrix / (np.sum(transition_matrix) + 1e-10)

        for lag_idx, lag in enumerate(time_lags_samples):
            # Create time-lagged design matrix
            X_lag = self._create_lagged_matrix(reactivation_probs, lag, alpha_control)

            # Fit regression for each state
            beta_matrix = np.zeros((n_states, n_states))

            for state_i in range(n_states):
                # Regression: Y_i = X(Δt) * β
                valid_indices = lag if lag > 0 else slice(-lag, None)
                y_i = Y[valid_indices:, state_i] if lag > 0 else Y[:valid_indices, state_i]

                if alpha_control:
                    # Include confound regressors for alpha (10Hz)
                    X_with_confounds = X_lag
                else:
                    X_with_confounds = X_lag[:, :n_states]

                # Ordinary least squares
                try:
                    beta = np.linalg.lstsq(X_with_confounds[:len(y_i)],
                                           y_i, rcond=None)[0]
                    beta_matrix[state_i, :] = beta[:n_states]
                except:
                    continue

            # Project onto transition matrix (Frobenius inner product)
            sequenceness_forward[lag_idx] = np.sum(beta_matrix * P)

            # Backward direction (transpose)
            sequenceness_backward[lag_idx] = np.sum(beta_matrix * P.T)

        # Sequenceness = forward - backward
        sequenceness = sequenceness_forward - sequenceness_backward

        return sequenceness, time_lags_ms

    def _create_lagged_matrix(self, reactivation_probs, lag, alpha_control=True):
        """
        Create time-lagged predictor matrix with optional alpha confounds.
        """
        n_timepoints, n_states = reactivation_probs.shape

        # Basic lagged matrix
        if lag > 0:
            X_lag = reactivation_probs[:-lag, :]
        else:
            X_lag = reactivation_probs[-lag:, :]

        if alpha_control:
            # Add confound regressors at Δt+100ms, Δt+200ms, ... up to Δt+600ms
            confounds = []
            for extra_lag_ms in range(100, 700, 100):
                extra_lag_samples = int(extra_lag_ms * self.sampling_rate / 1000)
                total_lag = lag + extra_lag_samples

                if total_lag < n_timepoints:
                    if total_lag > 0:
                        confound = reactivation_probs[:-total_lag, :]
                    else:
                        confound = reactivation_probs[-total_lag:, :]

                    # Match length
                    min_len = min(X_lag.shape[0], confound.shape[0])
                    confounds.append(confound[:min_len, :])

            if confounds:
                # Concatenate confounds
                min_len = min([X_lag.shape[0]] + [c.shape[0] for c in confounds])
                X_lag = X_lag[:min_len, :]
                confounds = [c[:min_len, :] for c in confounds]
                X_lag = np.hstack([X_lag] + confounds)

        # Add constant term
        X_lag = np.hstack([X_lag, np.ones((X_lag.shape[0], 1))])

        return X_lag

    def permutation_test(self, reactivation_probs, transition_matrix,
                         n_permutations=1000, time_lags_ms=None):
        """
        Statistical testing via permutation of stimulus labels.

        Parameters:
        -----------
        reactivation_probs : ndarray
            State reactivation probabilities
        transition_matrix : ndarray
            Hypothesized transition structure
        n_permutations : int
            Number of permutations
        time_lags_ms : array-like, optional
            Time lags to test

        Returns:
        --------
        p_values : ndarray
            P-values at each time lag
        threshold : float
            Significance threshold (corrected for multiple comparisons)
        """
        # Compute true sequenceness
        true_seq, time_lags = self.compute_sequenceness(
            reactivation_probs, transition_matrix, time_lags_ms
        )

        # Permutation distribution
        perm_max_abs = np.zeros(n_permutations)

        for perm in range(n_permutations):
            # Permute transition matrix (shuffle rows and columns together)
            perm_indices = np.random.permutation(self.n_states)
            P_perm = transition_matrix[perm_indices, :][:, perm_indices]

            # Compute sequenceness with permuted matrix
            perm_seq, _ = self.compute_sequenceness(
                reactivation_probs, P_perm, time_lags_ms
            )

            # Store maximum absolute value across lags
            perm_max_abs[perm] = np.max(np.abs(perm_seq))

        # Threshold at 95th percentile
        threshold = np.percentile(perm_max_abs, 95)

        # Compute p-values
        p_values = np.array([
            np.mean(perm_max_abs >= np.abs(seq_val))
            for seq_val in true_seq
        ])

        return p_values, threshold, true_seq

    def plot_sequenceness(self, sequenceness, time_lags, threshold=None):
        """
        Visualize sequenceness results.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(time_lags, sequenceness, 'k-', linewidth=2)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)

        if threshold is not None:
            plt.axhline(y=threshold, color='r', linestyle='--',
                        label=f'Threshold (p<0.05)')
            plt.axhline(y=-threshold, color='r', linestyle='--')

        plt.xlabel('Time Lag (ms)', fontsize=12)
        plt.ylabel('Sequenceness\n(Forward - Backward)', fontsize=12)
        plt.title('Neural Replay Sequenceness Analysis', fontsize=14)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        return plt.gcf()


# Example usage demonstration
def demonstration():
    """
    Demonstrate the algorithm with simulated data.
    """
    print("=" * 60)
    print("Multivariate Decoding Analysis for Neural Replay")
    print("Based on Liu et al. (2019) Cell")
    print("=" * 60)

    # Simulate training data (functional localizer)
    np.random.seed(42)
    n_trials = 200
    n_timepoints_per_trial = 50  # 500ms at 100Hz
    n_sensors = 100
    n_states = 4

    print("\n1. Simulating training data...")
    X_train = np.random.randn(n_trials, n_timepoints_per_trial, n_sensors)
    y_train = np.random.randint(0, n_states, n_trials)

    # Add signal at 200ms for each state
    time_idx = 20  # 200ms
    for trial in range(n_trials):
        state = y_train[trial]
        # Add state-specific pattern
        X_train[trial, time_idx, state * 10:(state + 1) * 10] += 2.0

    # Initialize decoder
    decoder = MultivariatReplayDecoder(n_states=n_states, max_lag_ms=600)

    # Train classifiers
    print("\n2. Training classifiers...")
    decoder.train_classifiers(X_train, y_train, time_point_ms=200)

    # Simulate resting state data with embedded sequence
    print("\n3. Simulating resting state with replay sequence...")
    n_rest_timepoints = 3000  # 30 seconds at 100Hz
    X_rest = np.random.randn(n_rest_timepoints, n_sensors) * 0.5

    # Embed sequence A->B->C->D at 40ms lag, at multiple time points
    lag_samples = 4  # 40ms at 100Hz
    sequence = [0, 1, 2, 3]  # A->B->C->D

    for start_time in range(100, n_rest_timepoints - 100, 200):
        for step, state in enumerate(sequence):
            time_point = start_time + step * lag_samples
            if time_point < n_rest_timepoints:
                X_rest[time_point, state * 10:(state + 1) * 10] += 1.5

    # Decode states
    print("\n4. Decoding states from resting data...")
    reactivation_probs = decoder.decode_states(X_rest)
    print(f"   Reactivation matrix shape: {reactivation_probs.shape}")

    # Define transition matrix (forward sequence)
    transition_matrix = np.array([
        [0, 1, 0, 0],  # A -> B
        [0, 0, 1, 0],  # B -> C
        [0, 0, 0, 1],  # C -> D
        [0, 0, 0, 0]  # D -> nothing
    ])

    # Compute sequenceness
    print("\n5. Computing sequenceness...")
    sequenceness, time_lags = decoder.compute_sequenceness(
        reactivation_probs, transition_matrix
    )

    # Find peak
    peak_idx = np.argmax(np.abs(sequenceness))
    peak_lag = time_lags[peak_idx]
    peak_value = sequenceness[peak_idx]

    print(f"   Peak sequenceness: {peak_value:.4f} at {peak_lag}ms lag")

    # Permutation test
    print("\n6. Running permutation test (100 permutations)...")
    p_values, threshold, true_seq = decoder.permutation_test(
        reactivation_probs, transition_matrix, n_permutations=100
    )

    significant_lags = time_lags[p_values < 0.05]
    print(f"   Significance threshold: {threshold:.4f}")
    print(f"   Significant time lags: {significant_lags}")

    # Visualize
    print("\n7. Generating visualization...")
    fig = decoder.plot_sequenceness(sequenceness, time_lags, threshold)
    plt.savefig('sequenceness_analysis.png', dpi=150, bbox_inches='tight')
    print("   Saved plot to 'sequenceness_analysis.png'")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    demonstration()
