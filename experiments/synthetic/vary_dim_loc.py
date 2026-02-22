# this is a wrong file with wrong goal.

import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima_process import arma_generate_sample
import matplotlib.pyplot as plt
import seaborn as sns
import math
from numba import jit, prange
import warnings
warnings.filterwarnings('ignore')

# ==================================
# Plotting Configuration
# ==================================
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11
sns.set_style("whitegrid")

# ==================================
# Numba Helpers
# ==================================
@jit(nopython=True)
def numba_ix(arr, rows, cols):
    one_d_index = np.zeros(len(rows) * len(cols), dtype=np.int32)
    for i in prange(len(rows)):
        r = rows[i]
        start = i * len(cols)
        one_d_index[start: start + len(cols)] = cols + arr.shape[1] * r
    arr_1d = arr.reshape((arr.shape[0] * arr.shape[1], 1))
    slice_1d = np.take(arr_1d, one_d_index)
    return slice_1d.reshape((len(rows), len(cols)))

def is_power_of_two(n):
    return (n != 0) and (n & (n-1) == 0)

@jit(nopython=True)
def _CLBEF_get_UCB(x_hat, t, K, theta_hat, d, T, V_inv, p_hat, x_sum):
    return x_hat@theta_hat+(np.sqrt((d+1)*np.log((t+2)*T))+x_sum*d*((1/p_hat)**(3/2))*np.sqrt(np.log(T)*np.log(K*T)/((t+2)*K))+\
                            np.sqrt(d*np.log(K*T)))*np.sqrt(x_hat@V_inv@x_hat.T)

@jit(nopython=True, parallel=True)
def _CLBEF_UCB(x_hat, t, K, theta_hat, d, T, V_inv, p_hat, x_sum):
    ucb_list = np.zeros(K)
    for k in prange(K):
        ucb_list[k] = _CLBEF_get_UCB(x_hat[k], t, K, theta_hat.copy(), d, T, V_inv.copy(), p_hat, x_sum)
    chosen_arm = np.argmax(ucb_list)
    max_ucb = ucb_list[chosen_arm]
    return chosen_arm, max_ucb

@jit(nopython=True)
def numba_idxSU(m):
    d = m.shape[0]
    idxS, idxU = [], []
    for i in range(d):
        if (m[i] < 1.0e-10) and (m[i] > -1.0e-10):
            idxU.append(i)
        else:
            idxS.append(i)
    return np.asarray(idxS), np.asarray(idxU)

@jit(nopython=True)
def _CLBEF_x_bars(nu_hat, Sigma_hat, x, m, x_bar_dummy):
    index_S, index_U = numba_idxSU(m)
    if len(index_S) > 0:
        x_S = x[index_S]
        x_bar_dummy[index_S] = x_S
        if len(index_U) > 0:
            x_U = nu_hat[index_U] + numba_ix(Sigma_hat, index_U, index_S) @ np.linalg.pinv(numba_ix(Sigma_hat, index_S, index_S)) @ (x_S - nu_hat[index_S]).T
            x_bar_dummy[index_U] = x_U
    else:
        x_bar_dummy[index_U] = nu_hat[index_U]
    return x_bar_dummy

@jit(nopython=True, parallel=True)
def _CLBEF_x_hats(nu_hat, Sigma_hat, x, m, K, x_bar_dummy, x_hat_dummy):
    for k in prange(K):
        x_hat_dummy[k, 1:] = _CLBEF_x_bars(nu_hat, Sigma_hat, x[k], m[k], x_bar_dummy[k])
    return x_hat_dummy

@jit(nopython=True)
def _CLBEF_get_estimators(d, K, x_t, m_t, Kt, xi, n, Z):
    p_hat = max([1, n]) / (d * Kt)
    nu_hat = 1 / (Kt * p_hat) * xi
    Sigma_hat = Z * (((p_hat - 1) / (p_hat * p_hat)) * np.identity(d) + 1 / (p_hat * p_hat)) / (Kt) - np.outer(nu_hat, nu_hat)
    return p_hat, nu_hat, Sigma_hat

# ==================================
# 1. Config Class
# ==================================
class LinearConfig:
    def __init__(self, seed=42):
        self.SEED = seed
        np.random.seed(seed)
        self.N_TRAJECTORIES = 200 
        self.T_TRAJECTORY = 50    
        self.N_TIME_STEPS = 1000
        self.N_EXPERIMENTS = 10   
        self.N_ACTIONS = 2
        self.MC_SAMPLES = 10
        
        self.AR_PARAMS = np.array([0.75, -0.25])
        self.MA_PARAMS = np.array([0.65, 0.35])
        self.STATE_NOISE_STD = 0.1 
        
        self.BETA_FEATURE_DIM = 10 # Initial value, will be overwritten
        self.TRUE_BETA = None
        
        self.S_PRIME_NOISE_STD = 0.1 
        
        self.THETA_FEATURE_DIM = 4
        self.BENCHMARK_FEATURE_DIM = 2
        
        self.TRUE_THETA = np.array([0.1, 0.1, 0.1, 1.5]) 
        self.BENCHMARK_THETA = np.random.randn(self.BENCHMARK_FEATURE_DIM)
        self.REWARD_NOISE_STD = 0.05

# ==================================
# 2. Feature Functions with LAG Support
# ==================================
def create_features_for_S_prime(s_history, config, lag=0):
    dim = config.BETA_FEATURE_DIM
    required = dim + lag
    
    if len(s_history) < required:
        return np.zeros(dim)
    
    features = np.zeros(dim)
    features[0] = 1.0
    hist_rev = s_history[::-1] 
    feat_needed = dim - 1
    selected_hist = hist_rev[lag : lag + feat_needed]
    features[1:] = selected_hist
    return features

def compute_S_prime(s_history, config, lag=0):
    real_features = create_features_for_S_prime(s_history, config, lag=0)
    if config.TRUE_BETA is None: # Safety check
         config.TRUE_BETA = np.random.randn(config.BETA_FEATURE_DIM)
    return np.dot(config.TRUE_BETA, real_features) + np.random.normal(0, config.S_PRIME_NOISE_STD)

def compute_S_prime_nonlinear(s_history, config):
    return compute_S_prime(s_history, config)

def create_OFUL_Full_features(s_t, s_prime_t, action, config):
    features = np.zeros(config.THETA_FEATURE_DIM)
    features[0] = 1.0
    features[1] = s_t
    features[2] = s_prime_t
    features[3] = s_prime_t * action 
    return features

def create_OFUL_features(s_t, action, config):
    features = np.zeros(config.BENCHMARK_FEATURE_DIM)
    features[0] = 1.0
    features[1] = s_t * action 
    return features

# ==================================
# 3. Data Generation
# ==================================
def generate_single_trajectory(config, length):
    ar = np.r_[1, -config.AR_PARAMS]
    ma = np.r_[1, config.MA_PARAMS]
    s_sequence = arma_generate_sample(ar=ar, ma=ma, nsample=length, scale=config.STATE_NOISE_STD)
    return s_sequence

def generate_pretraining_data(config, lag=0):
    X_beta, y_s_prime = [], []
    for traj in range(config.N_TRAJECTORIES):
        s_sequence = generate_single_trajectory(config, config.T_TRAJECTORY)
        start_index = config.BETA_FEATURE_DIM + lag
        for t in range(start_index, len(s_sequence)):
            s_history = list(s_sequence[:t+1])
            features = create_features_for_S_prime(s_history, config, lag=lag)
            s_prime = compute_S_prime(s_history, config, lag=0)
            X_beta.append(features)
            y_s_prime.append(s_prime)
    return np.array(X_beta), np.array(y_s_prime)

def compute_true_reward(s_t, s_prime_t, action, config):
    x_true = create_OFUL_Full_features(s_t, s_prime_t, action, config)
    return np.dot(config.TRUE_THETA, x_true) + np.random.normal(0, config.REWARD_NOISE_STD)

def get_optimal_reward(s_t, s_prime_t, config):
    return np.max([np.dot(config.TRUE_THETA, create_OFUL_Full_features(s_t, s_prime_t, a, config)) for a in range(config.N_ACTIONS)])

def train_beta_estimator(X_beta, y_s_prime):
    model = LinearRegression()
    if len(X_beta) > 0:
        model.fit(X_beta, y_s_prime)
    else:
        model.fit(np.zeros((1, X_beta.shape[1])), [0])
    predictions = model.predict(X_beta) if len(X_beta) > 0 else [0]
    residuals = y_s_prime - predictions if len(X_beta) > 0 else [0]
    std = np.std(residuals) if len(residuals) > 0 else 1.0
    return {"model": model, "prediction_std": std}

# ==================================
# 4. Agents
# ==================================
class OFUL_FullLinUCBAgent:
    def __init__(self, config, alpha=2.0):
        self.config = config
        self.alpha = alpha
        self.A = np.identity(config.THETA_FEATURE_DIM)
        self.b = np.zeros((config.THETA_FEATURE_DIM, 1))
    def choose_action(self, s_t, s_prime_t):
        if np.linalg.det(self.A) < 1e-10: self.A += 1e-6 * np.identity(self.A.shape[0])
        A_inv = np.linalg.inv(self.A)
        theta_hat = A_inv @ self.b
        ucb_scores = np.zeros(self.config.N_ACTIONS)
        for a in range(self.config.N_ACTIONS):
            x_ta = create_OFUL_Full_features(s_t, s_prime_t, a, self.config).reshape(-1, 1)
            mean_reward = (theta_hat.T @ x_ta).item()
            conf = self.alpha * np.sqrt(x_ta.T @ A_inv @ x_ta).item()
            ucb_scores[a] = mean_reward + conf
        return np.argmax(ucb_scores)
    def update(self, action, reward, s_t, s_prime_t):
        x_chosen = create_OFUL_Full_features(s_t, s_prime_t, action, self.config).reshape(-1, 1)
        self.A += x_chosen @ x_chosen.T
        self.b += reward * x_chosen

class OFULLinUCBAgent:
    def __init__(self, config, alpha=2.0):
        self.config = config
        self.alpha = alpha
        self.A = np.identity(config.BENCHMARK_FEATURE_DIM)
        self.b = np.zeros((config.BENCHMARK_FEATURE_DIM, 1))
    def choose_action(self, s_t):
        if np.linalg.det(self.A) < 1e-10: self.A += 1e-6 * np.identity(self.A.shape[0])
        A_inv = np.linalg.inv(self.A)
        theta_hat = A_inv @ self.b
        ucb_scores = np.zeros(self.config.N_ACTIONS)
        for a in range(self.config.N_ACTIONS):
            x_ta = create_OFUL_features(s_t, a, self.config).reshape(-1, 1)
            mean_reward = (theta_hat.T @ x_ta).item()
            conf = self.alpha * np.sqrt(x_ta.T @ A_inv @ x_ta).item()
            ucb_scores[a] = mean_reward + conf
        return np.argmax(ucb_scores)
    def update(self, action, reward, s_t):
        x_chosen = create_OFUL_features(s_t, action, self.config).reshape(-1, 1)
        self.A += x_chosen @ x_chosen.T
        self.b += reward * x_chosen

class PartialObservationLinUCBAgent:
    def __init__(self, config, beta_estimator, alpha=2.0, lag=0):
        self.config = config
        self.beta_estimator = beta_estimator
        self.alpha = alpha
        self.lag = lag
        self.A = np.identity(config.THETA_FEATURE_DIM)
        self.b = np.zeros((config.THETA_FEATURE_DIM, 1))
    def predict_s_prime(self, s_history):
        features = create_features_for_S_prime(s_history, self.config, lag=self.lag)
        return self.beta_estimator["model"].predict(features.reshape(1, -1))[0]
    def choose_action(self, s_history):
        if np.linalg.det(self.A) < 1e-10: self.A += 1e-6 * np.identity(self.A.shape[0])
        A_inv = np.linalg.inv(self.A)
        theta_hat = A_inv @ self.b
        ucb_scores = np.zeros(self.config.N_ACTIONS)
        s_t = s_history[-1]
        pred_s_prime = self.predict_s_prime(s_history)
        for a in range(self.config.N_ACTIONS):
            x_ta = create_OFUL_Full_features(s_t, pred_s_prime, a, self.config).reshape(-1, 1)
            mean_reward = (theta_hat.T @ x_ta).item()
            conf = self.alpha * np.sqrt(x_ta.T @ A_inv @ x_ta).item()
            ucb_scores[a] = mean_reward + conf
        return np.argmax(ucb_scores)
    def update(self, action, reward, s_history):
        s_t = s_history[-1]
        pred_s_prime = self.predict_s_prime(s_history)
        x_chosen = create_OFUL_Full_features(s_t, pred_s_prime, action, self.config).reshape(-1, 1)
        self.A += x_chosen @ x_chosen.T
        self.b += reward * x_chosen

class CLBBFAgent:
    def __init__(self, config, alpha=2.0):
        self.config = config
        self.alpha = alpha
        self.d = config.THETA_FEATURE_DIM - 1
        self.K = config.N_ACTIONS
        self.T = config.N_TIME_STEPS
        self.n = 0
        self.Z = np.zeros((self.d, self.d))
        self.xi = np.zeros(self.d)
        self.V = (self.d + 1) * math.log(self.K * self.T) * np.identity(self.d + 1)
        self.xy = np.zeros(self.d + 1)
        self.x_his, self.m_his, self.r = [], [], []
        self.x_sum, self.t = 0, 0
    def choose_action(self, s_t, s_prime_t):
        x_t = np.zeros((self.K, self.d))
        m_t = np.zeros((self.K, self.d))
        for a in range(self.K):
            features = create_OFUL_Full_features(s_t, s_prime_t, a, self.config)
            x_t[a] = features[1:]
            m_t[a, 0] = 1; m_t[a, 1] = 0; m_t[a, 2] = 1
        for k in range(self.K):
            self.n += np.sum(m_t[k])
            self.Z += np.outer(x_t[k], x_t[k])
            self.xi += x_t[k]
        Kt = self.K * (self.t + 1)
        self.p_hat, self.nu_hat, self.Sigma_hat = _CLBEF_get_estimators(
            self.d, self.K, x_t, m_t, Kt, self.xi, self.n, self.Z
        )
        x_hat_dummy = np.ones((self.K, self.d + 1))
        x_bar_dummy = np.zeros((self.K, self.d))
        self.x_hat = _CLBEF_x_hats(self.nu_hat, self.Sigma_hat, x_t, m_t, self.K, x_bar_dummy, x_hat_dummy)
        if self.t == 0:
            chosen_action = np.random.choice(self.K)
        else:
            if is_power_of_two(self.t):
                self.V = (self.d + 1) * math.log(self.K * self.T) * np.identity(self.d + 1)
                self.xy = np.zeros(self.d + 1)
                self.x_sum = 0
                for s in range(len(self.x_his)):
                    x_bar = np.insert(self._x_bars(self.x_his[s], self.m_his[s]), 0, 1)
                    self.V += np.outer(x_bar, x_bar)
                    if s < len(self.r): self.xy += x_bar * self.r[s]
                self.V_inv = np.linalg.pinv(self.V)
                for s in range(len(self.x_his)):
                    x_bar = np.insert(self._x_bars(self.x_his[s], self.m_his[s]), 0, 1)
                    self.x_sum += np.sqrt(x_bar @ self.V_inv @ x_bar)
            self.V_inv = np.linalg.pinv(self.V)
            self.theta_hat = self.V_inv @ self.xy.T
            chosen_action, _ = _CLBEF_UCB(self.x_hat, self.t, self.K, self.theta_hat, self.d, self.T, self.V_inv, self.p_hat, self.x_sum)
        self._current_x_t = x_t.copy()
        self._current_m_t = m_t.copy()
        return chosen_action
    def update(self, action, reward, s_t, s_prime_t):
        self.r.append(reward)
        if hasattr(self, '_current_x_t'):
            self.x_his.append(self._current_x_t[action])
            self.m_his.append(self._current_m_t[action])
            if hasattr(self, 'x_hat'):
                self.V += np.outer(self.x_hat[action], self.x_hat[action])
                self.xy += self.x_hat[action] * reward
        self.t += 1
    def _x_bars(self, x, m):
        x_bar_dummy = np.zeros(self.d)
        return _CLBEF_x_bars(self.nu_hat, self.Sigma_hat, x, m, x_bar_dummy)

# ==================================
# 6. Runners (Baselines & Varying)
# ==================================
def run_OFUL_Full_experiment(config):
    agent = OFUL_FullLinUCBAgent(config)
    s_sequence = generate_single_trajectory(config, config.N_TIME_STEPS)
    regret_log = np.zeros(config.N_TIME_STEPS)
    start_index = config.BETA_FEATURE_DIM 
    for t in range(start_index, config.N_TIME_STEPS):
        s_history = list(s_sequence[:t+1])
        s_t = s_history[-1]; s_prime_t = compute_S_prime(s_history, config)
        chosen_action = agent.choose_action(s_t, s_prime_t)
        observed_reward = compute_true_reward(s_t, s_prime_t, chosen_action, config)
        optimal_reward = get_optimal_reward(s_t, s_prime_t, config)
        agent.update(chosen_action, observed_reward, s_t, s_prime_t)
        regret_log[t] = optimal_reward - observed_reward
    return {'cumulative_regret': np.cumsum(regret_log)}

def run_OFUL_experiment(config):
    agent = OFULLinUCBAgent(config)
    s_sequence = generate_single_trajectory(config, config.N_TIME_STEPS)
    regret_log = np.zeros(config.N_TIME_STEPS)
    start_index = config.BETA_FEATURE_DIM
    for t in range(start_index, config.N_TIME_STEPS):
        s_history = list(s_sequence[:t+1])
        s_t = s_history[-1]; s_prime_t = compute_S_prime(s_history, config)
        chosen_action = agent.choose_action(s_t)
        observed_reward = compute_true_reward(s_t, s_prime_t, chosen_action, config)
        optimal_reward = get_optimal_reward(s_t, s_prime_t, config)
        agent.update(chosen_action, observed_reward, s_t)
        regret_log[t] = optimal_reward - observed_reward
    return {'cumulative_regret': np.cumsum(regret_log)}

def run_CLBBF_experiment(config):
    agent = CLBBFAgent(config)
    s_sequence = generate_single_trajectory(config, config.N_TIME_STEPS)
    regret_log = np.zeros(config.N_TIME_STEPS)
    start_index = config.BETA_FEATURE_DIM
    for t in range(start_index, config.N_TIME_STEPS):
        s_history = list(s_sequence[:t+1])
        s_t = s_history[-1]; s_prime_t = compute_S_prime(s_history, config)
        chosen_action = agent.choose_action(s_t, s_prime_t)
        observed_reward = compute_true_reward(s_t, s_prime_t, chosen_action, config)
        optimal_reward = get_optimal_reward(s_t, s_prime_t, config)
        agent.update(chosen_action, observed_reward, s_t, s_prime_t)
        regret_log[t] = optimal_reward - observed_reward
    return {'cumulative_regret': np.cumsum(regret_log)}

def run_varying_obs_experiment(config, lag=0):
    X_beta, y_s_prime = generate_pretraining_data(config, lag=lag)
    estimator = train_beta_estimator(X_beta, y_s_prime)
    agent = PartialObservationLinUCBAgent(config, estimator, lag=lag)
    s_sequence = generate_single_trajectory(config, config.N_TIME_STEPS)
    regret_log = np.zeros(config.N_TIME_STEPS)
    start_index = config.BETA_FEATURE_DIM + lag
    for t in range(start_index, config.N_TIME_STEPS):
        s_history = list(s_sequence[:t+1])
        s_t = s_history[-1]; s_prime_t = compute_S_prime(s_history, config, lag=0)
        chosen_action = agent.choose_action(s_history)
        x_true = create_OFUL_Full_features(s_t, s_prime_t, action=chosen_action, config=config)
        reward = np.dot(config.TRUE_THETA, x_true) + np.random.normal(0, config.REWARD_NOISE_STD)
        opt_rewards = [np.dot(config.TRUE_THETA, create_OFUL_Full_features(s_t, s_prime_t, a, config)) for a in range(config.N_ACTIONS)]
        optimal = max(opt_rewards)
        agent.update(chosen_action, reward, s_history)
        regret_log[t] = optimal - reward
    return np.sum(regret_log)

# ==================================
# 7. Main Analysis Runner
# ==================================
def run_robustness_analysis(seed=42):
    # Set global seed for reproducibility
    np.random.seed(seed)
    
    config = LinearConfig(seed=seed)
    
    print("Step 1: Evaluating Constant Baselines (OFUL-Full, OFUL)...")
    config.BETA_FEATURE_DIM = 20
    config.TRUE_BETA = np.random.randn(20)

    # Calculate Mean and Std Error for Baselines
    regrets = []
    for exp_idx in range(config.N_EXPERIMENTS):
        np.random.seed(seed + exp_idx)  # Different seed for each experiment
        r = run_OFUL_Full_experiment(config)
        regrets.append(r['cumulative_regret'][-1])
    oful_full_mean = np.mean(regrets)
    oful_full_se = np.std(regrets) / np.sqrt(config.N_EXPERIMENTS)
    print(f"  OFUL-Full: {oful_full_mean:.2f} +/- {oful_full_se:.2f}")

    regrets = []
    for exp_idx in range(config.N_EXPERIMENTS):
        np.random.seed(seed + exp_idx)
        r = run_OFUL_experiment(config)
        regrets.append(r['cumulative_regret'][-1])
    oful_mean = np.mean(regrets)
    oful_se = np.std(regrets) / np.sqrt(config.N_EXPERIMENTS)
    print(f"  OFUL: {oful_mean:.2f} +/- {oful_se:.2f}")

    # --- Experiment A: Varying Dimension ---
    dims = [5, 10, 20, 40]
    
    print("\nStep 2: Experiment A - Varying Observed Dimension...")
    
    # Prepare table data for Experiment A
    table_data_a = []
    headers_a = ["Dimension", "PULSE-UCB", "CLBBF"]
    table_data_a.append(headers_a)

    for d in dims:
        config.BETA_FEATURE_DIM = d
        config.TRUE_BETA = np.random.randn(d)
        
        # PULSE-UCB results
        pulse_regrets = []
        for exp_idx in range(config.N_EXPERIMENTS):
            np.random.seed(seed + 100 + d * 10 + exp_idx)  # Unique seed per dim and experiment
            r = run_varying_obs_experiment(config, lag=0)
            pulse_regrets.append(r)
        pulse_mean = np.mean(pulse_regrets)
        pulse_se = np.std(pulse_regrets) / np.sqrt(config.N_EXPERIMENTS)
        
        # CLBBF results
        clbbf_regrets = []
        for exp_idx in range(config.N_EXPERIMENTS):
            np.random.seed(seed + 100 + d * 10 + exp_idx)
            r = run_CLBBF_experiment(config)
            clbbf_regrets.append(r['cumulative_regret'][-1])
        clbbf_mean = np.mean(clbbf_regrets)
        clbbf_se = np.std(clbbf_regrets) / np.sqrt(config.N_EXPERIMENTS)
        
        print(f"  Dim={d}: PULSE-UCB={pulse_mean:.2f}+/-{pulse_se:.2f}, CLBBF={clbbf_mean:.2f}+/-{clbbf_se:.2f}")
        table_data_a.append([
            f"{d}", 
            f"{pulse_mean:.2f} +/- {pulse_se:.2f}",
            f"{clbbf_mean:.2f} +/- {clbbf_se:.2f}"
        ])

    # Print Table A
    print("\n" + "="*70)
    print("Table 2A: Robustness to Varying Dimension (Sparsity)")
    print("="*70)
    col_widths_a = [max(len(str(item)) for item in col) for col in zip(*table_data_a)]
    header_row_a = table_data_a[0]
    header_str_a = " | ".join(f"{item:<{width}}" for item, width in zip(header_row_a, col_widths_a))
    print(header_str_a)
    print("-" * len(header_str_a))
    for row in table_data_a[1:]:
        print(" | ".join(f"{item:<{width}}" for item, width in zip(row, col_widths_a)))
    print("="*70 + "\n")


    # --- Experiment B: Varying Lag ---
    config.BETA_FEATURE_DIM = 20
    config.TRUE_BETA = np.random.randn(20)
    lags = [0, 5, 10, 20]
    
    print("\nStep 3: Experiment B - Varying Observation Lag...")
    
    # Prepare table data for Experiment B
    table_data_b = []
    headers_b = ["Lag", "PULSE-UCB", "CLBBF"]
    table_data_b.append(headers_b)

    for l in lags:
        # PULSE-UCB results
        pulse_regrets = []
        for exp_idx in range(config.N_EXPERIMENTS):
            np.random.seed(seed + 500 + l * 10 + exp_idx)  # Unique seed per lag and experiment
            r = run_varying_obs_experiment(config, lag=l)
            pulse_regrets.append(r)
        pulse_mean = np.mean(pulse_regrets)
        pulse_se = np.std(pulse_regrets) / np.sqrt(config.N_EXPERIMENTS)
        
        # CLBBF results (note: CLBBF doesn't use lag parameter directly)
        clbbf_regrets = []
        for exp_idx in range(config.N_EXPERIMENTS):
            np.random.seed(seed + 500 + l * 10 + exp_idx)
            r = run_CLBBF_experiment(config)
            clbbf_regrets.append(r['cumulative_regret'][-1])
        clbbf_mean = np.mean(clbbf_regrets)
        clbbf_se = np.std(clbbf_regrets) / np.sqrt(config.N_EXPERIMENTS)
        
        print(f"  Lag={l}: PULSE-UCB={pulse_mean:.2f}+/-{pulse_se:.2f}, CLBBF={clbbf_mean:.2f}+/-{clbbf_se:.2f}")
        table_data_b.append([
            f"{l}", 
            f"{pulse_mean:.2f} +/- {pulse_se:.2f}",
            f"{clbbf_mean:.2f} +/- {clbbf_se:.2f}"
        ])

    # Print Table B
    print("\n" + "="*70)
    print("Table 2B: Robustness to Varying Lag (Feature Shift)")
    print("="*70)
    col_widths_b = [max(len(str(item)) for item in col) for col in zip(*table_data_b)]
    header_row_b = table_data_b[0]
    header_str_b = " | ".join(f"{item:<{width}}" for item, width in zip(header_row_b, col_widths_b))
    print(header_str_b)
    print("-" * len(header_str_b))
    for row in table_data_b[1:]:
        print(" | ".join(f"{item:<{width}}" for item, width in zip(row, col_widths_b)))
    print("="*70 + "\n")

if __name__ == "__main__":
    run_robustness_analysis(seed=666)