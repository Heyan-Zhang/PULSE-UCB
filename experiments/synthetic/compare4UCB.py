import numpy as np
from statsmodels.tsa.arima_process import arma_generate_sample
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
import warnings
import math
from numba import jit, prange
import os
warnings.filterwarnings('ignore')

# Plotting style
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11
sns.set_style("whitegrid")
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs", "synthetic")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================================
# CLBBF Helper Functions
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
# 1. Linear Config
# ==================================
class LinearConfig:
    def __init__(self):
        np.random.seed(42)
        self.N_TRAJECTORIES = 1000
        self.T_TRAJECTORY = 100
        self.N_TIME_STEPS = 1200
        self.N_EXPERIMENTS = 30
        self.N_ACTIONS = 2
        self.MC_SAMPLES = 30
        self.AR_PARAMS = np.array([0.75, -0.25])
        self.MA_PARAMS = np.array([0.65, 0.35])
        self.STATE_NOISE_STD = 0.1
        self.BETA_FEATURE_DIM = 2
        self.TRUE_BETA = np.random.randn(self.BETA_FEATURE_DIM)
        self.S_PRIME_NOISE_STD = 0.1
        self.THETA_FEATURE_DIM = 4
        self.BENCHMARK_FEATURE_DIM = 2
        self.TRUE_THETA = np.random.randn(self.THETA_FEATURE_DIM)
        self.BENCHMARK_THETA = np.random.randn(self.BENCHMARK_FEATURE_DIM)
        self.REWARD_NOISE_STD = 0.05

# ==================================
# 1b. nonlinear Config
# ==================================
class nonlinearConfig(LinearConfig):
    def __init__(self, gamma=1.0):
        super().__init__()
        self.GAMMA = gamma

# ==================================
# 2. Feature Functions
# ==================================
def create_features_for_S_prime(s_history, config):
    if len(s_history) < 3:
        return np.zeros(config.BETA_FEATURE_DIM)
    features = np.zeros(config.BETA_FEATURE_DIM)
    features[0] = 1.0
    features[1] = np.mean(s_history[-3:])
    return features

def compute_S_prime(s_history, config):
    features = create_features_for_S_prime(s_history, config)
    return np.dot(config.TRUE_BETA, features) + np.random.normal(0, config.S_PRIME_NOISE_STD)

def compute_S_prime_nonlinear(s_history, config):
    base_val = compute_S_prime(s_history, config)
    if len(s_history) < 3:
        return base_val
    avg_recent = np.mean(s_history[-3:])
    return base_val + config.GAMMA * np.sin(avg_recent)

def create_OFUL_Full_features(s_t, s_prime_t, action, config):
    features = np.zeros(config.THETA_FEATURE_DIM)
    features[0] = 1.0
    features[1] = s_t
    features[2] = s_prime_t
    features[3] = s_t * action
    return features

def create_OFUL_features(s_t, action, config):
    features = np.zeros(config.BENCHMARK_FEATURE_DIM)
    features[0] = 1.0
    features[1] = s_t * action
    return features

# ==================================
# 3. Beta Estimator
# ==================================
def train_beta_estimator(X_beta, y_s_prime):
    model = LinearRegression()
    model.fit(X_beta, y_s_prime)
    predictions = model.predict(X_beta)
    residuals = y_s_prime - predictions
    prediction_std = np.std(residuals)
    return {"model": model, "prediction_std": prediction_std}

# ==================================
# 4. Agents
# ==================================
class OFUL_FullLinUCBAgent:
    def __init__(self, config, alpha=2.0):
        self.config = config
        self.alpha = alpha
        self.A = np.identity(config.THETA_FEATURE_DIM)
        self.b = np.zeros((config.THETA_FEATURE_DIM, 1))
        self.action_history = []
        self.reward_history = []

    def choose_action(self, s_t, s_prime_t):
        if np.linalg.det(self.A) < 1e-10:
            self.A += 1e-6 * np.identity(self.A.shape[0])
        A_inv = np.linalg.inv(self.A)
        theta_hat = A_inv @ self.b
        ucb_scores = np.zeros(self.config.N_ACTIONS)
        for a in range(self.config.N_ACTIONS):
            x_ta = create_OFUL_Full_features(s_t, s_prime_t, a, self.config).reshape(-1, 1)
            mean_reward = (theta_hat.T @ x_ta).item()
            confidence_width = self.alpha * np.sqrt(x_ta.T @ A_inv @ x_ta).item()
            ucb_scores[a] = mean_reward + confidence_width
        chosen_action = np.argmax(ucb_scores)
        self.action_history.append(chosen_action)
        return chosen_action

    def update(self, action, reward, s_t, s_prime_t):
        self.reward_history.append(reward)
        x_chosen = create_OFUL_Full_features(s_t, s_prime_t, action, self.config).reshape(-1, 1)
        self.A += x_chosen @ x_chosen.T
        self.b += reward * x_chosen

class OFULLinUCBAgent:
    def __init__(self, config, alpha=2.0):
        self.config = config
        self.alpha = alpha
        self.A = np.identity(config.BENCHMARK_FEATURE_DIM)
        self.b = np.zeros((config.BENCHMARK_FEATURE_DIM, 1))
        self.action_history = []
        self.reward_history = []

    def choose_action(self, s_t):
        if np.linalg.det(self.A) < 1e-10:
            self.A += 1e-6 * np.identity(self.A.shape[0])
        A_inv = np.linalg.inv(self.A)
        theta_hat = A_inv @ self.b
        ucb_scores = np.zeros(self.config.N_ACTIONS)
        for a in range(self.config.N_ACTIONS):
            x_ta = create_OFUL_features(s_t, a, self.config).reshape(-1, 1)
            mean_reward = (theta_hat.T @ x_ta).item()
            confidence_width = self.alpha * np.sqrt(x_ta.T @ A_inv @ x_ta).item()
            ucb_scores[a] = mean_reward + confidence_width
        chosen_action = np.argmax(ucb_scores)
        self.action_history.append(chosen_action)
        return chosen_action

    def update(self, action, reward, s_t):
        self.reward_history.append(reward)
        x_chosen = create_OFUL_features(s_t, action, self.config).reshape(-1, 1)
        self.A += x_chosen @ x_chosen.T
        self.b += reward * x_chosen

class PartialObservationLinUCBAgent:
    def __init__(self, config, beta_estimator, alpha=2.0):
        self.config = config
        self.beta_estimator = beta_estimator
        self.alpha = alpha
        self.A = np.identity(config.THETA_FEATURE_DIM)
        self.b = np.zeros((config.THETA_FEATURE_DIM, 1))
        self.action_history = []
        self.reward_history = []

    def predict_s_prime_with_uncertainty(self, s_history, n_samples=None):
        if n_samples is None:
            n_samples = self.config.MC_SAMPLES
        features = create_features_for_S_prime(s_history, self.config)
        samples = []
        for _ in range(n_samples):
            mean_pred = self.beta_estimator["model"].predict(features.reshape(1, -1))[0]
            uncertainty_std = self.beta_estimator["prediction_std"]
            sampled_s_prime = np.random.normal(mean_pred, uncertainty_std)
            samples.append(sampled_s_prime)
        return np.array(samples)

    def compute_expected_features_mc(self, s_history, action):
        s_t = s_history[-1]
        s_prime_samples = self.predict_s_prime_with_uncertainty(s_history)
        feature_samples = []
        for s_prime_sample in s_prime_samples:
            features = create_OFUL_Full_features(s_t, s_prime_sample, action, self.config)
            feature_samples.append(features)
        return np.mean(feature_samples, axis=0)

    def choose_action(self, s_history):
        if np.linalg.det(self.A) < 1e-10:
            self.A += 1e-6 * np.identity(self.A.shape[0])
        A_inv = np.linalg.inv(self.A)
        theta_hat = A_inv @ self.b
        ucb_scores = np.zeros(self.config.N_ACTIONS)
        for a in range(self.config.N_ACTIONS):
            x_ta = self.compute_expected_features_mc(s_history, a).reshape(-1, 1)
            mean_reward = (theta_hat.T @ x_ta).item()
            confidence_width = self.alpha * np.sqrt(x_ta.T @ A_inv @ x_ta).item()
            ucb_scores[a] = mean_reward + confidence_width
        chosen_action = np.argmax(ucb_scores)
        self.action_history.append(chosen_action)
        return chosen_action

    def update(self, action, reward, s_history):
        self.reward_history.append(reward)
        x_chosen = self.compute_expected_features_mc(s_history, action).reshape(-1, 1)
        self.A += x_chosen @ x_chosen.T
        self.b += reward * x_chosen

class CLBBFAgent:
    def __init__(self, config, alpha=2.0):
        self.config = config
        self.alpha = alpha
        self.d = config.THETA_FEATURE_DIM - 1  # Exclude intercept
        self.K = config.N_ACTIONS
        self.T = config.N_TIME_STEPS
        
        self.n = 0
        self.Z = np.zeros((self.d, self.d))
        self.xi = np.zeros(self.d)
        
        self.V = (self.d + 1) * math.log(self.K * self.T) * np.identity(self.d + 1)
        self.xy = np.zeros(self.d + 1)
        
        self.x_his = []
        self.m_his = []
        self.r = []
        self.x_sum = 0
        self.t = 0
        
        self.action_history = []
        self.reward_history = []

    def choose_action(self, s_t, s_prime_t):
        # Create context features for all actions (mask s_prime as unobserved)
        x_t = np.zeros((self.K, self.d))
        m_t = np.zeros((self.K, self.d))
        
        for a in range(self.K):
            features = create_OFUL_Full_features(s_t, s_prime_t, a, self.config)
            # Remove intercept for CLBBF
            x_t[a] = features[1:]
            # Mark s_t and action features as observed, s_prime as unobserved
            m_t[a, 0] = 1  # s_t is observed
            m_t[a, 1] = 0  # s_prime is unobserved
            m_t[a, 2] = 1  # s_t * action is observed
        
        # Update statistics
        for k in range(self.K):
            self.n += np.sum(m_t[k])
            self.Z += np.outer(x_t[k], x_t[k])
            self.xi += x_t[k]
        
        # Get estimators
        Kt = self.K * (self.t + 1)
        self.p_hat, self.nu_hat, self.Sigma_hat = _CLBEF_get_estimators(
            self.d, self.K, x_t, m_t, Kt, self.xi, self.n, self.Z
        )
        
        # Compute x_hat
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
                    if s < len(self.r):
                        self.xy += x_bar * self.r[s]
                
                self.V_inv = np.linalg.pinv(self.V)
                
                for s in range(len(self.x_his)):
                    x_bar = np.insert(self._x_bars(self.x_his[s], self.m_his[s]), 0, 1)
                    self.x_sum += np.sqrt(x_bar @ self.V_inv @ x_bar)
            
            self.V_inv = np.linalg.pinv(self.V)
            self.theta_hat = self.V_inv @ self.xy.T
            chosen_action, _ = _CLBEF_UCB(self.x_hat, self.t, self.K, self.theta_hat, self.d, self.T, self.V_inv, self.p_hat, self.x_sum)
        
        self.action_history.append(chosen_action)
        self._current_x_t = x_t.copy()
        self._current_m_t = m_t.copy()
        
        return chosen_action

    def update(self, action, reward, s_t, s_prime_t):
        self.reward_history.append(reward)
        self.r.append(reward)
        
        if hasattr(self, '_current_x_t') and hasattr(self, '_current_m_t'):
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
# 5. Data Generation
# ==================================
def generate_single_trajectory(config, length):
    ar = np.r_[1, -config.AR_PARAMS]
    ma = np.r_[1, config.MA_PARAMS]
    s_sequence = arma_generate_sample(ar=ar, ma=ma, nsample=length, scale=config.STATE_NOISE_STD)
    return s_sequence

def generate_pretraining_data(config):
    X_beta, y_s_prime = [], []
    for traj in range(config.N_TRAJECTORIES):
        s_sequence = generate_single_trajectory(config, config.T_TRAJECTORY)
        for t in range(3, len(s_sequence)):
            s_history = list(s_sequence[:t+1])
            features = create_features_for_S_prime(s_history, config)
            s_prime = compute_S_prime(s_history, config)
            X_beta.append(features)
            y_s_prime.append(s_prime)
    return np.array(X_beta), np.array(y_s_prime)

def compute_true_reward(s_t, s_prime_t, action, config):
    OFUL_Full_features = create_OFUL_Full_features(s_t, s_prime_t, action, config)
    return np.dot(config.TRUE_THETA, OFUL_Full_features) + np.random.normal(0, config.REWARD_NOISE_STD)

def get_optimal_reward(s_t, s_prime_t, config):
    potential_rewards = []
    for a in range(config.N_ACTIONS):
        OFUL_Full_features = create_OFUL_Full_features(s_t, s_prime_t, a, config)
        expected_reward = np.dot(config.TRUE_THETA, OFUL_Full_features)
        potential_rewards.append(expected_reward)
    return np.max(potential_rewards)

# ==================================
# 6. Experiment Runners
# ==================================
def run_OFUL_Full_experiment(config, alpha=2.0, use_nonlinear=False):
    agent = OFUL_FullLinUCBAgent(config, alpha)
    s_sequence = generate_single_trajectory(config, config.N_TIME_STEPS)
    rewards_log = np.zeros(config.N_TIME_STEPS)
    regret_log = np.zeros(config.N_TIME_STEPS)
    for t in range(3, config.N_TIME_STEPS):
        s_history = list(s_sequence[:t+1])
        s_t = s_history[-1]
        s_prime_t = compute_S_prime_nonlinear(s_history, config) if use_nonlinear else compute_S_prime(s_history, config)
        chosen_action = agent.choose_action(s_t, s_prime_t)
        observed_reward = compute_true_reward(s_t, s_prime_t, chosen_action, config)
        optimal_reward = get_optimal_reward(s_t, s_prime_t, config)
        agent.update(chosen_action, observed_reward, s_t, s_prime_t)
        rewards_log[t] = observed_reward
        regret_log[t] = optimal_reward - observed_reward
    return {'rewards': rewards_log, 'regret': regret_log, 'cumulative_regret': np.cumsum(regret_log)}

def run_OFUL_experiment(config, alpha=2.0, use_nonlinear=False):
    agent = OFULLinUCBAgent(config, alpha)
    s_sequence = generate_single_trajectory(config, config.N_TIME_STEPS)
    rewards_log = np.zeros(config.N_TIME_STEPS)
    regret_log = np.zeros(config.N_TIME_STEPS)
    for t in range(3, config.N_TIME_STEPS):
        s_history = list(s_sequence[:t+1])
        s_t = s_history[-1]
        s_prime_t = compute_S_prime_nonlinear(s_history, config) if use_nonlinear else compute_S_prime(s_history, config)
        chosen_action = agent.choose_action(s_t)
        observed_reward = compute_true_reward(s_t, s_prime_t, chosen_action, config)
        optimal_reward = get_optimal_reward(s_t, s_prime_t, config)
        agent.update(chosen_action, observed_reward, s_t)
        rewards_log[t] = observed_reward
        regret_log[t] = optimal_reward - observed_reward
    return {'rewards': rewards_log, 'regret': regret_log, 'cumulative_regret': np.cumsum(regret_log)}

def run_partial_observation_experiment(config, alpha=2.0, use_nonlinear=False):
    X_beta, y_s_prime = generate_pretraining_data(config)
    beta_estimator = train_beta_estimator(X_beta, y_s_prime)
    agent = PartialObservationLinUCBAgent(config, beta_estimator, alpha)
    s_sequence = generate_single_trajectory(config, config.N_TIME_STEPS)
    rewards_log = np.zeros(config.N_TIME_STEPS)
    regret_log = np.zeros(config.N_TIME_STEPS)
    for t in range(3, config.N_TIME_STEPS):
        s_history = list(s_sequence[:t+1])
        s_t = s_history[-1]
        s_prime_t = compute_S_prime_nonlinear(s_history, config) if use_nonlinear else compute_S_prime(s_history, config)
        chosen_action = agent.choose_action(s_history)
        observed_reward = compute_true_reward(s_t, s_prime_t, chosen_action, config)
        optimal_reward = get_optimal_reward(s_t, s_prime_t, config)
        agent.update(chosen_action, observed_reward, s_history)
        rewards_log[t] = observed_reward
        regret_log[t] = optimal_reward - observed_reward
    return {'rewards': rewards_log, 'regret': regret_log, 'cumulative_regret': np.cumsum(regret_log)}

def run_CLBBF_experiment(config, alpha=2.0, use_nonlinear=False):
    agent = CLBBFAgent(config, alpha)
    s_sequence = generate_single_trajectory(config, config.N_TIME_STEPS)
    rewards_log = np.zeros(config.N_TIME_STEPS)
    regret_log = np.zeros(config.N_TIME_STEPS)
    for t in range(3, config.N_TIME_STEPS):
        s_history = list(s_sequence[:t+1])
        s_t = s_history[-1]
        s_prime_t = compute_S_prime_nonlinear(s_history, config) if use_nonlinear else compute_S_prime(s_history, config)
        chosen_action = agent.choose_action(s_t, s_prime_t)
        observed_reward = compute_true_reward(s_t, s_prime_t, chosen_action, config)
        optimal_reward = get_optimal_reward(s_t, s_prime_t, config)
        agent.update(chosen_action, observed_reward, s_t, s_prime_t)
        rewards_log[t] = observed_reward
        regret_log[t] = optimal_reward - observed_reward
    return {'rewards': rewards_log, 'regret': regret_log, 'cumulative_regret': np.cumsum(regret_log)}

# ==================================
# 7. Comparison Visualization
# ==================================
def run_comparison_analysis(config, methods, setting_name="linear"):
    print(f"\nRunning comparison for {setting_name} setting...")
    results = {}
    colors = {
        'OFUL-Full': 'green',
        'OFUL': 'red',
        'PULSE-UCB': 'blue',
        'CLBBF': 'purple'
    }
    for method_name, run_func in methods.items():
        all_cumulative_regrets = []
        all_rewards = []
        for exp_run in range(config.N_EXPERIMENTS):
            result = run_func(config)
            all_cumulative_regrets.append(result['cumulative_regret'])
            all_rewards.append(result['rewards'])
        mean_cumulative_regret = np.mean(all_cumulative_regrets, axis=0)
        std_cumulative_regret = np.std(all_cumulative_regrets, axis=0)
        mean_rewards = np.mean(all_rewards, axis=0)
        results[method_name] = {
            'mean_cumulative_regret': mean_cumulative_regret,
            'std_cumulative_regret': std_cumulative_regret,
            'mean_rewards': mean_rewards,
            'color': colors[method_name]
        }
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Cumulative Regret
    ax1 = axes[0]
    for method_name, data in results.items():
        time_steps = range(len(data['mean_cumulative_regret']))
        ax1.plot(time_steps, data['mean_cumulative_regret'], color=data['color'], linewidth=3, label=method_name, alpha=0.8)
        ax1.fill_between(time_steps, data['mean_cumulative_regret'] - data['std_cumulative_regret'],
                         data['mean_cumulative_regret'] + data['std_cumulative_regret'],
                         color=data['color'], alpha=0.2)
    ax1.set_title('Cumulative Regret', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Time Steps', fontsize=18) 
    ax1.set_ylabel('Cumulative Regret', fontsize=18)
    ax1.legend(fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.tick_params(axis='both', which='major', labelsize=18)
    
    # 2. Moving Avg Reward
    ax2 = axes[1]
    window_size = 100
    for method_name, data in results.items():
        rewards = data['mean_rewards'][data['mean_rewards'] != 0]
        if len(rewards) > window_size:
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            time_steps_ma = range(window_size-1, len(rewards))
            ax2.plot(time_steps_ma, moving_avg, color=data['color'], linewidth=3, label=method_name, alpha=0.8)
    ax2.set_title('Moving Average Reward', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Time Steps', fontsize=18) 
    ax2.set_ylabel(f'Reward ({window_size}-step avg)', fontsize=18) 
    ax2.legend(fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.tick_params(axis='both', which='major', labelsize=18)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88) 

    plt.savefig(os.path.join(OUTPUT_DIR, f'Comparison_{setting_name}.png'), dpi=300, bbox_inches='tight')
    #plt.show()
    
    format_markdown_table_from_results(results, config, setting_name)
    
    return results

def format_markdown_table_from_results(results, config, setting_name="linear"):
    """
    
    Time step | PULSE-UCB | OFUL | CLBBF | OFUL-Full
    1         | x.x +/- y.y | ...
    10        | ...
    100       | ...
    300       | ...
    800       | ...
    1000      | ...
    """
    import numpy as np

    preferred_order = ['PULSE-UCB', 'OFUL', 'CLBBF', 'OFUL-Full']
    methods_order = [m for m in preferred_order if m in results] + \
                    [m for m in results.keys() if m not in preferred_order]

    line = "=" * 100

    if setting_name:
        title = f"Cumulate Regret Comparison of algorithms in the {setting_name} setting"
    else:
        title = "Cumulate Regret Comparison of algorithms"

    print(line)
    print(f"Table 1: {title}")
    print(line)
    header_cells = [f"{'Time step':<8}"] + [f"{m:<13}" for m in methods_order]
    print(" | ".join(header_cells))
    print("-" * 100)

    T = len(next(iter(results.values()))['mean_cumulative_regret'])
    candidate_steps = [1, 10, 100, 300, 800, 1000]
    time_steps_to_show = [t for t in candidate_steps if t <= T]

    for t in time_steps_to_show:
        idx = t  # time step t corresponds to index t
        if idx < 0 or idx >= T:
            continue

        row_cells = [f"{t:<8}"]
        for m in methods_order:
            data = results[m]
            mean_val = data['mean_cumulative_regret'][idx]
            std_val = data['std_cumulative_regret'][idx]
            cell = f"{mean_val:.1f} +/- {std_val:.1f}"
            row_cells.append(f"{cell:<13}")
        print(" | ".join(row_cells))

    print(line)


# ==================================
# 8. Main
# ==================================
def main():
    linear_config = LinearConfig()
    linear_methods = {
        'OFUL-Full': lambda cfg: run_OFUL_Full_experiment(cfg, use_nonlinear=False),
        'OFUL': lambda cfg: run_OFUL_experiment(cfg, use_nonlinear=False),
        'PULSE-UCB': lambda cfg: run_partial_observation_experiment(cfg, use_nonlinear=False),
        'CLBBF': lambda cfg: run_CLBBF_experiment(cfg, use_nonlinear=False)
    }
    run_comparison_analysis(linear_config, linear_methods, setting_name="linear")

    nonlinear_config = nonlinearConfig(gamma=4)  # gamma is variable
    nonlinear_methods = {
        'OFUL-Full': lambda cfg: run_OFUL_Full_experiment(cfg, use_nonlinear=True),
        'OFUL': lambda cfg: run_OFUL_experiment(cfg, use_nonlinear=True),
        'PULSE-UCB': lambda cfg: run_partial_observation_experiment(cfg, use_nonlinear=True),
        'CLBBF': lambda cfg: run_CLBBF_experiment(cfg, use_nonlinear=True)
    }
    run_comparison_analysis(nonlinear_config, nonlinear_methods, setting_name="nonlinear")


if __name__ == "__main__":
    main()
