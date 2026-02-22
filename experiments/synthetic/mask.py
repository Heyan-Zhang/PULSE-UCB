import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima_process import arma_generate_sample
import matplotlib.pyplot as plt
import seaborn as sns
import math
from numba import jit, prange
import warnings
import os

# Suppress warnings for cleaner output
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
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs", "synthetic")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================================
# 0. Numba Helpers for CLBBF
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
# 1. Configuration
# ==================================
class MaskingConfig:
    def __init__(self, seed=42):
        self.SEED = seed
        np.random.seed(seed)  # Set seed for reproducibility
        
        # High-Dim / Data-Poor Setting
        self.N_TRAJECTORIES = 1000
        self.T_TRAJECTORY = 25
        self.N_TIME_STEPS = 1000
        self.N_EXPERIMENTS = 20  # Averaging over 20 runs
        self.N_ACTIONS = 2
        
        self.AR_PARAMS = np.array([0.75, -0.25])
        self.MA_PARAMS = np.array([0.65, 0.35])
        self.STATE_NOISE_STD = 0.1
        
        self.BETA_FEATURE_DIM = 20  # High dimension (20)
        self.TRUE_BETA = np.random.randn(self.BETA_FEATURE_DIM)
        self.S_PRIME_NOISE_STD = 0.1
        
        self.THETA_FEATURE_DIM = 4
        # Interaction depends on S' (index 3 corresponds to s_prime * action in Full features)
        self.TRUE_THETA = np.array([0.1, 0.1, 0.1, 1.0]) 
        self.REWARD_NOISE_STD = 0.05

# ==================================
# 2. Helper Functions
# ==================================
def create_features_for_S_prime(s_history, config):
    required_history = config.BETA_FEATURE_DIM
    if len(s_history) < required_history:
        return np.zeros(config.BETA_FEATURE_DIM)
    features = np.zeros(config.BETA_FEATURE_DIM)
    features[0] = 1.0
    # Lag features: most recent 'dim-1' observations
    features[1:] = s_history[-(required_history - 1):]
    return features

def compute_S_prime(s_history, config):
    features = create_features_for_S_prime(s_history, config)
    return np.dot(config.TRUE_BETA, features) + np.random.normal(0, config.S_PRIME_NOISE_STD)

def create_OFUL_Full_features(s_t, s_prime_t, action, config):
    features = np.zeros(config.THETA_FEATURE_DIM)
    features[0] = 1.0
    features[1] = s_t
    features[2] = s_prime_t
    features[3] = s_prime_t * action 
    return features

def compute_true_reward(s_t, s_prime_t, action, config):
    feats = create_OFUL_Full_features(s_t, s_prime_t, action, config)
    return np.dot(config.TRUE_THETA, feats) + np.random.normal(0, config.REWARD_NOISE_STD)

def get_optimal_reward(s_t, s_prime_t, config):
    return np.max([np.dot(config.TRUE_THETA, create_OFUL_Full_features(s_t, s_prime_t, a, config)) 
                   for a in range(config.N_ACTIONS)])

def generate_single_trajectory(config, length):
    ar = np.r_[1, -config.AR_PARAMS]
    ma = np.r_[1, config.MA_PARAMS]
    return arma_generate_sample(ar=ar, ma=ma, nsample=length, scale=config.STATE_NOISE_STD)

# ==================================
# 3. Masking Logic & Robust Estimator
# ==================================
def apply_mask(features, mask_prob):
    """
    Randomly zero out features with probability mask_prob.
    Always keep the intercept (index 0).
    """
    if mask_prob <= 0:
        return features
    
    # Generate mask for indices 1 to end
    mask_val = np.random.binomial(1, 1 - mask_prob, size=len(features)-1)
    mask = np.ones(len(features))
    mask[1:] = mask_val
    
    return features * mask

def train_robust_beta_estimator(config, mask_prob_range=(0.0, 0.5)):
    X_beta, y_s_prime = [], []
    for _ in range(config.N_TRAJECTORIES):
        s_sequence = generate_single_trajectory(config, config.T_TRAJECTORY)
        start_index = config.BETA_FEATURE_DIM
        for t in range(start_index, len(s_sequence)):
            s_history = list(s_sequence[:t+1])
            feat = create_features_for_S_prime(s_history, config)
            target = compute_S_prime(s_history, config)
            
            # 1. Clean data
            X_beta.append(feat)
            y_s_prime.append(target)
            
            # 2. Augmented (Corrupted) data for robustness
            # Add 1 corrupted copy per sample
            for _ in range(1): 
                p = np.random.uniform(mask_prob_range[0], mask_prob_range[1])
                masked_feat = apply_mask(feat.copy(), p)
                X_beta.append(masked_feat)
                y_s_prime.append(target)
                
    model = LinearRegression()
    model.fit(X_beta, y_s_prime)
    
    # Calculate std using clean data for simplicity
    preds = model.predict(X_beta)
    std = np.std(np.array(y_s_prime) - preds)
    return {"model": model, "prediction_std": std}

# ==================================
# 4. Agent Classes
# ==================================

# --- 1. Robust PULSE Agent ---
class RobustPULSEAgent:
    def __init__(self, config, beta_estimator, alpha=2.0):
        self.config = config
        self.estimator = beta_estimator
        self.alpha = alpha
        self.A = np.identity(config.THETA_FEATURE_DIM)
        self.b = np.zeros((config.THETA_FEATURE_DIM, 1))

    def choose_action(self, s_t, masked_features):
        # Predict S' using the *masked* features
        pred_mean = self.estimator["model"].predict(masked_features.reshape(1, -1))[0]
        
        # Sample for uncertainty (PULSE mechanism)
        samples = np.random.normal(pred_mean, self.estimator["prediction_std"], self.config.N_ACTIONS) 
        
        ucb_scores = np.zeros(self.config.N_ACTIONS)
        A_inv = np.linalg.inv(self.A)
        theta_hat = A_inv @ self.b
        
        for a in range(self.config.N_ACTIONS):
            x_ta = create_OFUL_Full_features(s_t, samples[a], a, self.config).reshape(-1, 1)
            mean = (theta_hat.T @ x_ta).item()
            conf = self.alpha * np.sqrt(x_ta.T @ A_inv @ x_ta).item()
            ucb_scores[a] = mean + conf
        return np.argmax(ucb_scores)

    def update(self, action, reward, s_t, masked_features):
        # Re-predict expected S' for update
        pred_s_prime = self.estimator["model"].predict(masked_features.reshape(1, -1))[0]
        x_chosen = create_OFUL_Full_features(s_t, pred_s_prime, action, self.config).reshape(-1, 1)
        self.A += x_chosen @ x_chosen.T
        self.b += reward * x_chosen

# --- 2. OFUL Agent (Naive) ---
class OFULAgent: 
    def __init__(self, config, alpha=2.0):
        self.config = config
        self.alpha = alpha
        # Naive OFUL usually assumes lower dimension, but here we just use what it sees
        # In the benchmark setting, OFUL sees [1, s_t * a]
        self.dim = 2 
        self.A = np.identity(self.dim)
        self.b = np.zeros((self.dim, 1))
        
    def choose_action(self, s_t):
        A_inv = np.linalg.inv(self.A)
        theta_hat = A_inv @ self.b
        ucb_scores = np.zeros(self.config.N_ACTIONS)
        for a in range(self.config.N_ACTIONS):
            feat = np.zeros(self.dim)
            feat[0] = 1.0; feat[1] = s_t * a
            x_ta = feat.reshape(-1, 1)
            mean = (theta_hat.T @ x_ta).item()
            conf = self.alpha * np.sqrt(x_ta.T @ A_inv @ x_ta).item()
            ucb_scores[a] = mean + conf
        return np.argmax(ucb_scores)

    def update(self, action, reward, s_t):
        feat = np.zeros(self.dim)
        feat[0] = 1.0; feat[1] = s_t * action
        x_chosen = feat.reshape(-1, 1)
        self.A += x_chosen @ x_chosen.T
        self.b += reward * x_chosen

# --- 3. OFUL-Full Agent (Oracle) ---
class OFULFullAgent:
    def __init__(self, config, alpha=2.0):
        self.config = config
        self.alpha = alpha
        self.A = np.identity(config.THETA_FEATURE_DIM)
        self.b = np.zeros((config.THETA_FEATURE_DIM, 1))
    
    def choose_action(self, s_t, s_prime_true):
        A_inv = np.linalg.inv(self.A)
        theta_hat = A_inv @ self.b
        ucb_scores = np.zeros(self.config.N_ACTIONS)
        for a in range(self.config.N_ACTIONS):
            x_ta = create_OFUL_Full_features(s_t, s_prime_true, a, self.config).reshape(-1, 1)
            mean = (theta_hat.T @ x_ta).item()
            conf = self.alpha * np.sqrt(x_ta.T @ A_inv @ x_ta).item()
            ucb_scores[a] = mean + conf
        return np.argmax(ucb_scores)
    
    def update(self, action, reward, s_t, s_prime_true):
        x_chosen = create_OFUL_Full_features(s_t, s_prime_true, action, self.config).reshape(-1, 1)
        self.A += x_chosen @ x_chosen.T
        self.b += reward * x_chosen

# --- 4. CLBBF Agent ---
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
        # CLBBF logic: treats s_prime_t as latent (unobserved)
        x_t = np.zeros((self.K, self.d))
        m_t = np.zeros((self.K, self.d))
        for a in range(self.K):
            # Full features [1, s_t, s', s'*a]
            # CLBBF excludes intercept, so indices are 0->s_t, 1->s', 2->s'*a
            features = create_OFUL_Full_features(s_t, s_prime_t, a, self.config)
            x_t[a] = features[1:]
            
            # Masking: 1 means observed, 0 means unobserved
            m_t[a, 0] = 1 # s_t is observed
            m_t[a, 1] = 0 # s_prime is MISSING
            m_t[a, 2] = 1 # We assume interaction term is partially observed? 
            # In compare4UCB.py, m_t[a, 2] = 1. This means CLBBF assumes it knows the interaction column exists, 
            # Ideally CLBBF imputes the missing parts. 
            
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
# 5. Main Execution
# ==================================
def run_masking_experiment():
    # Initialize Config with Seed
    config = MaskingConfig(seed=42)
    
    # Define Mask probabilities
    mask_probs = [0.0, 0.2, 0.4, 0.6, 0.8]
    
    print("Training Robust Beta Estimator (with data augmentation)...")
    robust_estimator = train_robust_beta_estimator(config, mask_prob_range=(0.0, 0.8))
    
    # Store results
    results = {'PULSE': [], 'OFUL': [], 'Oracle': [], 'CLBBF': []}
    
    print(f"\nRunning Time-Varying Masking Experiment (Dims={config.BETA_FEATURE_DIM})...")
    
    # Print Header
    header = f"{'Mask Prob':<10} | {'PULSE-UCB':<18} | {'OFUL-Full':<18} | {'OFUL':<18} | {'CLBBF':<18}"
    print(header)
    print("-" * len(header))
    
    for mp in mask_probs:
        r_pulse, r_oful, r_oracle, r_clbbf = [], [], [], []
        
        for _ in range(config.N_EXPERIMENTS):
            pulse = RobustPULSEAgent(config, robust_estimator)
            oful = OFULAgent(config)
            oracle = OFULFullAgent(config)
            clbbf = CLBBFAgent(config)
            
            s_seq = generate_single_trajectory(config, config.N_TIME_STEPS)
            reg_p, reg_o, reg_or, reg_c = 0, 0, 0, 0
            
            start_idx = config.BETA_FEATURE_DIM
            for t in range(start_idx, config.N_TIME_STEPS):
                s_hist = list(s_seq[:t+1])
                s_t = s_hist[-1]
                
                # Ground Truth Generation
                full_feats = create_features_for_S_prime(s_hist, config)
                s_prime_true = compute_S_prime(s_hist, config)
                
                # Apply Mask (Time-Varying)
                masked_feats = apply_mask(full_feats.copy(), mp)
                
                # Oracle Optimal for Regret
                opt = get_optimal_reward(s_t, s_prime_true, config)
                
                # 1. PULSE (Robust) - Uses MASKED features
                a = pulse.choose_action(s_t, masked_feats)
                r = compute_true_reward(s_t, s_prime_true, a, config)
                pulse.update(a, r, s_t, masked_feats)
                reg_p += (opt - r)
                
                # 2. OFUL (Naive) - Ignores features
                a = oful.choose_action(s_t)
                r = compute_true_reward(s_t, s_prime_true, a, config)
                oful.update(a, r, s_t)
                reg_o += (opt - r)
                
                # 3. OFUL-Full (Oracle) - Uses TRUE s_prime
                a = oracle.choose_action(s_t, s_prime_true)
                r = compute_true_reward(s_t, s_prime_true, a, config)
                oracle.update(a, r, s_t, s_prime_true)
                reg_or += (opt - r)
                
                # 4. CLBBF - Uses s_prime_t structure but treats it as missing
                a = clbbf.choose_action(s_t, s_prime_true)
                r = compute_true_reward(s_t, s_prime_true, a, config)
                clbbf.update(a, r, s_t, s_prime_true)
                reg_c += (opt - r)
                
            r_pulse.append(reg_p)
            r_oful.append(reg_o)
            r_oracle.append(reg_or)
            r_clbbf.append(reg_c)
            
        # Calc Stats
        means = [np.mean(x) for x in [r_pulse, r_oracle, r_oful, r_clbbf]]
        stds = [np.std(x) for x in [r_pulse, r_oracle, r_oful, r_clbbf]]
        
        results['PULSE'].append(means[0])
        results['Oracle'].append(means[1])
        results['OFUL'].append(means[2])
        results['CLBBF'].append(means[3])
        
        # Print Row with 2 Decimal Formatting
        row = f"{mp:<10} | {means[0]:.2f}+/-{stds[0]:.2f}      | {means[1]:.2f}+/-{stds[1]:.2f}      | {means[2]:.2f}+/-{stds[2]:.2f}      | {means[3]:.2f}+/-{stds[3]:.2f}"
        print(row)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(mask_probs, results['PULSE'], 'o-', label='PULSE-UCB (Robust)', linewidth=2.5, color='blue')
    plt.plot(mask_probs, results['Oracle'], 'x--', label='OFUL-Full (Oracle)', linewidth=2.5, color='green')
    plt.plot(mask_probs, results['CLBBF'], '^--', label='CLBBF', linewidth=2.5, color='purple')
    plt.plot(mask_probs, results['OFUL'], 's--', label='OFUL (Naive)', linewidth=2.5, color='red')
    
    plt.xlabel('Feature Missing Probability (Random Mask)', fontsize=14)
    plt.ylabel('Cumulative Regret (T=1000)', fontsize=14)
    plt.title('Robustness to Time-Varying Observation Subsets', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'masking_experiment_4agents.png'), dpi=300)
    print("\nExperiment finished. Plot saved to masking_experiment_4agents.png")

if __name__ == "__main__":
    run_masking_experiment()
