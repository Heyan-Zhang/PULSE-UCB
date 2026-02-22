import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima_process import arma_generate_sample
import matplotlib.pyplot as plt
import seaborn as sns
import math
from numba import jit, prange
import warnings
from scipy.stats import t as student_t
import scipy.signal

warnings.filterwarnings('ignore')

# ==================================
# Plotting Configuration
# ==================================
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")

# ==================================
# Numba Helpers
# ==================================
@jit(nopython=True)
def numba_ix(arr, rows, cols):
    """Helper for advanced indexing in Numba."""
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
    return chosen_arm

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
    def __init__(self):
        np.random.seed(42)
        
        # --- Balanced Experiment Settings ---
        self.N_TRAJECTORIES = 100  
        self.T_TRAJECTORY = 50    
        self.N_TIME_STEPS = 1000
        self.N_EXPERIMENTS = 10   
        self.N_ACTIONS = 2
        
        # --- Base ARMA Process Parameters ---
        self.AR_PARAMS = np.array([0.75, -0.25])
        self.MA_PARAMS = np.array([0.65, 0.35])
        self.STATE_NOISE_STD = 0.1 
        
        # --- High Dimensional Settings ---
        self.BETA_FEATURE_DIM = 20 
        self.TRUE_BETA = np.random.randn(self.BETA_FEATURE_DIM)
        
        self.S_PRIME_NOISE_STD = 0.1 
        
        self.THETA_FEATURE_DIM = 4
        self.BENCHMARK_FEATURE_DIM = 2
        
        # --- Reward Mechanism ---
        self.TRUE_THETA = np.array([0.1, 0.1, 0.1, 1.5]) 
        self.BENCHMARK_THETA = np.random.randn(self.BENCHMARK_FEATURE_DIM)
        self.REWARD_NOISE_STD = 0.05

# ==================================
# 2. Feature & Generation Functions
# ==================================
def create_features_for_S_prime(s_history, config):
    dim = config.BETA_FEATURE_DIM
    features = np.zeros(dim)
    features[0] = 1.0 # Bias term
    
    history_len = len(s_history)
    needed = dim - 1
    
    if history_len >= needed:
        features[1:] = s_history[-needed:]
    else:
        features[1 : 1+history_len] = s_history
        
    return features

def compute_S_prime(s_history, config):
    features = create_features_for_S_prime(s_history, config)
    return np.dot(config.TRUE_BETA, features) + np.random.normal(0, config.S_PRIME_NOISE_STD)

def create_OFUL_Full_features(s_t, s_prime_t, action, config):
    features = np.zeros(config.THETA_FEATURE_DIM)
    features[0] = 1.0
    features[1] = s_t
    features[2] = s_prime_t # W_t
    features[3] = s_prime_t * action 
    return features

def create_OFUL_features(s_t, action, config):
    features = np.zeros(config.BENCHMARK_FEATURE_DIM)
    features[0] = 1.0
    features[1] = s_t * action 
    return features

def generate_single_trajectory(config, length, noise_std=None):
    """Standard Gaussian ARMA generation (Base)"""
    if noise_std is None:
        noise_std = config.STATE_NOISE_STD
    ar = np.r_[1, -config.AR_PARAMS]
    ma = np.r_[1, config.MA_PARAMS]
    s_sequence = arma_generate_sample(ar=ar, ma=ma, nsample=length, scale=noise_std)
    return s_sequence

def generate_pretraining_data(config, fixed_noise=0.1):
    """Generates historical data (Base Distribution)"""
    original_noise = config.STATE_NOISE_STD
    config.STATE_NOISE_STD = fixed_noise
    
    X_beta, y_s_prime = [], []
    for traj in range(config.N_TRAJECTORIES):
        s_sequence = generate_single_trajectory(config, config.T_TRAJECTORY, noise_std=fixed_noise)
        start_index = config.BETA_FEATURE_DIM
        for t in range(start_index, len(s_sequence)):
            s_history = list(s_sequence[:t])
            features = create_features_for_S_prime(s_history, config)
            s_prime = compute_S_prime(s_history, config)
            X_beta.append(features)
            y_s_prime.append(s_prime)
            
    config.STATE_NOISE_STD = original_noise
    return np.array(X_beta), np.array(y_s_prime)

def train_beta_estimator(X_beta, y_s_prime):
    model = LinearRegression()
    if len(X_beta) > 0:
        model.fit(X_beta, y_s_prime)
    else:
        model.fit(np.zeros((1, X_beta.shape[1])), [0])
    residuals = y_s_prime - model.predict(X_beta) if len(X_beta) > 0 else [0]
    std = np.std(residuals) if len(residuals) > 0 else 1.0
    return {"model": model, "prediction_std": std}

# ==================================
# 3. Agents
# ==================================
class OFUL_FullLinUCBAgent:
    """Oracle Agent (OFUL-Full): Observes W_t"""
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
    """Benchmark Agent (OFUL): Observes only S_t"""
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
    """PULSE-UCB Agent: Infers W_t"""
    def __init__(self, config, beta_estimator, alpha=2.0):
        self.config = config
        self.beta_estimator = beta_estimator
        self.alpha = alpha
        self.A = np.identity(config.THETA_FEATURE_DIM)
        self.b = np.zeros((config.THETA_FEATURE_DIM, 1))
    def predict_s_prime(self, s_history):
        features = create_features_for_S_prime(s_history, self.config)
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
            chosen_action = _CLBEF_UCB(self.x_hat, self.t, self.K, self.theta_hat, self.d, self.T, self.V_inv, self.p_hat, self.x_sum)
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
# 4. Advanced Shift Logic
# ==================================

def generate_shifted_trajectory(config, shift_type, intensity):
    """
    Generates S_t under specific shifts.
    intensity: alpha [0.0, 1.0]
    """
    length = config.N_TIME_STEPS
    
    # Base AR (Fixed)
    base_ar = config.AR_PARAMS
    ma_params = config.MA_PARAMS
    
    # Defaults
    mean_shift = 0.0
    noise_scale = 0.1
    use_student_t = False
    
    if shift_type == 'Translation':
        # Mean Shift: Moves the centroid of the high-dim feature cloud
        mean_shift = intensity * 5.0 
        
    elif shift_type == 'Deformation':
        # Deformation: Changes density shape (Gaussian -> Heavy Tail) + Scale
        if intensity > 0.0:
            use_student_t = True
            noise_scale = 0.1 * (1 + intensity * 2) 

    ar_poly = np.r_[1, -base_ar]
    ma_poly = np.r_[1, ma_params]
    
    if use_student_t:
        # Generate heavy-tailed innovations (Student-t, df=3)
        dist = student_t(df=3)
        innovations = dist.rvs(size=length + 100) * noise_scale
        s_sequence = scipy.signal.lfilter(ma_poly, ar_poly, innovations)[100:]
    else:
        # Standard Gaussian ARMA
        s_sequence = arma_generate_sample(ar=ar_poly, ma=ma_poly, nsample=length, scale=noise_scale)
    
    # Apply Mean Shift
    s_sequence = s_sequence + mean_shift
    
    return s_sequence

def run_agent_on_sequence(config, agent_type, beta_estimator, s_sequence):
    """Runs an agent on the provided s_sequence."""
    regret_log = np.zeros(config.N_TIME_STEPS)
    start_index = config.BETA_FEATURE_DIM
    
    if agent_type == 'PULSE-UCB':
        agent = PartialObservationLinUCBAgent(config, beta_estimator)
    elif agent_type == 'OFUL':
        agent = OFULLinUCBAgent(config)
    elif agent_type == 'OFUL-Full':
        agent = OFUL_FullLinUCBAgent(config)
    elif agent_type == 'CLBBF':
        agent = CLBBFAgent(config)
    else:
        return 0 
    
    for t in range(start_index, config.N_TIME_STEPS):
        s_history = list(s_sequence[:t])
        s_t = s_sequence[t]
        s_prime_t = compute_S_prime(s_history, config) 
        
        if agent_type == 'PULSE-UCB':
            action = agent.choose_action(s_history)
        elif agent_type == 'OFUL':
            action = agent.choose_action(s_t)
        elif agent_type == 'OFUL-Full' or agent_type == 'CLBBF':
            action = agent.choose_action(s_t, s_prime_t)
            
        x_true = create_OFUL_Full_features(s_t, s_prime_t, action, config)
        reward = np.dot(config.TRUE_THETA, x_true) + np.random.normal(0, config.REWARD_NOISE_STD)
        
        if agent_type == 'PULSE-UCB':
            agent.update(action, reward, s_history)
        elif agent_type == 'OFUL':
            agent.update(action, reward, s_t)
        elif agent_type == 'OFUL-Full' or agent_type == 'CLBBF':
            agent.update(action, reward, s_t, s_prime_t)
            
        opt = np.max([np.dot(config.TRUE_THETA, create_OFUL_Full_features(s_t, s_prime_t, a, config)) for a in range(config.N_ACTIONS)])
        regret_log[t] = opt - reward
        
    return np.sum(regret_log)

# ==================================
# 5. Runner
# ==================================
def run_focused_shift_analysis():
    config = LinearConfig()
    
    # Define the two requested experiments
    experiments = {
        'Translation (Mean)': 'Translation',
        'Deformation (Scale/Geom)': 'Deformation'
    }
    
    shift_intensities = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    print("Training Beta Estimator on Base Distribution (Mean=0, Gaussian)...")
    config.AR_PARAMS = np.array([0.75, -0.25])
    X_beta, y_s_prime = generate_pretraining_data(config, fixed_noise=0.1)
    beta_estimator = train_beta_estimator(X_beta, y_s_prime)
    
    final_table = []
    headers = ["Exp Type", "Alpha", "Shift Detail", "PULSE-UCB", "OFUL", "OFUL-Full", "CLBBF"]
    final_table.append(headers)
    
    n_exp = config.N_EXPERIMENTS
    
    for exp_name, shift_type in experiments.items():
        print(f"\nRunning Experiment: {exp_name}...")
        
        for alpha in shift_intensities:
            regs_pulse, regs_oful, regs_oful_full, regs_clbbf = [], [], [], []
            
            for _ in range(n_exp):
                s_sequence = generate_shifted_trajectory(config, shift_type, alpha)
                
                r_pulse = run_agent_on_sequence(config, 'PULSE-UCB', beta_estimator, s_sequence)
                r_oful = run_agent_on_sequence(config, 'OFUL', None, s_sequence)
                r_oful_full = run_agent_on_sequence(config, 'OFUL-Full', None, s_sequence)
                r_clbbf = run_agent_on_sequence(config, 'CLBBF', None, s_sequence)
                
                regs_pulse.append(r_pulse)
                regs_oful.append(r_oful)
                regs_oful_full.append(r_oful_full)
                regs_clbbf.append(r_clbbf)
            
            # Formatting Details
            shift_detail = ""
            if shift_type == 'Translation':
                shift_detail = f"Mean +{alpha*5.0:.1f}"
            elif shift_type == 'Deformation':
                if alpha == 0: shift_detail = "Gaussian (0.1)"
                else: shift_detail = f"Student-t + Scale x{1+alpha*2:.1f}"
            
            # Format: Mean +/- SE
            # SE = Std / sqrt(N)
            pulse_mean, pulse_se = np.mean(regs_pulse), np.std(regs_pulse)/np.sqrt(n_exp)
            oful_mean, oful_se = np.mean(regs_oful), np.std(regs_oful)/np.sqrt(n_exp)
            full_mean, full_se = np.mean(regs_oful_full), np.std(regs_oful_full)/np.sqrt(n_exp)
            clbbf_mean, clbbf_se = np.mean(regs_clbbf), np.std(regs_clbbf)/np.sqrt(n_exp)
            
            pulse_res = f"{pulse_mean:.2f} +/- {pulse_se:.2f}"
            oful_res = f"{oful_mean:.2f} +/- {oful_se:.2f}"
            oful_full_res = f"{full_mean:.2f} +/- {full_se:.2f}"
            clbbf_res = f"{clbbf_mean:.2f} +/- {clbbf_se:.2f}"
            
            print(f"  Alpha {alpha}: PULSE={pulse_res}")
            final_table.append([exp_name, f"{alpha:.1f}", shift_detail, pulse_res, oful_res, oful_full_res, clbbf_res])

    # --- Print Table ---
    print("\n" + "="*145)
    print("High-Dimensional Shift Analysis: Translation & Deformation (Mean Cumulative Regret +/- Standard Error)")
    print("="*145)
    # Adjusted column widths to fit all columns
    col_widths = [22, 6, 25, 22, 22, 22, 22]
    
    header_str = " | ".join(f"{str(item):<{w}}" for item, w in zip(final_table[0], col_widths))
    print(header_str)
    print("-" * len(header_str))
    
    for row in final_table[1:]:
        print(" | ".join(f"{str(item):<{w}}" for item, w in zip(row, col_widths)))
    print("="*145)

if __name__ == "__main__":
    run_focused_shift_analysis()