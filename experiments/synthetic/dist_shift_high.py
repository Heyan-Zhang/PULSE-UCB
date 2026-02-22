import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima_process import arma_generate_sample
import matplotlib.pyplot as plt
import seaborn as sns
import math
from numba import jit, prange
import warnings
from scipy.linalg import sqrtm
import os

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
    """Calculate UCB for CLBBF agent."""
    return x_hat@theta_hat+(np.sqrt((d+1)*np.log((t+2)*T))+x_sum*d*((1/p_hat)**(3/2))*np.sqrt(np.log(T)*np.log(K*T)/((t+2)*K))+\
                            np.sqrt(d*np.log(K*T)))*np.sqrt(x_hat@V_inv@x_hat.T)

@jit(nopython=True, parallel=True)
def _CLBEF_UCB(x_hat, t, K, theta_hat, d, T, V_inv, p_hat, x_sum):
    """Parallelized UCB selection for CLBBF."""
    ucb_list = np.zeros(K)
    for k in prange(K):
        ucb_list[k] = _CLBEF_get_UCB(x_hat[k], t, K, theta_hat.copy(), d, T, V_inv.copy(), p_hat, x_sum)
    chosen_arm = np.argmax(ucb_list)
    max_ucb = ucb_list[chosen_arm]
    return chosen_arm, max_ucb

@jit(nopython=True)
def numba_idxSU(m):
    """Identify S (observed) and U (unobserved) indices."""
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
    """Reconstruct feature vector with imputed missing values."""
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
    """Batch processing for feature reconstruction."""
    for k in prange(K):
        x_hat_dummy[k, 1:] = _CLBEF_x_bars(nu_hat, Sigma_hat, x[k], m[k], x_bar_dummy[k])
    return x_hat_dummy

@jit(nopython=True)
def _CLBEF_get_estimators(d, K, x_t, m_t, Kt, xi, n, Z):
    """Update estimators for CLBBF."""
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
        self.MC_SAMPLES = 10
        
        # --- ARMA Process Parameters ---
        # Base AR params (used for training distribution / historical data)
        self.AR_PARAMS = np.array([0.75, -0.25])
        self.MA_PARAMS = np.array([0.65, 0.35])
        self.STATE_NOISE_STD = 0.1 
        
        # --- High Dimensional Settings ---
        # Dimension of observed history x_t = [1, S_{t-1}, ..., S_{t-19}]
        self.BETA_FEATURE_DIM = 20 
        self.TRUE_BETA = np.random.randn(self.BETA_FEATURE_DIM)
        
        self.S_PRIME_NOISE_STD = 0.1 
        
        self.THETA_FEATURE_DIM = 4
        self.BENCHMARK_FEATURE_DIM = 2
        
        # --- Reward Mechanism (Fixed) ---
        self.TRUE_THETA = np.array([0.1, 0.1, 0.1, 1.5]) 
        self.BENCHMARK_THETA = np.random.randn(self.BENCHMARK_FEATURE_DIM)
        self.REWARD_NOISE_STD = 0.05

# ==================================
# 2. Feature Functions
# ==================================
def create_features_for_S_prime(s_history, config):
    """
    Constructs the observed covariate vector x_t from history.
    x_t = (1, S_{t-1}, ..., S_{t-19}) in R^20.
    """
    dim = config.BETA_FEATURE_DIM
    features = np.zeros(dim)
    features[0] = 1.0 # Bias term
    
    # Extract last (dim-1) elements from history
    history_len = len(s_history)
    needed = dim - 1
    
    if history_len >= needed:
        features[1:] = s_history[-needed:]
    else:
        # Pad with available history if sequence is too short
        features[1 : 1+history_len] = s_history
        
    return features

def compute_S_prime(s_history, config):
    """
    Computes the Unobserved Context W_t.
    W_t = beta^T * x_t + xi_t
    The latent dynamics (beta) remain fixed throughout experiments.
    """
    features = create_features_for_S_prime(s_history, config)
    return np.dot(config.TRUE_BETA, features) + np.random.normal(0, config.S_PRIME_NOISE_STD)

def create_OFUL_Full_features(s_t, s_prime_t, action, config):
    """Constructs features for the oracle/full agent: Phi(Y_t, a_t)."""
    features = np.zeros(config.THETA_FEATURE_DIM)
    features[0] = 1.0
    features[1] = s_t
    features[2] = s_prime_t # W_t
    features[3] = s_prime_t * action 
    return features

def create_OFUL_features(s_t, action, config):
    """Constructs features for the benchmark agent (observed only)."""
    features = np.zeros(config.BENCHMARK_FEATURE_DIM)
    features[0] = 1.0
    features[1] = s_t * action 
    return features

# ==================================
# 3. Data Generation
# ==================================
def generate_single_trajectory(config, length, noise_std=None):
    """Generates a time-series trajectory using ARMA process."""
    if noise_std is None:
        noise_std = config.STATE_NOISE_STD
    ar = np.r_[1, -config.AR_PARAMS]
    ma = np.r_[1, config.MA_PARAMS]
    s_sequence = arma_generate_sample(ar=ar, ma=ma, nsample=length, scale=noise_std)
    return s_sequence

def generate_pretraining_data(config, fixed_noise=0.1):
    """
    Generates historical data for pre-training.
    Crucial: This should use the BASE AR parameters to represent historical distribution.
    """
    original_noise = config.STATE_NOISE_STD
    config.STATE_NOISE_STD = fixed_noise
    
    X_beta, y_s_prime = [], []
    for traj in range(config.N_TRAJECTORIES):
        s_sequence = generate_single_trajectory(config, config.T_TRAJECTORY, noise_std=fixed_noise)
        start_index = config.BETA_FEATURE_DIM
        for t in range(start_index, len(s_sequence)):
            s_history = list(s_sequence[:t]) # History up to t-1
            
            features = create_features_for_S_prime(s_history, config)
            s_prime = compute_S_prime(s_history, config) # W_t
            
            X_beta.append(features)
            y_s_prime.append(s_prime)
            
    config.STATE_NOISE_STD = original_noise
    return np.array(X_beta), np.array(y_s_prime)

def train_beta_estimator(X_beta, y_s_prime):
    """Trains the linear model to estimate W_t from x_t."""
    model = LinearRegression()
    if len(X_beta) > 0:
        model.fit(X_beta, y_s_prime)
    else:
        model.fit(np.zeros((1, X_beta.shape[1])), [0])
    residuals = y_s_prime - model.predict(X_beta) if len(X_beta) > 0 else [0]
    std = np.std(residuals) if len(residuals) > 0 else 1.0
    return {"model": model, "prediction_std": std}

# ==================================
# 4. Agents
# ==================================
class OFUL_FullLinUCBAgent:
    """Oracle Agent: Observes both S_t and W_t."""
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
    """Benchmark Agent: Observes only S_t."""
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
    """PULSE-UCB Agent: Infers W_t from history."""
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
    """CLBBF Agent: Handles missing variables via imputation."""
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
# 5. Unified Runner
# ==================================
def run_any_agent(config, agent_type, beta_estimator=None):
    """Executes one run of an agent on a generated trajectory."""
    s_sequence = generate_single_trajectory(config, config.N_TIME_STEPS)
    regret_log = np.zeros(config.N_TIME_STEPS)
    start_index = config.BETA_FEATURE_DIM
    
    # Initialize appropriate agent
    if agent_type == 'PULSE-UCB':
        agent = PartialObservationLinUCBAgent(config, beta_estimator)
    elif agent_type == 'OFUL':
        agent = OFULLinUCBAgent(config)
    elif agent_type == 'OFUL-Full':
        agent = OFUL_FullLinUCBAgent(config)
    elif agent_type == 'CLBBF':
        agent = CLBBFAgent(config)
    else:
        raise ValueError("Unknown Agent Type")
        
    for t in range(start_index, config.N_TIME_STEPS):
        # Construct history from t=0 to t-1
        s_history = list(s_sequence[:t]) 
        s_t = s_sequence[t]
        
        # True unobserved context W_t (for Oracle and regret calc)
        s_prime_t = compute_S_prime(s_history, config)
        
        # Action selection
        if agent_type == 'PULSE-UCB':
            action = agent.choose_action(s_history)
        elif agent_type == 'OFUL':
            action = agent.choose_action(s_t)
        elif agent_type == 'OFUL-Full' or agent_type == 'CLBBF':
            action = agent.choose_action(s_t, s_prime_t)
            
        # Reward observation
        x_true = create_OFUL_Full_features(s_t, s_prime_t, action, config)
        reward = np.dot(config.TRUE_THETA, x_true) + np.random.normal(0, config.REWARD_NOISE_STD)
        
        # Update agent state
        if agent_type == 'PULSE-UCB':
            agent.update(action, reward, s_history)
        elif agent_type == 'OFUL':
            agent.update(action, reward, s_t)
        elif agent_type == 'OFUL-Full' or agent_type == 'CLBBF':
            agent.update(action, reward, s_t, s_prime_t)
            
        # Regret Calculation
        opt = np.max([np.dot(config.TRUE_THETA, create_OFUL_Full_features(s_t, s_prime_t, a, config)) for a in range(config.N_ACTIONS)])
        regret_log[t] = opt - reward
        
    return np.sum(regret_log)

# ==================================
# 6. High-Dimensional Geometric Analysis
# ==================================

def get_covariance_matrix(ar_params, dim=20, n_samples=10000):
    """Helper: Generate covariance matrix of the feature vector."""
    np.random.seed(42)
    ar = np.r_[1, -np.array(ar_params)]
    ma = np.r_[1, [0.65, 0.35]]
    s = arma_generate_sample(ar=ar, ma=ma, nsample=n_samples + dim, scale=0.1)
    X = np.array([s[i:i+dim] for i in range(n_samples)])
    return np.cov(X.T)

def calculate_subspace_similarity(cov_base, cov_target, k=3):
    """
    Calculates the Subspace Similarity between the top-k Principal Components.
    
    Metric Logic:
    1. Perform PCA (eigen-decomposition) on both covariance matrices.
    2. Extract the subspace spanned by the top-k eigenvectors (U_base, U_target).
    3. Compute similarity: || U_base^T * U_target ||_F / sqrt(k).
    
    Interpretation:
    - 1.00: The principal subspace is identical (Invariant to scaling).
    - < 1.00: The principal subspace has rotated (Structural Shift).
    """
    # 1. Eigen decomposition
    _, vec_base = np.linalg.eigh(cov_base)
    _, vec_target = np.linalg.eigh(cov_target)
    
    # 2. Select top-k eigenvectors (last k columns from eigh)
    U_base = vec_base[:, -k:]
    U_target = vec_target[:, -k:]
    
    # 3. Compute subspace similarity (Grassmannian proximity)
    similarity = np.linalg.norm(np.dot(U_base.T, U_target), ord='fro') / np.sqrt(k)
    
    return similarity

def run_ar_structural_shift_analysis():
    """
    Runs the High-Dimensional Structural Shift Experiment.
    """
    config = LinearConfig()
    
    # Base: Positive correlation (Smooth trajectories)
    # This creates a "Low Frequency" feature manifold
    base_ar = np.array([0.75, -0.25]) 
    
    # Target: Negative correlation (Oscillating trajectories)
    # This creates a "High Frequency" feature manifold, geometrically orthogonal to Base.
    # Changing from Positive AR to Negative AR rotates eigenvectors significantly.
    target_ar = np.array([-0.75, -0.25]) 
    
    shift_intensities = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    
    # --- Pre-compute Base Geometry ---
    cov_base = get_covariance_matrix(base_ar, dim=config.BETA_FEATURE_DIM)
    
    # --- Train Beta Estimator ---
    print("\nTraining Beta Estimator on Historical Data (Base AR)...")
    config.AR_PARAMS = base_ar
    X_beta, y_s_prime = generate_pretraining_data(config, fixed_noise=0.1)
    beta_estimator = train_beta_estimator(X_beta, y_s_prime)
    
    results = {
        'PULSE-UCB': {'mean': [], 'se': [], 'color': 'blue', 'style': 'o-'},
        'OFUL': {'mean': [], 'se': [], 'color': 'red', 'style': 's--'},
        'CLBBF': {'mean': [], 'se': [], 'color': 'purple', 'style': '^--'},
        'OFUL-Full': {'mean': [], 'se': [], 'color': 'green', 'style': 'x--'}
    }
    
    # --- Table Setup ---
    table_data = []
    headers = ["Alpha", "AR Params", "PC Subspace Sim.", "PULSE-UCB", "OFUL", "CLBBF", "OFUL-Full"]
    table_data.append(headers)
    
    print(f"\n" + "="*100)
    print(f"Running High-Dimensional Structural Shift Analysis")
    print(f"Base AR (Smooth): {base_ar} -> Target AR (Oscillatory): {target_ar}")
    print("="*100)
    
    for alpha in shift_intensities:
        # Interpolate AR parameters
        # alpha controls the "mix" between smooth and oscillatory dynamics
        current_ar = (1 - alpha) * base_ar + alpha * target_ar
        config.AR_PARAMS = current_ar
        
        # 1. Calculate Geometric Metric
        cov_curr = get_covariance_matrix(current_ar, dim=config.BETA_FEATURE_DIM)
        # We calculate similarity of Top-5 Principal Components
        pc_sim = calculate_subspace_similarity(cov_base, cov_curr, k=5)
        
        ar_str = f"[{current_ar[0]:.2f}, {current_ar[1]:.2f}]"
        
        # 2. Run Experiment
        temp_results = {}
        for name in results.keys():
            regs = []
            for _ in range(config.N_EXPERIMENTS):
                config.AR_PARAMS = current_ar 
                r = run_any_agent(config, name, beta_estimator if name == 'PULSE-UCB' else None)
                regs.append(r)
            
            # Calculate Mean and Standard Error (SE)
            mean_val = np.mean(regs)
            se_val = np.std(regs) / np.sqrt(config.N_EXPERIMENTS)
            
            results[name]['mean'].append(mean_val)
            results[name]['se'].append(se_val)
            
            # Formatting: "123.45 +/- 6.78"
            temp_results[name] = f"{mean_val:.2f} +/- {se_val:.2f}"
        
        # 3. Logging
        print(f"Alpha={alpha:.2f} | PC Sim={pc_sim:.3f} | PULSE={temp_results['PULSE-UCB']}")
        
        row = [f"{alpha:.2f}", ar_str, f"{pc_sim:.3f}"]
        row.append(temp_results['PULSE-UCB'])
        row.append(temp_results['OFUL'])
        row.append(temp_results['CLBBF'])
        row.append(temp_results['OFUL-Full'])
        table_data.append(row)

    # --- Print Final Table ---
    print("\n" + "="*120)
    print("Table 4: High-Dimensional Structural Shift (Rotation of Feature Manifold)")
    print("Results shown as: Mean Cumulative Regret +/- Standard Error")
    print("="*120)
    col_widths = [max(len(str(item)) for item in col) for col in zip(*table_data)]
    header_str = " | ".join(f"{item:<{width}}" for item, width in zip(table_data[0], col_widths))
    print(header_str)
    print("-" * len(header_str))
    for row in table_data[1:]:
        print(" | ".join(f"{item:<{width}}" for item, width in zip(row, col_widths)))
    print("="*120 + "\n")

    # --- Plotting (Updated for SE) ---
    plt.figure(figsize=(8, 6))
    for name, data in results.items():
        means = np.array(data['mean'])
        ses = np.array(data['se'])
        # Plot with Standard Error shading
        plt.plot(shift_intensities, means, data['style'], color=data['color'], linewidth=2.5, label=name)
        plt.fill_between(shift_intensities, means-ses, means+ses, color=data['color'], alpha=0.2)
    
    plt.xlabel(r'Structural Shift Intensity $\alpha$', fontsize=14)
    plt.ylabel('Cumulative Regret', fontsize=14)
    plt.title('Robustness to High-Dim Manifold Rotation', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Structural_Shift_Analysis.png'), dpi=300)
    print("Plot saved as 'Structural_Shift_Analysis.png'")

if __name__ == "__main__":
    run_ar_structural_shift_analysis()
