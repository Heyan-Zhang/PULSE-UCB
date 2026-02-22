import numpy as np
from statsmodels.tsa.arima_process import arma_generate_sample
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
from scipy import stats
import warnings
import os
warnings.filterwarnings('ignore')

# Set up plotting style
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
# 1. Configurations for experiments
# ==================================
class ExperimentConfig:
    def __init__(self, experiment_type="linear"):
        # Pre-training data parameters
        self.N_TRAJECTORIES = 1000      # Number of independent trajectories for pre-training
        self.T_TRAJECTORY = 100         # Length of each trajectory for pre-training
        
        # Online learning parameters  
        self.N_TIME_STEPS = 1000        # Length of online learning trajectory
        self.N_EXPERIMENTS = 30         # Number of experiment runs for averaging
        self.N_ACTIONS = 2
        
        # Monte Carlo sampling parameters
        self.MC_SAMPLES = 20
        
        np.random.seed(42)
        
        # Set up different data generation mechanisms based on experiment type
        if experiment_type == "linear":
            self._setup_linear()
        elif experiment_type == "nonlinear_eta_0.1":
            self._setup_nonlinear(eta=0.1)
        elif experiment_type == "nonlinear_eta_1":
            self._setup_nonlinear(eta=1.0)
        elif experiment_type == "nonlinear_eta_10":
            self._setup_nonlinear(eta=10.0)
    
    def _setup_linear(self):
        """Linear baseline setup"""
        self.AR_PARAMS = np.array([0.75, -0.25])
        self.MA_PARAMS = np.array([0.65, 0.35])
        self.STATE_NOISE_STD = 0.1
        
        self.BETA_FEATURE_DIM = 2
        self.TRUE_BETA = np.random.randn(self.BETA_FEATURE_DIM)
        self.S_PRIME_NOISE_STD = 0.1
        
        self.THETA_FEATURE_DIM = 4
        self.TRUE_THETA = np.random.randn(self.THETA_FEATURE_DIM)
        self.REWARD_NOISE_STD = 0.05
        
        self.AGENT_FEATURE_DIM = 4
        self.AGENT_THETA_FEATURE_DIM = 4
        self.relation_type = "linear"
        self.estimator_type = "linear_regression"
        self.eta = 0  # No nonlinearity
    
    def _setup_nonlinear(self, eta):
        """Setup nonlinear S'_t with different eta values"""
        self._setup_linear()
        self.relation_type = "nonlinear_sprime"
        self.estimator_type = "polynomial"
        self.polynomial_degree = 2
        self.eta = eta
        self.nonlinear_coefficient = 2.5

# ==================================
# 2. Feature creation functions
# ==================================
def create_features_for_S_prime(s_history, config):
    """S'_t feature creation"""
    if len(s_history) < 3:
        return np.zeros(config.BETA_FEATURE_DIM)
    
    features = np.zeros(config.BETA_FEATURE_DIM)
    features[0] = 1.0
    features[1] = np.mean(s_history[-3:])
    return features

def compute_S_prime(s_history, config):
    """Compute S'_t with potential nonlinearity controlled by eta"""
    features = create_features_for_S_prime(s_history, config)
    
    # Linear part
    linear_part = np.dot(config.TRUE_BETA, features)
    
    if config.relation_type == "nonlinear_sprime":
        # Add nonlinear term: sin(eta * x)
        if len(features) >= 2:
            nonlinear_part = np.sin(config.eta * features[1])
            return linear_part + nonlinear_part + np.random.normal(0, config.S_PRIME_NOISE_STD)
    
    return linear_part + np.random.normal(0, config.S_PRIME_NOISE_STD)

def compute_S_prime_expected(s_history, config):
    """Compute expected S'_t (without noise) - used for optimal reward calculation"""
    features = create_features_for_S_prime(s_history, config)
    
    # Linear part
    linear_part = np.dot(config.TRUE_BETA, features)
    
    if config.relation_type == "nonlinear_sprime":
        # Add nonlinear term: sin(eta * x)
        if len(features) >= 2:
            nonlinear_part = config.nonlinear_coefficient * np.sin(config.eta * features[1])
            return linear_part + nonlinear_part
    
    return linear_part

def create_reward_features(s_t, s_prime_t, action, config):
    """Create reward features"""
    features = np.zeros(config.THETA_FEATURE_DIM)
    features[0] = 1.0
    features[1] = s_t
    features[2] = s_prime_t
    features[3] = s_t * action
    return features

def compute_reward(s_t, s_prime, action, config, current_theta=None):
    """Compute reward"""
    if current_theta is None:
        current_theta = config.TRUE_THETA
    
    reward_features = create_reward_features(s_t, s_prime, action, config)
    return np.dot(current_theta, reward_features)

# ==================================
# 3. Model training for beta estimation
# ==================================
class EstimatorFactory:
    @staticmethod
    def create_estimator(config):
        if config.estimator_type == "linear_regression":
            return LinearRegression()
        elif config.estimator_type == "polynomial":
            return LinearRegression()  # Use LinearRegression for polynomial regression
        elif config.estimator_type == "random_forest":
            return RandomForestRegressor(n_estimators=50, random_state=42)
        else:
            return LinearRegression()

def train_beta_estimator(X_beta, y_s_prime, config):
    """Train estimator and estimate prediction uncertainty"""
    estimator = EstimatorFactory.create_estimator(config)
    
    if config.estimator_type == "polynomial":
        # Polynomial feature expansion
        poly = PolynomialFeatures(degree=config.polynomial_degree, include_bias=False)
        X_poly = poly.fit_transform(X_beta)
        linear_model = LinearRegression()
        linear_model.fit(X_poly, y_s_prime)
        
        # Estimate prediction uncertainty using residuals
        predictions = linear_model.predict(X_poly)
        residuals = y_s_prime - predictions
        prediction_std = np.std(residuals)
        
        return {
            "model": linear_model, 
            "poly_features": poly, 
            "type": "polynomial",
            "prediction_std": prediction_std
        }
    
    else:
        estimator.fit(X_beta, y_s_prime)
        
        # Estimate prediction uncertainty
        predictions = estimator.predict(X_beta)
        residuals = y_s_prime - predictions
        prediction_std = np.std(residuals)
        
        return {
            "model": estimator, 
            "type": config.estimator_type,
            "prediction_std": prediction_std
        }

# ==================================
# 4. Enhanced LinUCB Agent with Monte Carlo Sampling
# ==================================
class EnhancedLinUCBAgent:
    def __init__(self, config, beta_estimator, alpha=1.0):
        self.config = config
        self.beta_estimator = beta_estimator
        self.alpha = alpha
        
        self.A = np.identity(config.AGENT_THETA_FEATURE_DIM)
        self.b = np.zeros((config.AGENT_THETA_FEATURE_DIM, 1))
    
        self.action_history = []
        self.reward_history = []
        
    def sample_S_prime_with_uncertainty(self, s_history, n_samples=None):
        """Sample S'_t multiple times considering prediction uncertainty"""
        if n_samples is None:
            n_samples = self.config.MC_SAMPLES
            
        features = create_features_for_S_prime(s_history, self.config)
        samples = []
        
        for _ in range(n_samples):
            if self.beta_estimator["type"] == "polynomial":
                features_poly = self.beta_estimator["poly_features"].transform(features.reshape(1, -1))
                mean_pred = self.beta_estimator["model"].predict(features_poly)[0]
            else:
                mean_pred = self.beta_estimator["model"].predict(features.reshape(1, -1))[0]
            
            # Add uncertainty by sampling from normal distribution
            uncertainty_std = self.beta_estimator.get("prediction_std", self.config.S_PRIME_NOISE_STD)
            sampled_s_prime = np.random.normal(mean_pred, uncertainty_std)
            samples.append(sampled_s_prime)
        
        return np.array(samples)
    
    def compute_expected_reward_features_mc(self, s_history, action):
        """Compute expected reward features using Monte Carlo sampling"""
        s_t = s_history[-1]
        
        # Sample multiple S'_t values
        s_prime_samples = self.sample_S_prime_with_uncertainty(s_history)
        
        # Compute reward features for each sample
        feature_samples = []
        for s_prime_sample in s_prime_samples:
            features = create_reward_features(s_t, s_prime_sample, action, self.config)
            feature_samples.append(features)
        
        # Return average of feature vectors
        return np.mean(feature_samples, axis=0)
    
    def choose_action(self, s_history):
        """Choose action using Monte Carlo averaged reward features"""
        # Add numerical stability check
        if np.linalg.det(self.A) < 1e-10:
            self.A += 1e-6 * np.identity(self.A.shape[0])
            
        A_inv = np.linalg.inv(self.A)
        theta_hat = A_inv @ self.b
        
        ucb_scores = np.zeros(self.config.N_ACTIONS)
        for a in range(self.config.N_ACTIONS):
            # Use Monte Carlo sampling to get expected reward features
            x_ta = self.compute_expected_reward_features_mc(s_history, a).reshape(-1, 1)
            
            mean_reward = (theta_hat.T @ x_ta).item()
            confidence_width = self.alpha * np.sqrt(x_ta.T @ A_inv @ x_ta).item()
            ucb_scores[a] = mean_reward + confidence_width
        
        chosen_action = np.argmax(ucb_scores)
        self.action_history.append(chosen_action)
        return chosen_action
    
    def update(self, action, reward, s_history):
        """Update parameters using Monte Carlo averaged features"""
        self.reward_history.append(reward)
        
        # Use Monte Carlo sampling for feature computation in update as well
        x_chosen = self.compute_expected_reward_features_mc(s_history, action).reshape(-1, 1)
        
        self.A += x_chosen @ x_chosen.T
        self.b += reward * x_chosen

# ==================================
# 5. Environment data generation
# ==================================
def generate_single_trajectory(config, length):
    """Generate a single trajectory of given length"""
    ar = np.r_[1, -config.AR_PARAMS]
    ma = np.r_[1, config.MA_PARAMS]
    s_sequence = arma_generate_sample(
        ar=ar, ma=ma, 
        nsample=length, 
        scale=config.STATE_NOISE_STD
    )
    return s_sequence

def generate_pretraining_data(config):
    """Generate multiple independent trajectories for pre-training"""
    X_beta, y_s_prime = [], []
    
    for traj in range(config.N_TRAJECTORIES):
        # Generate one trajectory
        s_sequence = generate_single_trajectory(config, config.T_TRAJECTORY)
        
        # Extract features and targets from this trajectory
        for t in range(3, len(s_sequence)):
            s_history = list(s_sequence[:t+1])
            features = create_features_for_S_prime(s_history, config)
            s_prime = compute_S_prime(s_history, config)
            
            X_beta.append(features)
            y_s_prime.append(s_prime)
    
    return np.array(X_beta), np.array(y_s_prime)

def get_true_reward(s_history, action, config):
    """Calculate true reward"""
    s_prime = compute_S_prime(s_history, config)
    s_t = s_history[-1]
    
    expected_reward = compute_reward(s_t, s_prime, action, config)
    return expected_reward + np.random.normal(0, config.REWARD_NOISE_STD)

def get_optimal_reward(s_history, config):
    """Calculate optimal reward - include nonlinear terms"""
    potential_rewards = []
    for a in range(config.N_ACTIONS):
        # Use the corrected function that includes nonlinear terms
        expected_s_prime = compute_S_prime_expected(s_history, config)  # No noise, but includes nonlinearity
        
        s_t = s_history[-1]
        expected_reward = compute_reward(s_t, expected_s_prime, a, config)
        potential_rewards.append(expected_reward)
    
    return np.max(potential_rewards)

# ==================================
# 6. Experiment Runner
# ==================================
def run_experiment(config, alpha=1.0):
    """Run a single experiment"""
    # Generate pre-training data from multiple independent trajectories
    X_beta, y_s_prime = generate_pretraining_data(config)
    beta_estimator = train_beta_estimator(X_beta, y_s_prime, config)
    
    # Create agent
    agent = EnhancedLinUCBAgent(config, beta_estimator, alpha)
    
    # Generate online learning trajectory (single long trajectory)
    s_bandit = generate_single_trajectory(config, config.N_TIME_STEPS)
    
    rewards_log = np.zeros(config.N_TIME_STEPS)
    optimal_rewards_log = np.zeros(config.N_TIME_STEPS)
    regret_log = np.zeros(config.N_TIME_STEPS)
    
    for t in range(3, config.N_TIME_STEPS):
        current_s_history = list(s_bandit[:t+1])
        
        chosen_action = agent.choose_action(current_s_history)
        observed_reward = get_true_reward(current_s_history, chosen_action, config)
        optimal_reward = get_optimal_reward(current_s_history, config)
        
        agent.update(chosen_action, observed_reward, current_s_history)
        
        rewards_log[t] = observed_reward
        optimal_rewards_log[t] = optimal_reward
        regret_log[t] = optimal_reward - observed_reward
    
    return {
        'rewards': rewards_log,
        'optimal_rewards': optimal_rewards_log,
        'regret': regret_log,
        'cumulative_regret': np.cumsum(regret_log),
        'action_history': agent.action_history,
        'reward_history': agent.reward_history
    }

# ==================================
# 7. Visualization Functions
# ==================================
def create_performance_analysis():
    """Create performance analysis visualization"""
    
    # Run different types of experiments
    experiment_types = [
        "linear",
        "nonlinear_eta_0.1", 
        "nonlinear_eta_1",
        "nonlinear_eta_10"
    ]
    
    print("Starting experiments and generating visualizations...")
    
    # Create figure
    # fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    results = {}
    
    # Define color mapping and labels
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    exp_labels = ['Linear', 'Nonlinear (ρ=0.1)', 'Nonlinear (ρ=1)', 'Nonlinear (ρ=10)']
    
    # Run experiments
    for i, exp_type in enumerate(experiment_types):
        print(f"Running experiment: {exp_type}")
        config = ExperimentConfig(exp_type)
        
        # Run multiple experiments and take average
        all_cumulative_regrets = []
        all_instant_regrets = []
        all_rewards = []
        
        for exp_run in range(config.N_EXPERIMENTS):
            result = run_experiment(config, alpha=1.0)
            all_cumulative_regrets.append(result['cumulative_regret'])
            all_instant_regrets.append(result['regret'])
            all_rewards.append(result['rewards'])
        
        # Calculate mean and standard deviation
        mean_cumulative_regret = np.mean(all_cumulative_regrets, axis=0)
        std_cumulative_regret = np.std(all_cumulative_regrets, axis=0)/np.sqrt(config.N_EXPERIMENTS)
        mean_instant_regret = np.mean(all_instant_regrets, axis=0)
        mean_rewards = np.mean(all_rewards, axis=0)
        
        results[exp_type] = {
            'mean_cumulative_regret': mean_cumulative_regret,
            'std_cumulative_regret': std_cumulative_regret,
            'mean_instant_regret': mean_instant_regret,
            'mean_rewards': mean_rewards,
            'color': colors[i],
            'label': exp_labels[i]
        }
    
    # 1. Cumulative Regret Comparison (Top Left)
    ax1 = axes[0]
    for exp_type, data in results.items():
        time_steps = range(len(data['mean_cumulative_regret']))
        ax1.plot(time_steps, data['mean_cumulative_regret'], 
                color=data['color'], linewidth=3, label=data['label'], alpha=0.8)
        # Add confidence intervals
        ax1.fill_between(time_steps, 
                        data['mean_cumulative_regret'] - data['std_cumulative_regret'],
                        data['mean_cumulative_regret'] + data['std_cumulative_regret'],
                        color=data['color'], alpha=0.2)
    
    ax1.set_title('Cumulative Regret Comparison', fontsize=18, fontweight='bold')
    ax1.set_xlabel('Time Steps', fontsize=18)
    ax1.set_ylabel('Cumulative Regret', fontsize=18)
    ax1.legend(loc='upper left', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=18)
    # 2. Smoothed Instant Regret (Top Right)
    ax2 = axes[1]
    window_size = 50
    for exp_type, data in results.items():
        # Smooth instant regret
        if len(data['mean_instant_regret']) > window_size:
            smoothed_regret = np.convolve(data['mean_instant_regret'], 
                                        np.ones(window_size)/window_size, mode='valid')
            time_steps_smooth = range(window_size-1, len(data['mean_instant_regret']))
            ax2.plot(time_steps_smooth, smoothed_regret, 
                    color=data['color'], linewidth=2.5, label=data['label'], alpha=0.8)
    
    ax2.set_title('Smoothed Instant Regret', fontsize=18, fontweight='bold')
    ax2.set_xlabel('Time Steps', fontsize=18)
    ax2.set_ylabel('Instant Regret (50-step avg)', fontsize=18)
    ax2.legend(loc='upper right', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=18)
    # # 3. Final Performance Bar Chart (Bottom Left)
    # ax3 = axes[1, 0]
    # final_regrets = [data['mean_cumulative_regret'][-1] for data in results.values()]
    # exp_names = [data['label'] for data in results.values()]
    # bar_colors = [data['color'] for data in results.values()]
    
    # bars = ax3.bar(range(len(exp_names)), final_regrets, color=bar_colors, alpha=0.8, 
    #                edgecolor='black', linewidth=1.5)
    # ax3.set_title('Final Cumulative Regret', fontsize=14, fontweight='bold')
    # ax3.set_ylabel('Final Cumulative Regret', fontsize=12)
    # ax3.set_xticks(range(len(exp_names)))
    # ax3.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=10)
    
    # # Add value labels on bars
    # for bar, value in zip(bars, final_regrets):
    #     height = bar.get_height()
    #     ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
    #             f'{value:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # # 4. Cumulative Reward Comparison (Bottom Right)
    # ax4 = axes[1, 1]
    # for exp_type, data in results.items():
    #     cum_reward = np.cumsum(data['mean_rewards'])
    #     ax4.plot(range(len(cum_reward)), cum_reward, 
    #             color=data['color'], linewidth=2.5, label=data['label'], alpha=0.8)
    
    # ax4.set_title('Cumulative Reward Comparison', fontsize=14, fontweight='bold')
    # ax4.set_xlabel('Time Steps', fontsize=12)
    # ax4.set_ylabel('Cumulative Reward', fontsize=12)
    # ax4.legend(loc='lower right', fontsize=10)
    # ax4.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save the figure
    plt.savefig(os.path.join(OUTPUT_DIR, 'eta_performance_analysis.png'), dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    # Print numerical results summary
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    
    for exp_type, data in results.items():
        final_regret = data['mean_cumulative_regret'][-1]
        final_reward = np.sum(data['mean_rewards'])
        print(f"\n{data['label']}:")
        print(f"  Final Cumulative Regret: {final_regret:.2f}")
        print(f"  Total Cumulative Reward: {final_reward:.2f}")
        print(f"  Average Instant Reward: {np.mean(data['mean_rewards'][data['mean_rewards'] != 0]):.3f}")
    
    return results

# ==================================
# 8. Quick Demo Function
# ==================================
def quick_demo():
    """Quick demonstration"""
    print("Running LinUCB Quick Demo...")
    
    # Create a simple linear experiment
    config = ExperimentConfig("linear")
    print("\n=== PARAMETERS USED IN THIS RUN ===")
    print("TRUE_BETA:", config.TRUE_BETA)
    print("TRUE_THETA:", config.TRUE_THETA)
    print("==================================\n")
    result = run_experiment(config, alpha=1.0)
    
    # Create simplified visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('LinUCB Algorithm Quick Demo - Linear Setting', 
                 fontsize=16, fontweight='bold')
    
    # 1. Cumulative Regret
    axes[0].plot(result['cumulative_regret'], color='red', linewidth=3, 
                 label='Cumulative Regret', alpha=0.8)
    axes[0].set_title('Cumulative Regret Over Time', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Time Steps', fontsize=12)
    axes[0].set_ylabel('Cumulative Regret', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=11)
    
    # 2. Cumulative Reward Comparison
    cumulative_rewards = np.cumsum(result['rewards'])
    cumulative_optimal = np.cumsum(result['optimal_rewards'])
    axes[1].plot(cumulative_rewards, color='blue', linewidth=3, 
                 label='Actual Cumulative Reward', alpha=0.8)
    axes[1].plot(cumulative_optimal, color='green', linewidth=3, linestyle='--', 
                 label='Optimal Cumulative Reward', alpha=0.8)
    axes[1].set_title('Cumulative Reward Comparison', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Time Steps', fontsize=12)
    axes[1].set_ylabel('Cumulative Reward', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'linucb_quick_demo.png'), dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    # Print key metrics
    print(f"\nKEY PERFORMANCE METRICS:")
    print(f"Final Cumulative Regret: {result['cumulative_regret'][-1]:.2f}")
    print(f"Total Reward: {sum(result['rewards']):.2f}")
    print(f"Average Reward: {np.mean(result['rewards'][result['rewards'] != 0]):.3f}")

# ==================================
# 9. Main Function
# ==================================
def main():
    """Main execution function"""
    print("LinUCB Algorithm Performance Analysis System")
    print("="*60)
    print("Choose running mode:")
    print("1. Quick Demo (Linear setting only)")
    print("2. Full Analysis (All four settings comparison)")
    
    try:
        choice = input("Please enter your choice (1 or 2, default is 2): ").strip()
        if choice == "1":
            print("\nRunning quick demo...")
            quick_demo()
            return None
        else:
            print("\nRunning full performance analysis...")
            results = create_performance_analysis()
            return results
    except KeyboardInterrupt:
        print("\nUser interrupted, running quick demo instead...")
        quick_demo()
        return None
    except Exception as e:
        print(f"\nError occurred: {e}")
        print("Running quick demo as fallback...")
        quick_demo()
        return None

if __name__ == "__main__":
    main()
