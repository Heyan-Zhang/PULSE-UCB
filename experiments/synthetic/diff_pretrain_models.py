import numpy as np
from statsmodels.tsa.arima_process import arma_generate_sample
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import seaborn as sns
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
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs", "synthetic")
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_style("whitegrid")

# ==================================
# 1. Enhanced Configuration with Model Mismatch
# ==================================
class EnhancedLinearConfig:
    def __init__(self, complexity_level="high"):
        # Pre-training data parameters
        self.N_TRAJECTORIES = 1000      
        self.T_TRAJECTORY = 100         
        
        # Online learning parameters  
        self.N_TIME_STEPS = 1000        
        self.N_EXPERIMENTS = 8          
        self.N_ACTIONS = 2
        
        # Monte Carlo sampling parameters
        self.MC_SAMPLES = 20
        
        # Environment parameters (linear setting)
        self.AR_PARAMS = np.array([0.75, -0.25])
        self.MA_PARAMS = np.array([0.65, 0.35])
        self.STATE_NOISE_STD = 0.1
        
        # Enhanced S'_t generation parameters (CREATE MODEL MISMATCH)
        self.BETA_FEATURE_DIM = 4  # Increased feature dimension
        self.complexity_level = complexity_level
        
        # Different complexity levels for model mismatch
        if complexity_level == "low":
            # Simple linear (should favor linear models)
            self.TRUE_BETA = np.array([1.0, 0.5, 0.3, 0.0])
            self.use_nonlinear = False
            self.use_interactions = False
            self.use_conditionals = False
        elif complexity_level == "medium":
            # Moderate nonlinearity (should favor polynomial/neural networks)
            self.TRUE_BETA = np.array([1.0, 0.8, -0.4, 0.6])
            self.use_nonlinear = True
            self.use_interactions = True
            self.use_conditionals = False
        else:  # "high"
            # High complexity with conditionals (should favor tree-based models)
            self.TRUE_BETA = np.array([1.2, -0.6, 0.9, -0.3])
            self.use_nonlinear = True
            self.use_interactions = True
            self.use_conditionals = True
        
        self.S_PRIME_NOISE_STD = 0.15  # Increased noise for more challenging scenario
        
        # Reward parameters
        self.THETA_FEATURE_DIM = 4  
        self.TRUE_THETA = np.random.randn(self.THETA_FEATURE_DIM)
        self.REWARD_NOISE_STD = 0.05
        
        np.random.seed(42)
        print(f"Configuration: Complexity Level = {complexity_level}")
        print(f"Nonlinear: {self.use_nonlinear}, Interactions: {self.use_interactions}, Conditionals: {self.use_conditionals}")

# ==================================
# 2. Enhanced Feature Creation with Model Mismatch
# ==================================
def create_enhanced_features_for_S_prime(s_history, config):
    """Enhanced S'_t feature creation with potential model mismatch"""
    if len(s_history) < 5:
        return np.zeros(config.BETA_FEATURE_DIM)
    
    features = np.zeros(config.BETA_FEATURE_DIM)
    
    # Basic features
    features[0] = 1.0  # Intercept
    features[1] = np.mean(s_history[-3:])  # Recent mean
    features[2] = np.std(s_history[-5:]) if len(s_history) >= 5 else 0  # Recent volatility
    features[3] = s_history[-1] - s_history[-2] if len(s_history) >= 2 else 0  # Recent change
    
    return features

def compute_enhanced_S_prime(s_history, config):
    """
    Compute S'_t with complex nonlinear relationships to create model mismatch
    """
    if len(s_history) < 5:
        return np.random.normal(0, config.S_PRIME_NOISE_STD)
    
    features = create_enhanced_features_for_S_prime(s_history, config)
    
    # Start with linear base
    s_prime = np.dot(config.TRUE_BETA, features)
    
    # Add nonlinear transformations based on complexity level
    if config.use_nonlinear:
        # Add polynomial terms
        s_prime += 0.3 * features[1]**2  # Quadratic term
        s_prime += 0.2 * np.sin(2 * features[1])  # Trigonometric term
        s_prime -= 0.15 * np.exp(-features[2]**2)  # Gaussian-like term
    
    if config.use_interactions:
        # Add interaction terms
        s_prime += 0.4 * features[1] * features[2]  # Mean × Volatility interaction
        s_prime += 0.25 * features[1] * features[3]  # Mean × Change interaction
        s_prime -= 0.2 * features[2] * np.abs(features[3])  # Volatility × |Change|
    
    if config.use_conditionals:
        # Add conditional logic (tree-like structure)
        recent_mean = features[1]
        recent_vol = features[2]
        recent_change = features[3]
        
        if recent_mean > 0.5:
            if recent_vol > 0.3:
                s_prime += 0.8 * np.tanh(recent_change)  # High mean, high vol
            else:
                s_prime += 0.4 * recent_change**2  # High mean, low vol
        else:
            if recent_vol > 0.3:
                s_prime -= 0.6 * np.abs(recent_change)  # Low mean, high vol
            else:
                s_prime += 0.3 * np.log(1 + np.abs(recent_change))  # Low mean, low vol
        
        # Additional threshold-based logic
        if np.abs(recent_change) > 0.5:
            s_prime *= 1.3  # Amplify during large changes
        
        if recent_vol < 0.1:
            s_prime += 0.2 * np.random.uniform(-1, 1)  # Add randomness in stable periods
    
    # Add noise
    s_prime += np.random.normal(0, config.S_PRIME_NOISE_STD)
    
    return s_prime

def create_oracle_features(s_t, s_prime_t, action, config):
    """Oracle features: Y_t = (S_t, S'_t)"""
    features = np.zeros(config.THETA_FEATURE_DIM)
    features[0] = 1.0
    features[1] = s_t
    features[2] = s_prime_t
    features[3] = s_t * action
    return features

# ==================================
# 3. Enhanced Pre-trained Model Factory
# ==================================
class EnhancedPretrainedModelFactory:
    """Enhanced factory for creating different types of pre-trained models"""
    
    @staticmethod
    def create_simple_linear_model():
        """Simple linear regression (should struggle with complex data)"""
        return {
            'name': 'Linear Regression',
            'model': LinearRegression(),
            'type': 'sklearn',
            'color': '#1f77b4'
        }
    
    @staticmethod
    def create_polynomial_model(degree=3):
        """Polynomial regression (should handle moderate nonlinearity)"""
        return {
            'name': f'Polynomial (deg={degree})',
            'model': Pipeline([
                ('poly', PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)),
                ('scaler', StandardScaler()),
                ('linear', LinearRegression())
            ]),
            'type': 'sklearn',
            'color': '#ff7f0e'
        }
    
    @staticmethod
    def create_ridge_model(alpha=10.0):
        """Ridge regression with higher regularization"""
        return {
            'name': f'Ridge (α={alpha})',
            'model': Pipeline([
                ('scaler', StandardScaler()),
                ('ridge', Ridge(alpha=alpha))
            ]),
            'type': 'sklearn',
            'color': '#2ca02c'
        }
    
    @staticmethod
    def create_lasso_model(alpha=0.1):
        """Lasso regression for feature selection"""
        return {
            'name': f'Lasso (α={alpha})',
            'model': Pipeline([
                ('scaler', StandardScaler()),
                ('lasso', Lasso(alpha=alpha, max_iter=2000))
            ]),
            'type': 'sklearn',
            'color': '#d62728'
        }
    
    @staticmethod
    def create_random_forest_model(n_estimators=100, max_depth=15):
        """Random Forest (should excel with conditionals)"""
        return {
            'name': f'Random Forest (n={n_estimators})',
            'model': RandomForestRegressor(
                n_estimators=n_estimators, 
                max_depth=max_depth,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'type': 'sklearn',
            'color': '#9467bd'
        }
    
    @staticmethod
    def create_gradient_boosting_model(n_estimators=150, learning_rate=0.1):
        """Gradient Boosting (should handle complex patterns)"""
        return {
            'name': f'Gradient Boosting (lr={learning_rate})',
            'model': GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=6,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=42
            ),
            'type': 'sklearn',
            'color': '#8c564b'
        }
    
    @staticmethod
    def create_neural_network_model(hidden_layers=(100, 50, 25), learning_rate=0.001):
        """Deep Neural Network (should handle nonlinearities and interactions)"""
        return {
            'name': f'Neural Network {hidden_layers}',
            'model': Pipeline([
                ('scaler', StandardScaler()),
                ('mlp', MLPRegressor(
                    hidden_layer_sizes=hidden_layers,
                    learning_rate_init=learning_rate,
                    activation='relu',
                    solver='adam',
                    alpha=0.001,
                    random_state=42,
                    max_iter=2000,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=20
                ))
            ]),
            'type': 'sklearn',
            'color': '#e377c2'
        }

def train_pretrained_model(X_beta, y_s_prime, model_config):
    """Train a pre-trained model and evaluate its performance"""
    model = model_config['model']
    
    # Train the model
    model.fit(X_beta, y_s_prime)
    
    # Evaluate prediction accuracy
    train_predictions = model.predict(X_beta)
    train_mse = np.mean((y_s_prime - train_predictions) ** 2)
    train_r2 = 1 - train_mse / np.var(y_s_prime)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_beta, y_s_prime, cv=5, scoring='r2')
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    
    # Estimate prediction uncertainty using residuals
    residuals = y_s_prime - train_predictions
    prediction_std = np.std(residuals)
    
    return {
        'model': model,
        'name': model_config['name'],
        'type': model_config['type'],
        'color': model_config['color'],
        'train_mse': train_mse,
        'train_r2': train_r2,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'prediction_std': prediction_std
    }

# ==================================
# 4. LinUCB Agent with Configurable Pre-trained Model
# ==================================
class ConfigurableLinUCBAgent:
    """LinUCB Agent with configurable pre-trained model"""
    
    def __init__(self, config, pretrained_estimator, alpha=1.0):
        self.config = config
        self.pretrained_estimator = pretrained_estimator
        self.alpha = alpha
        
        self.A = np.identity(config.THETA_FEATURE_DIM)
        self.b = np.zeros((config.THETA_FEATURE_DIM, 1))
        
        self.action_history = []
        self.reward_history = []
        
    def predict_s_prime_with_uncertainty(self, s_history, n_samples=None):
        """Predict S'_t with uncertainty using the pre-trained model"""
        if n_samples is None:
            n_samples = self.config.MC_SAMPLES
            
        features = create_enhanced_features_for_S_prime(s_history, self.config)
        samples = []
        
        for _ in range(n_samples):
            # Get prediction from the model
            mean_pred = self.pretrained_estimator['model'].predict(features.reshape(1, -1))[0]
            
            # Add uncertainty by sampling from normal distribution
            uncertainty_std = self.pretrained_estimator['prediction_std']
            sampled_s_prime = np.random.normal(mean_pred, uncertainty_std)
            samples.append(sampled_s_prime)
        
        return np.array(samples)
    
    def compute_expected_features_mc(self, s_history, action):
        """Compute expected features using Monte Carlo"""
        s_t = s_history[-1]
        s_prime_samples = self.predict_s_prime_with_uncertainty(s_history)
        
        feature_samples = []
        for s_prime_sample in s_prime_samples:
            features = create_oracle_features(s_t, s_prime_sample, action, self.config)
            feature_samples.append(features)
        
        return np.mean(feature_samples, axis=0)
    
    def choose_action(self, s_history):
        """Choose action using predicted S'_t"""
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
        """Update using predicted features"""
        self.reward_history.append(reward)
        x_chosen = self.compute_expected_features_mc(s_history, action).reshape(-1, 1)
        
        self.A += x_chosen @ x_chosen.T
        self.b += reward * x_chosen

# ==================================
# 5. Enhanced Data Generation Functions
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
        s_sequence = generate_single_trajectory(config, config.T_TRAJECTORY)
        
        for t in range(5, len(s_sequence)):  # Need at least 5 points for enhanced features
            s_history = list(s_sequence[:t+1])
            features = create_enhanced_features_for_S_prime(s_history, config)
            s_prime = compute_enhanced_S_prime(s_history, config)
            
            X_beta.append(features)
            y_s_prime.append(s_prime)
    
    return np.array(X_beta), np.array(y_s_prime)

def compute_true_reward(s_t, s_prime_t, action, config):
    """Compute true reward using oracle features"""
    oracle_features = create_oracle_features(s_t, s_prime_t, action, config)
    return np.dot(config.TRUE_THETA, oracle_features) + np.random.normal(0, config.REWARD_NOISE_STD)

def get_optimal_reward(s_t, s_prime_t, config):
    """Calculate optimal reward"""
    potential_rewards = []
    for a in range(config.N_ACTIONS):
        oracle_features = create_oracle_features(s_t, s_prime_t, a, config)
        expected_reward = np.dot(config.TRUE_THETA, oracle_features)
        potential_rewards.append(expected_reward)
    
    return np.max(potential_rewards)

# ==================================
# 6. Enhanced Experiment Runner
# ==================================
def run_experiment_with_model(config, pretrained_estimator, alpha=1.0):
    """Run experiment with specific pre-trained model"""
    agent = ConfigurableLinUCBAgent(config, pretrained_estimator, alpha)
    s_sequence = generate_single_trajectory(config, config.N_TIME_STEPS)
    
    rewards_log = np.zeros(config.N_TIME_STEPS)
    optimal_rewards_log = np.zeros(config.N_TIME_STEPS)
    regret_log = np.zeros(config.N_TIME_STEPS)
    
    for t in range(5, config.N_TIME_STEPS):  # Start from 5 for enhanced features
        s_history = list(s_sequence[:t+1])
        s_t = s_history[-1]
        s_prime_t = compute_enhanced_S_prime(s_history, config)
        
        chosen_action = agent.choose_action(s_history)
        observed_reward = compute_true_reward(s_t, s_prime_t, chosen_action, config)
        optimal_reward = get_optimal_reward(s_t, s_prime_t, config)
        
        agent.update(chosen_action, observed_reward, s_history)
        
        rewards_log[t] = observed_reward
        optimal_rewards_log[t] = optimal_reward
        regret_log[t] = optimal_reward - observed_reward
    
    return {
        'rewards': rewards_log,
        'optimal_rewards': optimal_rewards_log,
        'regret': regret_log,
        'cumulative_regret': np.cumsum(regret_log),
        'action_history': agent.action_history
    }

# ==================================
# 7. Enhanced Model Comparison Analysis
# ==================================
def run_enhanced_models_comparison(complexity_level="high"):
    """Run enhanced comparison analysis with model mismatch scenarios"""
    
    print(f"Running Enhanced Pre-trained Models Comparison (Complexity: {complexity_level})...")
    print("Testing different models for S'_t prediction in LinUCB with MODEL MISMATCH")
    
    config = EnhancedLinearConfig(complexity_level=complexity_level)
    
    # Generate pre-training data once (shared across all models)
    print("\nGenerating enhanced pre-training data...")
    X_beta, y_s_prime = generate_pretraining_data(config)
    print(f"Generated {len(X_beta)} training samples")
    print(f"Data complexity indicators:")
    print(f"  Target variance: {np.var(y_s_prime):.4f}")
    print(f"  Target mean: {np.mean(y_s_prime):.4f}")
    print(f"  Target range: [{np.min(y_s_prime):.3f}, {np.max(y_s_prime):.3f}]")
    
    # Define models to compare (enhanced set)
    model_factories = [
        EnhancedPretrainedModelFactory.create_simple_linear_model(),
        EnhancedPretrainedModelFactory.create_polynomial_model(degree=3),
        EnhancedPretrainedModelFactory.create_ridge_model(alpha=10.0),
        EnhancedPretrainedModelFactory.create_lasso_model(alpha=0.1),
        EnhancedPretrainedModelFactory.create_random_forest_model(n_estimators=100),
        EnhancedPretrainedModelFactory.create_gradient_boosting_model(n_estimators=150),
        EnhancedPretrainedModelFactory.create_neural_network_model(hidden_layers=(100, 50, 25))
    ]
    
    # Train all models and evaluate their prediction performance
    print("\nTraining and evaluating pre-trained models...")
    pretrained_models = {}
    model_performances = {}
    
    for model_config in model_factories:
        print(f"  Training {model_config['name']}...")
        trained_model = train_pretrained_model(X_beta, y_s_prime, model_config)
        pretrained_models[model_config['name']] = trained_model
        
        model_performances[model_config['name']] = {
            'train_r2': trained_model['train_r2'],
            'cv_mean': trained_model['cv_mean'],
            'cv_std': trained_model['cv_std'],
            'prediction_std': trained_model['prediction_std']
        }
        
        print(f"    Train R²: {trained_model['train_r2']:.4f}")
        print(f"    CV R² (mean ± std): {trained_model['cv_mean']:.4f} ± {trained_model['cv_std']:.4f}")
    
    # Run LinUCB experiments with each pre-trained model
    print("\nRunning LinUCB experiments with different pre-trained models...")
    results = {}
    
    for model_name, trained_model in pretrained_models.items():
        print(f"  Running experiments with {model_name}...")
        
        all_cumulative_regrets = []
        all_rewards = []
        
        for exp_run in range(config.N_EXPERIMENTS):
            result = run_experiment_with_model(config, trained_model, alpha=1.0)
            all_cumulative_regrets.append(result['cumulative_regret'])
            all_rewards.append(result['rewards'])
        
        mean_cumulative_regret = np.mean(all_cumulative_regrets, axis=0)
        std_cumulative_regret = np.std(all_cumulative_regrets, axis=0)
        mean_rewards = np.mean(all_rewards, axis=0)
        
        results[model_name] = {
            'mean_cumulative_regret': mean_cumulative_regret,
            'std_cumulative_regret': std_cumulative_regret,
            'mean_rewards': mean_rewards,
            'color': trained_model['color'],
            'final_regret': mean_cumulative_regret[-1],
            'model_performance': model_performances[model_name]
        }
        
        print(f"    Final Cumulative Regret: {mean_cumulative_regret[-1]:.2f}")
    
    # Create comprehensive visualization
    create_enhanced_models_comparison_visualization(results, model_performances, complexity_level)
    
    return results, model_performances

# ==================================
# 8. Enhanced Visualization
# ==================================
def create_enhanced_models_comparison_visualization(results, model_performances, complexity_level):
    """Create enhanced visualization comparing different pre-trained models"""
    
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)
    
    # 1. Cumulative Regret Comparison (Top Left - spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    for model_name, data in results.items():
        time_steps = range(len(data['mean_cumulative_regret']))
        ax1.plot(time_steps, data['mean_cumulative_regret'], 
                color=data['color'], linewidth=3, label=model_name, alpha=0.85)
        ax1.fill_between(time_steps, 
                        data['mean_cumulative_regret'] - data['std_cumulative_regret'],
                        data['mean_cumulative_regret'] + data['std_cumulative_regret'],
                        color=data['color'], alpha=0.25)
    
    ax1.set_title(f'Cumulative Regret: Different Models (Complexity: {complexity_level.title()})', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time Steps', fontsize=12)
    ax1.set_ylabel('Cumulative Regret', fontsize=12)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Model Performance vs LinUCB Performance (Top Right - spans 2 columns) 
    ax2 = fig.add_subplot(gs[0, 2:])
    final_regrets = [data['final_regret'] for data in results.values()]
    cv_scores = [data['model_performance']['cv_mean'] for data in results.values()]
    colors = [data['color'] for data in results.values()]
    labels = list(results.keys())
    
    # Identify Lasso for special handling
    lasso_indices = [i for i, label in enumerate(labels) if 'Lasso' in label]
    non_lasso_indices = [i for i, label in enumerate(labels) if 'Lasso' not in label]
    
    # Plot non-Lasso models normally
    if non_lasso_indices:
        non_lasso_cv = [cv_scores[i] for i in non_lasso_indices]
        non_lasso_regrets = [final_regrets[i] for i in non_lasso_indices]
        non_lasso_colors = [colors[i] for i in non_lasso_indices]
        non_lasso_labels = [labels[i] for i in non_lasso_indices]
        
        scatter = ax2.scatter(non_lasso_cv, non_lasso_regrets, c=non_lasso_colors, 
                             s=150, alpha=0.7, edgecolors='black', linewidth=2)
        
        for i, label in enumerate(non_lasso_labels):
            ax2.annotate(label.split(' ')[0], (non_lasso_cv[i], non_lasso_regrets[i]), 
                        xytext=(8, 8), textcoords='offset points', fontsize=10, fontweight='bold')
    
    # Plot Lasso models with special annotation
    if lasso_indices:
        for idx in lasso_indices:
            # Plot Lasso with different marker
            ax2.scatter(cv_scores[idx], final_regrets[idx], c=colors[idx], 
                       s=150, alpha=0.9, edgecolors='black', linewidth=3, 
                       marker='^', label='Lasso (Poor Fit)')
            
            # Special annotation for Lasso
            ax2.annotate(f"Lasso\n(R²={cv_scores[idx]:.3f})", 
                        (cv_scores[idx], final_regrets[idx]),
                        xytext=(15, -15), textcoords='offset points', 
                        fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'))
    
    ax2.set_xlabel('Model CV R² Score', fontsize=12)
    ax2.set_ylabel('Final Cumulative Regret', fontsize=12)
    ax2.set_title('Prediction Accuracy vs LinUCB Performance', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add trend line for non-Lasso models only if we have enough points
    if len(non_lasso_indices) >= 3:
        # Extract non-Lasso data for trend analysis
        non_lasso_cv = [cv_scores[i] for i in non_lasso_indices]
        non_lasso_regrets = [final_regrets[i] for i in non_lasso_indices]
        
        z = np.polyfit(non_lasso_cv, non_lasso_regrets, 1)
        p = np.poly1d(z)
        ax2.plot(non_lasso_cv, p(non_lasso_cv), "r--", alpha=0.7, linewidth=2, 
                label='Trend (excl. Lasso)')
        
        correlation = np.corrcoef(non_lasso_cv, non_lasso_regrets)[0, 1]
        ax2.text(0.05, 0.95, f'Correlation: {correlation:.3f}\n(excluding Lasso)', 
                transform=ax2.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=11)
    else:
        # Calculate correlation with all models if not enough non-Lasso models
        correlation = np.corrcoef(cv_scores, final_regrets)[0, 1]
        ax2.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax2.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=11)
    
    # Add legend if Lasso is present
    if lasso_indices:
        ax2.legend(loc='upper right', fontsize=10)
    
    # 3. Model Performance Comparison (Middle Row - spans all columns)
    ax3 = fig.add_subplot(gs[1, :])
    
    model_names = list(model_performances.keys())
    train_r2_scores = [perf['train_r2'] for perf in model_performances.values()]
    cv_r2_scores = [perf['cv_mean'] for perf in model_performances.values()]
    cv_stds = [perf['cv_std'] for perf in model_performances.values()]
    
    x_pos = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax3.bar(x_pos - width/2, train_r2_scores, width, 
                    label='Training R²', alpha=0.8, color='skyblue', edgecolor='black')
    bars2 = ax3.errorbar(x_pos + width/2, cv_r2_scores, yerr=cv_stds, 
                        fmt='o', capsize=5, capthick=2, color='red', 
                        markersize=10, label='CV R² (mean ± std)', linewidth=2)
    
    ax3.set_xlabel('Pre-trained Models', fontsize=12)
    ax3.set_ylabel('R² Score', fontsize=12)
    ax3.set_title('S\'_t Prediction Performance Comparison', fontsize=14, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([name.split(' ')[0] for name in model_names], rotation=45, ha='right')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars1, train_r2_scores):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 4. Final LinUCB Performance Bar Chart (Bottom Left)
    ax4 = fig.add_subplot(gs[2, 0])
    final_regrets = [data['final_regret'] for data in results.values()]
    model_names_short = [name.split(' ')[0] for name in results.keys()]
    bar_colors = [data['color'] for data in results.values()]
    
    bars = ax4.bar(range(len(model_names_short)), final_regrets, 
                   color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.set_title('Final Cumulative Regret\nby Model', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Final Cumulative Regret', fontsize=11)
    ax4.set_xticks(range(len(model_names_short)))
    ax4.set_xticklabels(model_names_short, rotation=45, ha='right', fontsize=10)
    
    for bar, value in zip(bars, final_regrets):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 5. Prediction Uncertainty Comparison (Bottom Middle Left)
    ax5 = fig.add_subplot(gs[2, 1])
    pred_stds = [perf['prediction_std'] for perf in model_performances.values()]
    bars = ax5.bar(range(len(model_names_short)), pred_stds, 
                   color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax5.set_title('Model Prediction\nUncertainty', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Prediction Std Dev', fontsize=11)
    ax5.set_xticks(range(len(model_names_short)))
    ax5.set_xticklabels(model_names_short, rotation=45, ha='right', fontsize=10)
    
    for bar, value in zip(bars, pred_stds):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 6. Performance Gap Analysis (Bottom Middle Right)
    ax6 = fig.add_subplot(gs[2, 2])
    best_regret = min(final_regrets)
    regret_gaps = [regret - best_regret for regret in final_regrets]
    
    bars = ax6.bar(range(len(model_names_short)), regret_gaps, 
                   color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax6.set_title('Performance Gap\n(vs Best Model)', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Regret Gap', fontsize=11)
    ax6.set_xticks(range(len(model_names_short)))
    ax6.set_xticklabels(model_names_short, rotation=45, ha='right', fontsize=10)
    
    for bar, value in zip(bars, regret_gaps):
        height = bar.get_height()
        if height > 0:
            ax6.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'+{value:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 7. Ranking Summary (Bottom Right)
    ax7 = fig.add_subplot(gs[2, 3])
    ax7.axis('off')
    
    # Create ranking based on final regret (lower is better)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['final_regret'])
    
    ranking_text = f"Performance Ranking\n({complexity_level.title()} Complexity)\n\n"
    for i, (model_name, data) in enumerate(sorted_results):
        ranking_text += f"{i+1}. {model_name.split(' ')[0]}\n"
        ranking_text += f"   Regret: {data['final_regret']:.1f}\n"
        ranking_text += f"   Model R²: {data['model_performance']['cv_mean']:.3f}\n"
        if i < 3:  # Show top 3
            ranking_text += "\n"
    
    ranking_text += "\nKey Insights:\n"
    best_model = sorted_results[0][0].split(' ')[0]
    worst_model = sorted_results[-1][0].split(' ')[0] 
    ranking_text += f"• Best: {best_model}\n"
    ranking_text += f"• Worst: {worst_model}\n"
    ranking_text += f"• Gap: {sorted_results[-1][1]['final_regret'] - sorted_results[0][1]['final_regret']:.1f}"
    
    ax7.text(0.05, 0.95, ranking_text, transform=ax7.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle(f'LinUCB with Different Pre-trained Models: Enhanced Analysis (Complexity: {complexity_level.title()})', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(os.path.join(OUTPUT_DIR, f'enhanced_models_comparison_{complexity_level}.png'), dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()

# ==================================
# 9. Multi-Complexity Analysis
# ==================================
def run_multi_complexity_analysis():
    """Run analysis across different complexity levels"""
    
    print("="*80)
    print("MULTI-COMPLEXITY ANALYSIS")
    print("="*80)
    print("Running experiments with different data complexity levels...")
    
    complexity_levels = ["low", "medium", "high"]
    all_results = {}
    all_performances = {}
    
    for complexity in complexity_levels:
        print(f"\n{'='*20} COMPLEXITY LEVEL: {complexity.upper()} {'='*20}")
        results, performances = run_enhanced_models_comparison(complexity)
        all_results[complexity] = results
        all_performances[complexity] = performances
        
        # Print summary for this complexity level
        print(f"\nSummary for {complexity.title()} Complexity:")
        sorted_models = sorted(results.items(), key=lambda x: x[1]['final_regret'])
        print(f"  Best Model: {sorted_models[0][0]} (Regret: {sorted_models[0][1]['final_regret']:.1f})")
        print(f"  Worst Model: {sorted_models[-1][0]} (Regret: {sorted_models[-1][1]['final_regret']:.1f})")
        print(f"  Performance Spread: {sorted_models[-1][1]['final_regret'] - sorted_models[0][1]['final_regret']:.1f}")
    
    # Create cross-complexity comparison
    create_cross_complexity_visualization(all_results, all_performances)
    
    return all_results, all_performances

def create_cross_complexity_visualization(all_results, all_performances):
    """Create visualization comparing results across complexity levels"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Model Performance Across Different Data Complexity Levels', 
                 fontsize=16, fontweight='bold')
    
    complexity_levels = ["low", "medium", "high"]
    
    # Get all unique model names
    all_model_names = set()
    for results in all_results.values():
        all_model_names.update(results.keys())
    all_model_names = sorted(list(all_model_names))
    
    # Create color map for models
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(all_model_names)))
    model_colors = {name: color for name, color in zip(all_model_names, colors)}
    
    # Plot 1-3: Cumulative Regret for each complexity level
    for i, complexity in enumerate(complexity_levels):
        ax = axes[0, i]
        results = all_results[complexity]
        
        for model_name, data in results.items():
            time_steps = range(len(data['mean_cumulative_regret']))
            ax.plot(time_steps, data['mean_cumulative_regret'], 
                   color=model_colors[model_name], linewidth=2.5, 
                   label=model_name.split(' ')[0], alpha=0.8)
        
        ax.set_title(f'{complexity.title()} Complexity', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Steps', fontsize=11)
        ax.set_ylabel('Cumulative Regret', fontsize=11)
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Final regret comparison across complexities
    ax = axes[1, 0]
    width = 0.8 / len(complexity_levels)
    x_pos = np.arange(len(all_model_names))
    
    for i, complexity in enumerate(complexity_levels):
        results = all_results[complexity]
        final_regrets = []
        for model_name in all_model_names:
            if model_name in results:
                final_regrets.append(results[model_name]['final_regret'])
            else:
                final_regrets.append(0)
        
        ax.bar(x_pos + i * width, final_regrets, width, 
               label=f'{complexity.title()}', alpha=0.8)
    
    ax.set_title('Final Regret Across Complexities', fontsize=12, fontweight='bold')
    ax.set_xlabel('Models', fontsize=11)
    ax.set_ylabel('Final Cumulative Regret', fontsize=11)
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels([name.split(' ')[0] for name in all_model_names], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Model prediction performance across complexities
    ax = axes[1, 1]
    for i, complexity in enumerate(complexity_levels):
        performances = all_performances[complexity]
        cv_scores = []
        for model_name in all_model_names:
            if model_name in performances:
                cv_scores.append(performances[model_name]['cv_mean'])
            else:
                cv_scores.append(0)
        
        ax.bar(x_pos + i * width, cv_scores, width, 
               label=f'{complexity.title()}', alpha=0.8)
    
    ax.set_title('Model R² Across Complexities', fontsize=12, fontweight='bold')
    ax.set_xlabel('Models', fontsize=11)
    ax.set_ylabel('CV R² Score', fontsize=11)
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels([name.split(' ')[0] for name in all_model_names], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Performance spread analysis
    ax = axes[1, 2]
    spreads = []
    labels = []
    
    for complexity in complexity_levels:
        results = all_results[complexity]
        regrets = [data['final_regret'] for data in results.values()]
        spread = max(regrets) - min(regrets)
        spreads.append(spread)
        labels.append(complexity.title())
    
    bars = ax.bar(labels, spreads, color=['lightblue', 'orange', 'lightcoral'], 
                  alpha=0.8, edgecolor='black', linewidth=2)
    ax.set_title('Performance Spread\nby Complexity', fontsize=12, fontweight='bold')
    ax.set_ylabel('Regret Spread (Max - Min)', fontsize=11)
    
    for bar, value in zip(bars, spreads):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'multi_complexity_analysis.png'), dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()

# ==================================
# 10. Enhanced Summary and Analysis
# ==================================
def print_enhanced_detailed_summary(all_results, all_performances):
    """Print enhanced detailed summary of results across complexity levels"""
    
    print("\n" + "="*100)
    print("ENHANCED DETAILED RESULTS SUMMARY")
    print("="*100)
    
    complexity_levels = ["low", "medium", "high"]
    
    for complexity in complexity_levels:
        print(f"\n{'='*30} {complexity.upper()} COMPLEXITY {'='*30}")
        results = all_results[complexity]
        performances = all_performances[complexity]
        
        print(f"\nMODEL PREDICTION PERFORMANCE ({complexity.title()}):")
        print("-" * 60)
        sorted_by_r2 = sorted(performances.items(), key=lambda x: x[1]['cv_mean'], reverse=True)
        
        for i, (model_name, perf) in enumerate(sorted_by_r2):
            print(f"{i+1:2d}. {model_name:25s}: R² = {perf['cv_mean']:.4f} ± {perf['cv_std']:.4f}")
        
        print(f"\nLINUCB PERFORMANCE ({complexity.title()}):")
        print("-" * 60)
        sorted_by_regret = sorted(results.items(), key=lambda x: x[1]['final_regret'])
        
        for i, (model_name, data) in enumerate(sorted_by_regret):
            print(f"{i+1:2d}. {model_name:25s}: Regret = {data['final_regret']:8.1f}")
        
        # Calculate correlations
        r2_scores = [performances[name]['cv_mean'] for name in results.keys()]
        regrets = [results[name]['final_regret'] for name in results.keys()]
        correlation = np.corrcoef(r2_scores, regrets)[0, 1]
        
        print(f"\nKEY INSIGHTS FOR {complexity.upper()} COMPLEXITY:")
        print("-" * 60)
        print(f"• Best LinUCB Model: {sorted_by_regret[0][0]} (Regret: {sorted_by_regret[0][1]['final_regret']:.1f})")
        print(f"• Best Predictor: {sorted_by_r2[0][0]} (R²: {sorted_by_r2[0][1]['cv_mean']:.4f})")
        print(f"• Performance Spread: {sorted_by_regret[-1][1]['final_regret'] - sorted_by_regret[0][1]['final_regret']:.1f}")
        print(f"• Correlation (R² vs Regret): {correlation:.3f}")
        
        if correlation < -0.5:
            print(f"• Strong negative correlation: Better predictors → Lower regret")
        elif correlation > 0.5:
            print(f"• Strong positive correlation: Better predictors → Higher regret (unexpected!)")
        else:
            print(f"• Weak correlation: Prediction accuracy doesn't strongly predict LinUCB performance")
    
    print(f"\n{'='*30} CROSS-COMPLEXITY INSIGHTS {'='*30}")
    
    # Analyze which models perform consistently well
    model_rankings = {}
    for complexity in complexity_levels:
        results = all_results[complexity]
        sorted_models = sorted(results.items(), key=lambda x: x[1]['final_regret'])
        for rank, (model_name, _) in enumerate(sorted_models):
            if model_name not in model_rankings:
                model_rankings[model_name] = []
            model_rankings[model_name].append(rank + 1)
    
    print("\nCONSISTENT PERFORMERS (Average Ranking):")
    print("-" * 60)
    avg_rankings = {name: np.mean(ranks) for name, ranks in model_rankings.items()}
    sorted_avg = sorted(avg_rankings.items(), key=lambda x: x[1])
    
    for i, (model_name, avg_rank) in enumerate(sorted_avg):
        ranks_str = ', '.join([str(r) for r in model_rankings[model_name]])
        print(f"{i+1:2d}. {model_name:25s}: Avg Rank = {avg_rank:.1f} (Ranks: {ranks_str})")
    
    # Analyze performance spread trends
    print(f"\nPERFORMANCE SPREAD ANALYSIS:")
    print("-" * 60)
    for complexity in complexity_levels:
        results = all_results[complexity]
        regrets = [data['final_regret'] for data in results.values()]
        spread = max(regrets) - min(regrets)
        print(f"• {complexity.title():8s} Complexity: Spread = {spread:6.1f}")
    
    print(f"\nCONCLUSIONS:")
    print("-" * 60)
    print("• Higher complexity data should create larger performance gaps between models")
    print("• Tree-based models (RF, GB) should excel in high complexity scenarios")  
    print("• Linear models should struggle as complexity increases")
    print("• Neural networks should show improved relative performance in complex scenarios")

# ==================================
# 11. Main Function
# ==================================
def main():
    """Enhanced main execution function"""
    print("Enhanced LinUCB with Model Mismatch Analysis")
    print("="*80)
    print("This experiment creates different data complexity levels to demonstrate")
    print("model mismatch scenarios where simple models fail and complex models excel.")
    print("\nComplexity Levels:")
    print("• LOW: Simple linear relationships (favors linear models)")
    print("• MEDIUM: Polynomial + interactions (favors neural networks, polynomial)")  
    print("• HIGH: Complex conditionals + nonlinearities (favors tree-based models)")
    
    try:
        # Option 1: Run single complexity level
        # results, performances = run_enhanced_models_comparison("high")
        # print_detailed_summary(results, performances)
        
        # Option 2: Run full multi-complexity analysis
        all_results, all_performances = run_multi_complexity_analysis()
        print_enhanced_detailed_summary(all_results, all_performances)
        
        return all_results, all_performances
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    main()
