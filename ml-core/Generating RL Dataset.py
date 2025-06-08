# ✅ Install dependencies
!pip install gymnasium[classic-control] stable-baselines3 shap lightgbm scikit-learn pandas --quiet

# 📥 Imports
import gymnasium as gym
import numpy as np
import pandas as pd
import shap
import lightgbm as lgb
import os, json
from gymnasium import spaces
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from stable_baselines3 import DQN

# 📁 Output Directory
output_dir = '/content/drive/MyDrive/TrustFusion/LightGBM-RL'
os.makedirs(output_dir, exist_ok=True)

# 📂 Load dataset
df = pd.read_csv('/content/drive/MyDrive/TrustFusion/preprocessed/preprocessed-dataset.csv')
target_col = 'user_trust_tier'
X = df.drop(columns=[target_col])
y = df[target_col].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 🎯 Feature modalities
behavioral_feats = ['usage_frequency', 'peer_rating', 'violation_count', 'complaint_count']
contextual_feats = ['location', 'term', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
profile_feats = ['account_age_days', 'total_interactions', 'admin_flag']
modality_groups = {
    "Behavioral": behavioral_feats,
    "Contextual": contextual_feats,
    "Profile Priors": profile_feats,
    "Others": list(set(X.columns) - set(behavioral_feats + contextual_feats + profile_feats))
}

# ✅ Train LightGBM model on profile priors only
model_profile = lgb.train({
    'objective': 'multiclass', 'num_class': 4, 'metric': 'multi_logloss',
    'learning_rate': 0.05, 'verbose': -1, 'random_state': 42
}, lgb.Dataset(X_train[profile_feats], label=y_train), num_boost_round=100)

# 💾 Save trained LightGBM model
model_path = os.path.join(output_dir, 'lightgbm-profile-model.txt')
model_profile.save_model(model_path)
print(f"✅ LightGBM model saved to: {model_path}")

# ✅ LightGBM Wrapper for RL use
class LGBWrapper:
    def __init__(self, booster): self.booster = booster
    def predict(self, X): return np.argmax(self.booster.predict(X.values.reshape(1, -1)), axis=1)

wrapped_model = LGBWrapper(model_profile)

# ✅ SHAP explainer (on profile priors)
explainer = shap.TreeExplainer(model_profile)

# 🎮 Custom RL Environment
class TrustFusionEnv(gym.Env):
    def __init__(self, X, y, model):
        super().__init__()
        self.X = X.reset_index(drop=True)
        self.y = y.reset_index(drop=True)
        self.model = model
        self.current_idx = 0
        self.step_count = 0
        self.action_space = spaces.Discrete(len(np.unique(y)))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(X.shape[1],), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.current_idx = np.random.randint(0, len(self.X))
        obs = self.X.iloc[self.current_idx].values.astype(np.float32)
        return obs, {}

    def step(self, action):
        self.step_count += 1
        state = self.X.iloc[self.current_idx]
        true_label = self.y.iloc[self.current_idx]
        pred = self.model.predict(state[profile_feats])[0]

        reward = 1.0 if action == true_label else -1.0
        weight_factor = min(1.0, self.step_count / 25)

        shap_values = explainer.shap_values(state[profile_feats].to_frame().T)
        mean_shap = np.abs(np.stack(shap_values)).mean(axis=(0, 1))
        shap_dict = {f: float(v) for f, v in zip(state[profile_feats].index.tolist(), mean_shap)}
        total = sum(shap_dict.values()) or 1.0

        profile_ratio = sum(shap_dict.get(f, 0.0) for f in profile_feats) / total
        behavior_ratio = sum(shap_dict.get(f, 0.0) for f in behavioral_feats if f in state.index) / total
        context_ratio = sum(shap_dict.get(f, 0.0) for f in contextual_feats if f in state.index) / total

        if action == true_label:
            reward += 1.5 * behavior_ratio + 1.0 * context_ratio
            reward -= 1.5 * profile_ratio

        reward = np.clip(reward * weight_factor, -2.0, 2.0)
        obs, _ = self.reset()
        return obs, reward, True, False, {}

# ✅ Train DQN agent
env = TrustFusionEnv(X_test, y_test, wrapped_model)
agent = DQN("MlpPolicy", env, learning_rate=0.005, exploration_fraction=0.4,
            exploration_final_eps=0.05, buffer_size=5000, batch_size=32,
            learning_starts=100, target_update_interval=200, device="cpu", verbose=1)
agent.learn(total_timesteps=10000)

# 💾 Save trained RL agent
agent_path = os.path.join(output_dir, 'dqn-trustfusion-agent.zip')
agent.save(agent_path)
print(f"✅ DQN agent saved to: {agent_path}")

# 🧪 Simulate RL-enhanced trust interactions
rl_samples = []
for _ in range(1000):
    obs, _ = env.reset()
    for _ in range(25):
        action, _ = agent.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        if done: break
    rl_samples.append(obs)

# 💾 Save RL-enhanced interaction dataset
X_rl = pd.DataFrame(rl_samples, columns=X.columns)
rl_data_path = os.path.join(output_dir, 'rl-generated-data.csv')
X_rl.to_csv(rl_data_path, index=False)
print(f"✅ RL-enhanced dataset saved to: {rl_data_path}")

# ✅ Optional: Evaluate accuracy of profile-only model (for paper)
y_pred = np.argmax(model_profile.predict(X_test[profile_feats]), axis=1)
acc = accuracy_score(y_test, y_pred)
summary = {
    "original_model_accuracy": round(acc, 4),
    "rl_samples_generated": len(X_rl),
    "agent_timesteps": 10000,
    "reward_design": "SHAP-driven modality-weighted reward"
}
summary_path = os.path.join(output_dir, 'summary.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=4)
print(f"📄 Summary saved to: {summary_path}")

# 💾 Save SHAP modality attribution from one sample interaction
shap_sample = X_test.sample(1, random_state=42)
shap_values = explainer.shap_values(shap_sample[profile_feats])
mean_shap = np.abs(np.stack(shap_values)).mean(axis=(0, 1))
shap_dict = {f: float(v) for f, v in zip(profile_feats, mean_shap)}
shap_total = sum(shap_dict.values())
shap_modality = {
    "Profile Priors": round(100 * shap_total, 2),
    "Contextual": 0.0,
    "Behavioral": 0.0,
    "Others": 0.0
}

# 💾 Save SHAP attribution
shap_json_path = os.path.join(output_dir, 'shap-attribution-sample.json')
with open(shap_json_path, 'w') as f:
    json.dump(shap_modality, f, indent=4)
print(f"✅ SHAP attribution saved to: {shap_json_path}")
