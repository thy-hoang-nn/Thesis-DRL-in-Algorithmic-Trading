preprocessed_path = "/content/drive/MyDrive/data_vn/30stocks_done(1).csv"
data = pd.read_csv(preprocessed_path, index_col=0)
train = data_split(data, start="2012-05-21", end="2016-10-01")
env_train = DummyVecEnv([lambda: StockEnvTrain(train)])

model = A2C('MlpPolicy', env_train, ent_coef=1,
            tensorboard_log="./a2c_tensorboard/")
model.learn(total_timesteps=100000)

model = PPO('MlpPolicy', env_train, ent_coef=0.01,
            tensorboard_log="./ppo_tensorboard/")
model.learn(total_timesteps=100000)

model = SAC('MlpPolicy', env_train, ent_coef='auto_0.1',
            batch_size=128, tensorboard_log="./sac_tensorboard/")
model.learn(total_timesteps=100000)
