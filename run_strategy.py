def run_ensemble_strategy(df, unique_trade_date, rebalance_window, validation_window) -> None:
    """Ensemble Strategy that combines PPO, A2C and DDPG"""
    print("============Start Ensemble Strategy============")
    # for ensemble model, it's necessary to feed the last state
    # of the previous model to the current model as the initial state
    last_state_ensemble = []

    ppo_sharpe_list = []
    #ddpg_sharpe_list = []
    a2c_sharpe_list = []
    sac_sharpe_list = []
    #td3_sharpe_list = []

    model_use = []
    validation_start_date_list = []
    validation_end_date_list = []
    iteration_list = []

    # based on the analysis of the in-sample data
    insample_turbulence = df[(df.Date < "2016-10-01")
                             & (df.Date >= "2012-05-21")]
    insample_turbulence = insample_turbulence.drop_duplicates(subset=['Date'])
    insample_turbulence_threshold = np.quantile(
        insample_turbulence.turbulence.values, .90)
    #print("Insample turbulence threshold: ", insample_turbulence_threshold)

    start = time.time()
    for i in range(rebalance_window + validation_window, len(unique_trade_date), rebalance_window):
        print("============================================")
        validation_start_date = unique_trade_date[i -
                                                  rebalance_window - validation_window]
        validation_end_date = unique_trade_date[i - rebalance_window]

        validation_start_date_list.append(validation_start_date)
        validation_end_date_list.append(validation_end_date)
        iteration_list.append(i)
        # initial state is empty
        if i - rebalance_window - validation_window == 0:
            # inital state
            initial = True
        else:
            # previous state
            initial = False

        # Tuning trubulence index based on historical data
        # Turbulence lookback window is one quarter
        end_date_index = df.index[df["Date"] == unique_trade_date[i -
                                                                  rebalance_window - validation_window]].to_list()[-1]
        start_date_index = end_date_index - validation_window*30 + 1  # 30 tickers

        historical_turbulence = df.iloc[start_date_index:(
            end_date_index + 1), :]
        #historical_turbulence = df[(df.Date<unique_trade_date[i - rebalance_window - validation_window]) & (df.Date>=(unique_trade_date[i - rebalance_window - validation_window - 63]))]

        historical_turbulence = historical_turbulence.drop_duplicates(subset=[
                                                                      'Date'])
        #print("Historical_turbulence: ", historical_turbulence)
        # historical_turbulence.to_csv("/content/drive/MyDrive/DRL_DataVN_sb3/test_date/historical_tur_{}.csv".format(i))

        historical_turbulence_mean = np.mean(
            historical_turbulence.turbulence.values)
        print("Historical turbulence mean: ", historical_turbulence_mean)
        print("Insample turbulence threshold: ", insample_turbulence_threshold)

        if historical_turbulence_mean > insample_turbulence_threshold:
            # if the mean of the historical data is greater than the 90% quantile of insample turbulence data
            # then we assume that the current market is volatile,
            # therefore we set the 90% quantile of insample turbulence data as the turbulence threshold
            # meaning the current turbulence can't exceed the 90% quantile of insample turbulence data
            turbulence_threshold = np.quantile(
                insample_turbulence.turbulence.values, 1)
        else:
            # if the mean of the historical data is less than the 90% quantile of insample turbulence data
            # then we tune up the turbulence_threshold, meaning we lower the risk
            turbulence_threshold = np.quantile(
                insample_turbulence.turbulence.values, 0.99)

        turbulence_threshold = np.quantile(
            insample_turbulence.turbulence.values, 0.999)
        print("Turbulence_threshold: ", turbulence_threshold)
        print("Value of i = ", i)
        #print("======Trading from: ", unique_trade_date[i - rebalance_window], "to ", unique_trade_date[i])

        ############## Environment Setup starts ##############
        # training env
        #print("Value of i in training = ", i)
        train = data_split(
            df, start="2012-05-21", end=unique_trade_date[i - rebalance_window - validation_window])
        env_train = DummyVecEnv([lambda: StockEnvTrain(train)])

        # validation env
        #print("Value of i in validation = ", i)
        validation = data_split(df, start=unique_trade_date[i - rebalance_window - validation_window],
                                end=unique_trade_date[i - rebalance_window])
        env_val = DummyVecEnv([lambda: StockEnvValidation(validation,
                                                          turbulence_threshold=turbulence_threshold,
                                                          iteration=i)])
        obs_val = env_val.reset()
        ############## Environment Setup ends ##############

        ############## Training and Validation starts ##############
        print("======Model training from: 2012-05-21 to ",
              unique_trade_date[i - rebalance_window - validation_window])
        print("==============Model Training===========")

        print("======A2C Training========")
        model_a2c = train_A2C(
            env_train, model_name="A2C_80k_dow_{}".format(i), timesteps=80000)
        print("======A2C Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
              unique_trade_date[i - rebalance_window])
        DRL_validation(model=model_a2c, test_data=validation,
                       test_env=env_val, test_obs=obs_val)
        sharpe_a2c = get_validation_sharpe(i)
        print("A2C Sharpe Ratio: ", sharpe_a2c)

        print("======PPO Training========")
        model_ppo = train_PPO(
            env_train, model_name="PPO_70k_dow_{}".format(i), timesteps=70000)
        print("======PPO Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
              unique_trade_date[i - rebalance_window])
        DRL_validation(model=model_ppo, test_data=validation,
                       test_env=env_val, test_obs=obs_val)
        sharpe_ppo = get_validation_sharpe(i)
        print("PPO Sharpe Ratio: ", sharpe_ppo)

        print("======SAC Training========")
        model_sac = train_SAC(
            env_train, model_name="SAC_20k_dow_{}".format(i), timesteps=20000)
        print("======SAC Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
              unique_trade_date[i - rebalance_window])
        DRL_validation(model=model_sac, test_data=validation,
                       test_env=env_val, test_obs=obs_val)
        sharpe_sac = get_validation_sharpe(i)
        print("SAC Sharpe Ratio: ", sharpe_sac)

        a2c_sharpe_list.append(sharpe_a2c)
        ppo_sharpe_list.append(sharpe_ppo)
        sac_sharpe_list.append(sharpe_sac)

        # Model Selection based on sharpe ratio

        if (sharpe_ppo >= sharpe_a2c) & (sharpe_ppo >= sharpe_sac):
            model_ensemble = model_ppo
            model_use.append('PPO')
        elif (sharpe_sac > sharpe_ppo) & (sharpe_sac > sharpe_a2c):
            model_ensemble = model_sac
            model_use.append('SAC')
        else:
            model_ensemble = model_a2c
            model_use.append('A2C')

        ############## Training and Validation ends ##############

        ############## Trading starts ##############
        print("======Trading from: ",
              unique_trade_date[i - rebalance_window], "to ", unique_trade_date[i])
        print("Used Model: ", model_ensemble)
        last_state_ensemble = DRL_prediction(df=df, model=model_ensemble, name="ensemble",
                                             last_state=last_state_ensemble, iter_num=i,
                                             unique_trade_date=unique_trade_date,
                                             rebalance_window=rebalance_window,
                                             turbulence_threshold=turbulence_threshold,
                                             initial=initial)
        print("============Trading Done============")
        ############## Trading ends ##############

    end = time.time()
    print("Ensemble Strategy took: ", (end - start) / 60, " minutes")

    df_summary = pd.DataFrame([iteration_list, validation_start_date_list, validation_end_date_list,
                              model_use, sac_sharpe_list, ppo_sharpe_list, a2c_sharpe_list]).T
    df_summary.columns = ['Iter', 'Val Start', 'Val End',
                          'Model Used', 'SAC Sharpe', 'PPO Sharpe', 'A2C Sharpe']
    df_summary.to_csv(
        "/content/drive/MyDrive/DRL_DataVN_sb3/results/Model_used.csv")

    return df_summary


def run_model() -> None:
    """Train the model."""

    # read and preprocess data
    preprocessed_path = "/content/drive/MyDrive/data_vn/30stocks_done(1).csv"
    if os.path.exists(preprocessed_path):
        data = pd.read_csv(preprocessed_path, index_col=0)
    else:
        data = preprocess_data()
        data = add_turbulence(data)
        data.to_csv(preprocessed_path)

    print(data.head())
    print(data.size)

    # 2016/10/01 is the date that validation starts
    # 2017/01/01 is the date that real trading starts
    # unique_trade_date needs to start from 2016/10/01 for validation purpose
    unique_trade_date = data[(data.Date >= "2016-10-01")
                             & (data.Date <= "2020-12-31")].Date.unique()
    # print(unique_trade_date)

    # rebalance_window is the number of months to retrain the model
    # validation_window is the number of months to validation the model and select for trading
    rebalance_window = 63
    validation_window = 63

    # Ensemble Strategy
    summary = run_ensemble_strategy(df=data,
                                    unique_trade_date=unique_trade_date,
                                    rebalance_window=rebalance_window,
                                    validation_window=validation_window)
    print(summary)

    #_logger.info(f"saving model version: {_version}")


if __name__ == "__main__":
    run_model()
