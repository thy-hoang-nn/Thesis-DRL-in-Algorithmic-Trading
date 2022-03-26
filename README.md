# Thesis_DRL-in-Algorithmic-Trading

## Research Ideas
Different from a developed market, a pre-emerging market like Vietnam has unique characteristics which has gained more and more attention due to the promising high profits. This thesis aims to develop an stock trading for Vietnam stock market. The proposed solution includes a set of selective indicators and a learning framework which can train a good trading policy.

## Research Methods
The main purpose is to propose an deep reinforcement learning approach for training a trading policy which including the design and implementation of simulator environment, feature sets and learning algorithms. The feature set is a set of selective indicators which is proposed by using literature review and data analysis techniques. Secondly, an ensemble strategy is adopted from a state-of-the-art study, which is selected as the baseline, with a modification on selected learning algorithms. Finally, the proposed model is compared with the selected baseline and the VN30-INDEX using several financial metrics such as Cumulative Return and Sharpe Ratio, etc.


## Dataset
- Sources of data and changing compositions of VN30 are available on websites ([cafef.vn](https://s.cafef.vn/Lich-su-giao-dich-VN30INDEX-1.chn#data), [vietstock.vn](https://finance.vietstock.vn/))

## Hypeparameter Optimization
- [Optuna](https://optuna.readthedocs.io/en/stable/) - a useful toolkit is conducted to tuning hyperparameters for StableBaseline3 models.

## Performance under market crash
- After adjusting the turbulence threshold in an ensemble strategy, the agent can detect the confusing situation and escape the market crash in 2020. In addition, proposed technical indicators provide advantages to recognize the 2018 rising peak and trade as usual. As a result, our method can reduce losses and obtain positive returns in chaotic environment cases.
![image](https://user-images.githubusercontent.com/37342769/160243523-00b377fb-4099-4a9d-83af-78d2a40428c2.png)

## Result
This work is proposing a deep reinforcement learning which achieves high performance in both cumulative return risk management, compared to the selected baselines, and also successfully handles two extreme periods of Vietnam market in 2018 and 2020 that is a challenge for the state-of-the-art approaches.

|                       | PPO   |  SAC  |  A2C |  Ensemble |  VN30  |
|     :---:             | :---: | :---: | :---:|  :---:    |  :---: |
| **Cumulative Return** | 35.54 |42.97  | 37.99|  54.47    |  42.30 |
| **Annual Return**     |8.44   |9.99   | 8.96 | 12.28     |  9.85  |
| **Annual Volatility** |11.39  |17.65  | 15.94| 16.99     |18.69   |
|**Sharpe Ratio**       |0.77   | 0.63  | 0.27 |0.77       |0.60    |
|**Max drawdown**       |-28.71 |-36.97 |-33.51|-33.32     |-48.14  |

- PPO agent has the lowest annual volatility and max drawdown among the three agents -> PPO is good at handling a bearish market. 
- SAC is good at uptrend and carries out well in generating more profits, it has the highest cumulative return and annual returns -> SAC is recommended when adapting to the bullish market.

![image](https://user-images.githubusercontent.com/37342769/160242332-849ecf53-939f-4231-803b-c921d009f283.png)

- We observe that our ensemble strategy outperforms the benchmark in terms of profitability. The proposed model can identify the market fall at the beginning of 2020 and act to preserve the invested capital. Besides, taking into close consideration in SAC, and then it generates more significant returns than the proposed in several periods (from 2016 to 2018). It offers an effective strategy, with a highly positive behavior that beats the market and the ensemble.







