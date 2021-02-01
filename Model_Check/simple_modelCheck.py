import pandas as pd

path = '../RSI/result.csv'
df = pd.read_csv(path)

trades_ave = df['trades'].mean()
total_return_ave = df['Total return'].mean()
average_gain_ave = df['Average Gain'].mean()
average_loss_ave = df['Average Loss'].mean()
average_max_return = df['Max Return'].mean()
average_max_loss = df['Max Loss'].mean()
average_risk_reward_ratio = df['Gain/Loss Ratio'].mean()
average_batting_ratio = df['Batting Average'].mean()
print('trade average: {}'.format(trades_ave))
print('total return average: {}'.format(total_return_ave))
print('gain average: {}'.format(average_gain_ave))
print('loss average: {}'.format(average_loss_ave))
print('max return average: {}'.format(average_max_return))
print('max loss average: {}'.format(average_max_loss))
print('Gain/Loss ratio average: {}'.format(average_risk_reward_ratio))
print('Batting ratio average: {}'.format(average_batting_ratio))