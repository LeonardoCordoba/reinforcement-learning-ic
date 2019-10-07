import pandas as pd
import matplotlib.pyplot as plt
df_1 = pd.read_csv("atari/results/model_1/performance/performance_parciales/performance (17).csv")
df_2 = pd.read_csv("atari/results/model_1/performance/performance_parciales/performance_corrida_1.csv")
df_full = pd.concat([df_2, df_1])

df_full = df_full.reset_index(drop=True)

df_full["score"].rolling(window=100).mean().plot()
plt.savefig("plot.png")