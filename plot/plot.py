import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("neural_network\plot\error_file.csv")

plt.plot(df['epochs'], df['error'])
plt.xlabel('epochs')
plt.ylabel('error')
plt.show()