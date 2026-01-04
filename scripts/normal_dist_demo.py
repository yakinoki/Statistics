import numpy as np
import matplotlib.pyplot as plt

# 平均と標準偏差を設定
mu1, sigma1 = 2.0, 1.0  # 第1の正規分布の平均と標準偏差
mu2, sigma2 = 3.0, 0.5  # 第2の正規分布の平均と標準偏差

# サンプルデータ生成
np.random.seed(0)  # 再現性のために乱数シードを設定
data1 = np.random.normal(mu1, sigma1, 1000)
data2 = np.random.normal(mu2, sigma2, 1000)

# 和を計算
sum_data = data1 + data2

# 確率密度関数を計算
hist, bins = np.histogram(sum_data, bins=50, density=True)
bin_centers = 0.5 * (bins[1:] + bins[:-1])

# グラフを描画
plt.figure(figsize=(8, 5))
plt.plot(bin_centers, hist, label="Sum of Normal Distributions", linewidth=2)
plt.xlabel("Sum")
plt.ylabel("Probability Density")
plt.title("Probability Distribution of the Sum of Two Normal Distributions")
plt.legend()
plt.grid(True)

plt.show()
