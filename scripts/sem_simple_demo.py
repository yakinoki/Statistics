import os
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"

from semopy import Model
from semopy import semplot
import matplotlib.pyplot as plt

desc = """
Latent1 =~ x1 + x2 + x3
Latent2 =~ x4 + x5 + x6
Latent2 ~ Latent1
"""

model = Model(desc)

import pandas as pd
import numpy as np
np.random.seed(42)
data = pd.DataFrame({
    "x1": np.random.randn(100),
    "x2": np.random.randn(100),
    "x3": np.random.randn(100),
    "x4": np.random.randn(100),
    "x5": np.random.randn(100),
    "x6": np.random.randn(100),
})
model.fit(data)

# パス図の描画
g = semplot(model, "semopy_path_diagram.png")
print("パス図を semopy_path_diagram.png に保存しました")
