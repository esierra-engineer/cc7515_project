import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

d_cpu = pd.read_csv("/media/storage/git/cc7515_project/output/outputCPU.csv")
d_gpu = pd.read_csv("/media/storage/git/cc7515_project/output/outputGPU.csv")
d_cpu["engine"] = "CPU"
d_gpu["engine"] = "GPU"

data = pd.concat([
    d_gpu, d_cpu
])

for engine, step in product(["CPU", "GPU"], data["step"].unique()):
    filename = f"/media/storage/git/cc7515_project/output/png/{engine}/output_step_{int(step):04d}.png"
    # noinspection PyTypeChecker
    plt.matshow(data[(data["step"] == step) & (data["engine"] == engine)].pivot(columns="y", index="x", values="T").values, vmin=0, vmax=100)
    plt.title(f"{engine} test, step {step}")
    plt.savefig(filename)
    plt.close()