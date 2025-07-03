import pandas as pd
import matplotlib.pyplot as plt


data = pd.concat([
    pd.read_csv("/media/storage/git/cc7515_project/output/benchmarkCPU.csv"),
    pd.read_csv("/media/storage/git/cc7515_project/output/benchmarkGPU.csv")
])

data["size2"] = data[["size"]].pow(2)

filename = f"/media/storage/git/cc7515_project/output/CPU_vs_GPU.png"

# Plotting each group as a separate line on the same plot
data.set_index("size2", inplace=True)
data.groupby('engine')["time"].plot(legend=True, marker="o")
plt.title('GPU vs CPU')
plt.xlabel('Elements')
plt.ylabel('Time [s]')
plt.savefig(filename)
plt.show()
plt.close()
