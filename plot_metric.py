import numpy as np
import pickle as pkl
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from os import listdir, path

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", "-d", type=str)
parser.add_argument("--metric", "-m", type=str)
args = parser.parse_args()

data = []

for file in listdir(args.data_folder):
    if(path.splitext(file)[1]!=".out"):
        continue
    with open(args.data_folder+"/"+file, 'rb') as f:
        results = pkl.load(f)
        value = results[args.metric]
    data+=[value]

data = np.asarray(data)
epoch = np.tile(np.arange(data.shape[1]),(data.shape[0],1))
for d,e in zip(list(data[0]), list(epoch[0])):
    print(str(e)+": "+str(d))
data_frame = pd.DataFrame({args.metric:data.flatten(),"epoch":epoch.flatten()})

plot = sns.lineplot(x="epoch",y=args.metric,data=data_frame)

plt.show()
