import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go
import argparse
from plotly.subplots import make_subplots


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="display dimension reduction")
    parser.add_argument("name", help="name of the dataset")
    args = parser.parse_args()
    name = args.name
    folder = "../DATA/"+name+"/"
    files = os.listdir(folder)
    fig = make_subplots(rows=3, cols=2)
    for i, file in enumerate(files):
        data = pd.read_csv(folder+file).drop(columns="Unnamed: 0").to_numpy()
        fig.add_trace(go.Scatter(
            x=data[:, 0], y=data[:, 1], mode='markers', name=file), row=(i//2)+1, col=(i % 2)+1)
        fig.update_xaxes(title_text="X", row=(i//2)+1, col=(i % 2)+1)
        fig.update_yaxes(title_text="Y", row=(i//2)+1, col=(i % 2)+1)
        fig.update_layout(title_text=name)
    fig.show()
