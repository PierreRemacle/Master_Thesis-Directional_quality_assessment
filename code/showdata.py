from Utils import *
import sys
import plotly.express as px
import plotly.graph_objects as go

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python showdata.py file")
        sys.exit(1)
    file = sys.argv[1]
    HD_data = pd.read_csv("../DATA/"+file+"HD.csv")
    LD_data = pd.read_csv("../DATA/"+file+"LD.csv")

    X = HD_data[HD_data.columns[1:]]
    y = LD_data[LD_data.columns[0]]
    X_embedded = LD_data[LD_data.columns[1:]]
    print(X_embedded)
    if len(HD_data.columns) == 4:
        # two plot side by side

        fig = make_subplots(
            rows=1,
            cols=2,
            start_cell="top-left",
            specs=[[{"type": "scatter3d"}, {"type": "scatter"}]]
        )
        fig.add_trace(go.Scatter3d(x=X["0"], y=X["1"], z=X["2"], mode='markers', marker=dict(
            color=y, colorscale='Viridis', opacity=0.8, size=4)), row=1, col=1)
        fig.add_trace(go.Scatter(x=X_embedded["1"], y=X_embedded["0"], mode='markers', marker=dict(
            color=y, colorscale='Viridis', opacity=0.8)), row=1, col=2)
        fig.show()
        print("Data plotted")
    else:
        fig = px.scatter(x=X_embedded["0"], y=X_embedded["1"], color=y)
        fig.show()
        print("Data plotted")
