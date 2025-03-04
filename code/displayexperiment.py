import argparse
import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def hover_template(x, y):
    out = []
    for i in range(len(x)):
        out.append(f"X: {x[i]} <br>Y: {y[i]}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="display dimension reduction")
    parser.add_argument("outputfile", help="name of the output file")
    parser.add_argument("-s", "-select", type=str,
                        help="display QNX", default="ALL")
    args = parser.parse_args()

    # display the output file

    all_results = []
    names = []
    i = 0
    with open(args.outputfile+".txt", "r") as file:

        for line in file:
            if i % 2 == 0:
                names.append(line)
            else:
                results = {}
                line = line.replace("{", "")
                line = line.replace("}", "")
                lines = line.split(",")
                for l in lines:
                    data = l.split(":")
                    results[int(data[0])] = float(data[1])

                all_results.append(results)
            i = i+1

    plt = make_subplots(rows=4, cols=1)
    colors = ['blue', 'orange', 'green', 'red', 'purple',
              'brown', 'pink', 'gray', 'olive', 'cyan']

    # # 1
    # for i, result in enumerate(all_results):
    #     x_steps = np.asarray(list(result.keys()))
    #     y_values = np.array(list(result.values()))
    #     y_diago = x_steps-2
    #     standerized_x = (x_steps-2) / np.max(x_steps-2) * 100
    #     hover = hover_template(x_steps, y_values)
    #
    #     plt.add_trace(go.Scatter(x=x_steps-2, y=y_values, mode='lines',
    #                              name=names[i] + "base", hovertemplate=hover, line={"color": colors[i]}), row=1, col=1)
    #     plt.add_trace(go.Scatter(x=x_steps-2, y=y_diago, mode='lines',
    #                              name=names[i] + "esperance", hovertemplate=hover, line={"color": colors[i], "dash": 'dot'}), row=1, col=1)
    # 2
    list_interpol = []
    for i, result in enumerate(all_results):
        x_steps = np.asarray(list(result.keys()))
        y_values = np.array(list(result.values()))
        y_diago = x_steps-2
        standerized_x = (x_steps-2) / np.max(x_steps-2) * 100
        interpolate_x = []
        for j in range(len(x_steps)):
            interpolate_x.append(np.linspace(
                x_steps[j]-2, standerized_x[j], 100))
        interpolate_x = np.array(interpolate_x).T
        list_interpol.append(interpolate_x)
        hover = hover_template(x_steps, y_values)
        plt.add_trace(go.Scatter(x=x_steps, y=y_values, mode='lines',
                                 name=names[i] + "data", hovertemplate=hover, line={"color": colors[i]}), row=1, col=1)
        plt.add_trace(go.Scatter(x=x_steps, y=y_diago, mode='lines',
                                 name=names[i] + "esperance", hovertemplate=hover, line={"color": colors[i], "dash": 'dot'}), row=1, col=1)
    # 3
    for i, result in enumerate(all_results):
        x_steps = np.asarray(list(result.keys()))
        y_values = np.array(list(result.values()))
        y_diago = x_steps-2
        standerized_x = (x_steps-2) / np.max(x_steps-2) * 100
        interpolate_x = np.linspace(x_steps-2, standerized_x, 100)
        list_interpol.append(interpolate_x)
        hover = hover_template(x_steps, y_values)
        my_rnx = ((x_steps-2) - y_values)/(x_steps-2)
        my_rnx[0] = 1
        plt.add_trace(go.Scatter(x=standerized_x, y=y_diago - y_values, mode='lines',
                                 name=names[i] + "values", hovertemplate=hover, line={"color": colors[i]}), row=2, col=1)
    # 4
    for i, result in enumerate(all_results):
        x_steps = np.asarray(list(result.keys()))
        y_values = np.array(list(result.values()))
        y_diago = x_steps-2
        standerized_x = (x_steps-2) / np.max(x_steps-2) * 100
        interpolate_x = np.linspace(x_steps-2, standerized_x, 100)
        list_interpol.append(interpolate_x)
        hover = hover_template(x_steps, y_values)
        my_rnx = ((x_steps-2) - y_values)/(x_steps-2)
        my_rnx[0] = 1
        plt.add_trace(go.Scatter(x=x_steps, y=(y_diago - y_values)/(x_steps-2), mode='lines',
                                 name=names[i] + "esperance", hovertemplate=hover, line={"color": colors[i]}), row=3, col=1)
    # 5
    for i, result in enumerate(all_results):
        x_steps = np.asarray(list(result.keys()))
        y_values = np.array(list(result.values()))
        # y_diago = x_steps-2
        # standerized_x = (x_steps-2) / np.max(x_steps-2) * 100
        # interpolate_x = np.linspace(x_steps-2, standerized_x, 100)
        # list_interpol.append(interpolate_x)
        my_rnx = ((x_steps-2) - y_values)/(x_steps-2)

        y_diago = (x_steps-2)/(x_steps)
        standerized_x = (x_steps-2) / np.max(x_steps-2) * 100
        my_rnx = 1-(y_values / y_diago)
        my_rnx[0] = 1
        hover = hover_template(x_steps, y_values)
        plt.add_trace(go.Scatter(x=standerized_x, y=my_rnx, mode='lines',
                                 name=names[i] + "my_rnx", hovertemplate=hover, line={"color": colors[i]}), row=4, col=1)
    # plt.add_trace(go.Scatter(x=x, y=y_diago - y_values, mode='lines',
    #                          name=names[i]+"esperance", line={"color": colors[i]}), row=3, col=1)
    # plt.add_trace(go.Scatter(x=x, y=y_, mode='lines',
    #                          name=names[i]+"values", line={"color": colors[i]}), row=3, col=1)
    plt.update_yaxes(title_text="average levenstein", row=1, col=1)
    plt.update_xaxes(title_text="len of path", row=1, col=1)
    plt.update_yaxes(
        title_text="average levenstein - esperance", row=2, col=1)
    plt.update_xaxes(title_text="standerized len of path", row=2, col=1)
    plt.update_yaxes(
        title_text="(average levenstein - esperance)/(len of path)", row=3, col=1)
    plt.update_xaxes(
        title_text="len of path", row=3, col=1)
    plt.update_yaxes(
        title_text="(average levenstein - esperance)/(len of path)", row=4, col=1)
    plt.update_xaxes(
        title_text="standerized len of path", row=4, col=1)
# plt.add_trace(go.Scatter(x=np.arange(0, max_recorded_x), y=np.arange(-2, max_recorded_x-2), mode='lines',
#                          name="diagonal", hovertemplate="y = x-2", line={"color": "black"}), row=1, col=1)

    plt.update_layout(height=4000)  # Specify the height in pixels
    # Create frames
    print(len(list_interpol), len(list_interpol[0]), len(list_interpol[0][0]))
    frames = []
    for j in range(100):
        data_frame = []

        for i in range(len(all_results)):
            y_values = np.array(list(all_results[i].values()))
            y_diago = np.array(list(all_results[i].keys())) - 2
            data_frame.append(go.Scatter(x=list_interpol[i][j], y=y_values, mode='lines', name="data", line={
                "color": colors[i]}))
            data_frame.append(go.Scatter(x=list_interpol[i][j], y=y_diago, mode='lines', name="esperance", line={
                "color": colors[i], "dash": "dot"}))
        for i in range(len(all_results)):
            data_frame.append(go.Scatter(visible=True))
        for i in range(len(all_results)):
            data_frame.append(go.Scatter(visible=True))
        for i in range(len(all_results)):
            data_frame.append(go.Scatter(visible=True))
        # print(data_frame)
        # print(np.arange(len(all_results)))
        # print(0/0)
        frames.append(
            go.Frame(data=data_frame, traces=np.arange(5*len(all_results))))
    # Associate frames with figure
    button = dict(
        label='Play',
        method='animate',
        args=[None, dict(frame=dict(duration=35, redraw=False),
                         transition=dict(duration=0),
                         fromcurrent=True,
                         mode='immediate')])
    button2 = dict(
        label='Reverse',
        method='animate',
        args=[None, dict(frame=dict(duration=35, redraw=False),
                         transition=dict(duration=0),
                         fromcurrent=True,
                         mode='immediate')])
    updatemenus = [dict(type='buttons', showactive=False,
                        buttons=[button, button2])]

    plt.update_layout(updatemenus=updatemenus)
    plt.frames = frames
    go.Figure(
        data=plt.data,
        frames=[
            fr.update(
                layout={
                    "xaxis": {"range": [min(fr.data[1*len(all_results)+3].x) - 0.1, max(fr.data[1*len(all_results)+3].x) + 0.1]},
                }
            )
            for fr in plt.frames
        ],
        layout=plt.layout,
    )
    plt.show()
