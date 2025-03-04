import pickle
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def path_rnx_distance_sorted(LD_data, HD_data, LD_paths, HD_paths):
    results = {}

    for index in range(len(LD_data)):
        results[index] = {}
        for i in range(len(LD_paths[index])):
            LD_path = [x for x in LD_paths[index][i] if x != -1][1:]
            HD_path = [x for x in HD_paths[index][i] if x != -1][1:]
            distance = levenshteinDistanceDP(LD_path, HD_path)
            path_len = len(LD_path)

            if path_len not in results[index]:
                results[index][path_len] = [distance]
            else:
                results[index][path_len].append(distance)

        results[index] = {k: np.mean(v)
                          for k, v in sorted(results[index].items())}

    with open("results_color.pkl", "wb") as fp:
        pickle.dump(results, fp)


def paths_quality_from_one_start(start_index, LD_data, HD_data, LD_paths, HD_paths):
    results = np.zeros(len(LD_paths[start_index]))

    for i in range(len(LD_paths[start_index])):
        LD_path = [x for x in LD_paths[start_index][i] if x != -1][1:]
        HD_path = [x for x in HD_paths[start_index][i] if x != -1][1:]
        distance = levenshteinDistanceDP(LD_path, HD_path)
        results[i] = distance / len(LD_path)

    return results, LD_paths[start_index]


def rescale(LD_data, HD_data, LD_paths, HD_paths):
    longest_path_len = 0
    ys = np.zeros([len(LD_data), 500])

    with open("results_color.pkl", "rb") as fp:
        results = pickle.load(fp)

    for index, data in results.items():
        longest_path_len = max(longest_path_len, max(data.keys()))

    means = np.zeros(longest_path_len)
    colors = np.zeros(len(LD_data))
    frames = []

    for index in range(len(LD_data)):
        data = results[index]
        data = {k: v for k, v in sorted(
            data.items(), key=lambda item: item[0])}
        keys = list(data.keys())
        for i in range(len(keys)-1, -1, -1):
            data[keys[i]+2] = data[keys[i]]
        del data[1]
        del data[0]
        data = {k: v for k, v in sorted(
            data.items(), key=lambda item: item[0])}
        y = np.array(list(data.values()))/list(data.keys()) - \
            ((np.array(list(data.keys())) - 1.7)/(np.array(list(data.keys()))))

        y_test = - ((np.arange(longest_path_len + 5) - 1.7) /
                    (np.arange(longest_path_len + 5)))
        ys[index] = np.pad(-y, (0, 500 - len(y)), "constant")

        if index % 10 == 0 or index == len(LD_data)-1:
            ys_transpose = ys[:index].T
            means = np.zeros(longest_path_len)

            for i in range(longest_path_len):
                tempo = np.trim_zeros(np.sort(ys_transpose[i]))
                means[i] = np.mean(tempo)

            for i in range(index+1):
                colors[i] = np.mean(ys[i][:len(means)] - means)

            scatter_colored_points = go.Scatter(
                x=LD_data[:, 0],
                y=LD_data[:, 1],
                mode="markers",
                marker=dict(
                    size=5,
                    color=colors,
                    colorscale="Viridis",
                    showscale=True,
                ),
                name="Colored Points",
            )

            rescaled_line_plot = go.Scatter(
                x=np.arange(2, len(means) + 2),
                y=means,
                mode="lines",
                line=dict(color="blue"),
                name="Rescaled Means",
            )

            frame = go.Frame(
                data=[
                    scatter_colored_points,
                    rescaled_line_plot,
                ],
                name=f"frame{index}",
            )
            frames.append(frame)

    # Build Figure with Subplots
    fig = make_subplots(
        rows=2, cols=1,  # 2 rows, 1 column
        # shared_xaxes=True,  # Optionally share x-axis
        vertical_spacing=0.1,  # Space between plots
        subplot_titles=("2D Scatter Plot", "Rescaled Means Plot"),
    )

    # Add 2D Scatter Plot to the first subplot
    fig.add_trace(
        go.Scatter(x=LD_data[:, 0], y=LD_data[:, 1],
                   mode="markers", marker=dict(size=5)),
        row=1, col=1
    )

    # Add Rescaled Line Plot to the second subplot
    fig.add_trace(
        go.Scatter(x=np.arange(2, len(means) + 2), y=means, mode="lines"),
        row=2, col=1
    )
    for i in range(0, 13):
        y_values = i / 10 - ((np.arange(longest_path_len + 5) - 1.7) /
                             (np.arange(longest_path_len + 5)))
        fig.add_trace(
            go.Scatter(
                x=np.arange(longest_path_len + 5),
                y=-y_values,
                mode="lines",
                line=dict(dash="dash", color="grey", width=1),
                name=f"Guide Line {i}",
            ),
            row=2, col=1
        )
    # Update layout with sliders, play/pause buttons, and titles
    fig.update_layout(
        title="Dynamic Rescaling",
        xaxis=dict(title="LD X"),
        yaxis=dict(title="LD Y"),
        yaxis2=dict(
            title="Rescaled Means",
            # Set y-axis range for the second subplot (Rescaled Means)
            range=[0, 1]
        ),
        sliders=[dict(
            steps=[dict(
                method='animate',
                args=[[f'frame{k*10}'],
                      dict(mode='immediate', frame=dict(duration=0, redraw=False),
                           transition=dict(duration=0))],
                label=f'{k+1}'
            ) for k in range(len(frames))],
            active=0,
            transition=dict(duration=0),
            currentvalue=dict(font=dict(size=12),
                              prefix='Frame: ',
                              visible=True,
                              xanchor='center'),
            len=1.0,
            x=0,  # slider starting position
            y=0,
        )],
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[None, dict(
                        frame=dict(duration=100, redraw=True),
                        fromcurrent=True,
                        mode="immediate",
                        transition=dict(duration=0)
                    )],
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[[None], dict(
                        frame=dict(duration=0, redraw=False),
                        mode="immediate",
                        transition=dict(duration=0)
                    )],
                )
            ],
            x=0.1,
            y=-0.1,
            xanchor="center",
            yanchor="top"
        )]
    )

    # Assign frames to the figure object directly
    fig.frames = frames

    fig.show()


# Load data
folder = "t-SNE_MNIST_data"
LD_data = np.load(f"./{folder}/LD_data.npy")
HD_data = np.load(f"./{folder}/HD_data.npy")
LD_paths = np.load(f"./{folder}/LD_all_paths_2.npy")
HD_paths = np.load(f"./{folder}/HD_all_paths_2.npy")

# Process and visualize
# path_rnx_distance_sorted(LD_data, HD_data, LD_paths, HD_paths)
rescale(LD_data, HD_data, LD_paths, HD_paths)
