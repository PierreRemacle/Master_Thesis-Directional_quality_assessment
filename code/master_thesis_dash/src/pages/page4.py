
import threading
from dash import ctx
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import io
import base64

import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE, Isomap, MDS
import umap
from minisom import MiniSom  # SOM implementation
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
import time
import pickle

from Utils import *


def path_rnx_distance_sorted(LD_data, HD_data, LD_paths, HD_paths, folder_name):
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
    file_name = folder_name + "/results_color.pkl"
    with open(file_name, "wb") as fp:
        pickle.dump(results, fp)


def apply_reduction_methods(X, y, method_name):
    # Standardize features
    X = StandardScaler().fit_transform(X)

    methods = {
        "PCA": PCA(n_components=2),
        "t-SNE": TSNE(n_components=2, random_state=42, max_iter=300),
        "UMAP": umap.UMAP(n_components=2),
        "Isomap": Isomap(n_components=2),
        "MDS": MDS(n_components=2),
    }

    method = methods[method_name]
    name = method_name
    print(f"Applying {name}...")
    if name == "SOM":
        som = MiniSom(10, 10, X.shape[1], sigma=0.5, learning_rate=0.5)
        som.train_random(X, 100)
        X_reduced = np.array([som.winner(x) for x in X]).reshape(-1, 2)
    # LDA needs labels and >=2 classes
    elif name == "LDA" and len(np.unique(y)) > 1:
        X_reduced = method.fit_transform(X, y)
    else:
        X_reduced = method.fit_transform(X)

    return X_reduced


def compute_new_rnx(X, X_reduced, embedding_name):

    _, _, edges = alpha_shape(X_reduced, alpha=0.00001)

    ALL_path_3(
        X, X_reduced, edges, embedding_name)
    X_reduced[:, 0] = X_reduced[:, 0] * 1.5
    # X_reduced_reshaped = (X_reduced - np.min(X_reduced)) / \
    #     (np.max(X_reduced) - np.min(X_reduced)) * 100

    _, _, edges = alpha_shape(X_reduced, alpha=0.00001)

    ALL_path_3(
        X, X_reduced, edges, embedding_name + "_distorded_x1.5")


# Page 4 Layout
page4_layout = html.Div(
    [

        dbc.NavbarSimple(
            children=[
                # Navbar Items
                dbc.NavItem(dbc.NavLink("Home", href="/")),
                dbc.NavItem(dbc.NavLink("Page 1", href="/page1")),
                dbc.NavItem(dbc.NavLink("Page 2", href="/page2")),
                dbc.NavItem(dbc.NavLink("Page 3", href="/page3")),
                dbc.NavItem(dbc.NavLink("Page 4", href="/page4")),
                dbc.NavItem(dbc.NavLink("Page 5", href="/page5")),
                dbc.NavItem(dbc.NavLink("Page 6", href="/page6")),
            ],
        ),
        # Header
        html.H3("Upload and Process Dataset", style={
                "text-align": "center", "margin-bottom": "1rem"}),

        # Drag-and-Drop Upload Component
        dcc.Upload(
            id="upload-data",
            children=html.Div(
                [
                    "Drag and Drop or ",
                    html.A("Select a File"),
                ],
                style={
                    "borderWidth": "2px",
                    "borderStyle": "dashed",
                    "borderRadius": "5px",
                    "textAlign": "center",
                    "padding": "40px",
                    "cursor": "pointer",
                },
            ),
            multiple=False,  # Only accept one file at a time
            style={"margin-bottom": "1rem"}
        ),
        dcc.Interval(id="progress-interval", n_intervals=0,
                     interval=500),
        dbc.Progress(id="progress", style={"padding": "2rem"}),

        # Output message and table preview
        html.Div(id="upload-message",
                 style={"margin-top": "1rem", "padding": "2rem"}),
        html.Div(id="data-preview", style={"padding": "2rem"}),

        # display the time taken for each file
        html.Div(id="time-taken"),
    ],
)

# Callback to process uploaded file


# Shared variable for progress
progress_value = 0
lock = threading.Lock()


def register_callbacks_page4(app):

    # load time taken file and display it
    with open("time_taken.txt", "r") as f:
        time_taken = f.read()

    @ app.callback(
        Output("time-taken", "children"),
        [Input("upload-data", "contents")],
    )
    def display_time_taken(contents):
        return html.Div(
            [
                html.H4("Time taken for each file:"),
                html.Pre(time_taken),
            ],
            style={"margin-top": "1rem"}
        )

    @ app.callback(
        [Output("upload-message", "children"),
            Output("data-preview", "children"),
            Output("progress", "value", allow_duplicate=True),
            Output("progress", "label", allow_duplicate=True)],
        [Input("upload-data", "contents")],
        [State("upload-data", "filename"),
         State("upload-data", "last_modified")], prevent_initial_call=True,
    )
    def handle_file_upload(contents, filename, last_modified):
        global progress_value
        progress_value = 0  # Reset progress

        if contents is None:
            return "", "", 0, "0%"

        contents = contents.split(",")[1]
        contents = io.StringIO(base64.b64decode(contents).decode("utf-8"))

        try:
            start = time.time()
            df = pd.read_csv(contents)
            message = dbc.Alert(f"File {filename} uploaded successfully!",
                                color="success", dismissable=True)
            preview = dbc.Table.from_dataframe(
                df.head(), striped=True, bordered=True, hover=True, style={"margin-top": "1rem"}
            )

            methods = ["PCA", "t-SNE", "UMAP", "Isomap", "MDS"]
            total_methods = len(methods) * 3

            for i, method_name in enumerate(methods):
                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1].values
                X_reduced = apply_reduction_methods(X, y, method_name)

                with lock:
                    progress_value = int(((i*3 + 1) / total_methods) * 100)
                folder_name = f"../{method_name}_{filename.split('.')[0]}"
                compute_new_rnx(X, X_reduced, folder_name)

                with lock:
                    progress_value = int(((i*3 + 2) / total_methods) * 100)
                folder_name = folder_name + "_data"
                # Simulate heavy processing
                LD_data = np.load(f"{folder_name}/LD_data.npy")
                HD_data = np.load(f"{folder_name}/HD_data.npy")
                LD_paths = np.load(f"{folder_name}/LD_all_paths_2.npy")
                HD_paths = np.load(f"{folder_name}/HD_all_paths_2.npy")
                path_rnx_distance_sorted(
                    LD_data, HD_data, LD_paths, HD_paths, folder_name)

                # Update progress
                with lock:
                    progress_value = int(((i*3 + 3) / total_methods) * 100)
            end = time.time()
            print(f"Time taken: {end - start}")
            # happen to a file with all the time taken
            with open("time_taken.txt", "a") as f:
                f.write(f"{filename}: {end - start}\n")
            return message, preview, 100, "100%"

        except Exception as e:
            return dbc.Alert(f"Error processing file: {e}", color="danger", dismissable=True), "", 0, "0%"

    @ app.callback(
        [Output("progress", "value"), Output("progress", "label")],
        [Input("upload-data", "contents")],
    )
    def update_progress(contents):
        global progress_value
        with lock:
            value = progress_value
        return value, f"{value}%"
