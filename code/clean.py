
def compute_new_rnx(X, X_reduced, embedding_name):

    X_reduced_reshaped = (X_reduced - np.min(X_reduced)) / \
        (np.max(X_reduced) - np.min(X_reduced)) * 100
    _, _, edges = alpha_shape(X_reduced_reshaped, alpha=0.00001)

    # Initialize results
    distance_matrix = compute_distance_matrix(HD_data)
    LD_distance_matrix = compute_distance_matrix(LD_data)
    edges = graph
    graph = np.zeros((len(LD_data), len(LD_data)))
    # graph = graph**2 + np.max(graph) * 100
    for i in range(len(graph)):
        graph[i][i] = 0  # remove self loop
    for edge in edges:
        graph[edge[0]][edge[1]] = LD_distance_matrix[edge[0]][edge[1]]
        graph[edge[1]][edge[0]] = LD_distance_matrix[edge[1]][edge[0]]
    LD_all_paths_2 = -np.ones((len(LD_data), len(LD_data), 200))
    HD_all_paths_2 = -np.ones((len(LD_data), len(LD_data), 200))
    # Progress bar
    with alive_bar(len(HD_data)) as bar:
        for i in range(len(HD_data)):

            LD_paths, HD_paths = LDHDPathAll(
                graph, distance_matrix, i, LD_data, HD_data)
            for j in range(i + 1, len(LD_paths)):
                LD_path_2 = create_HD_path(
                    LD_paths[j], LD_distance_matrix, i, j)
                HD_path_2 = create_HD_path(
                    LD_paths[j], distance_matrix, i, j)
                # reshape by adding -1 to get same lengths as LD_path
                LD_path_2 = np.append(
                    LD_path_2, [-1] * (200 - len(LD_path_2)))
                HD_path_2 = np.append(
                    HD_path_2, [-1] * (200 - len(HD_path_2)))
                LD_all_paths_2[i][j] = LD_path_2
                LD_all_paths_2[j][i] = LD_path_2
                HD_all_paths_2[i][j] = HD_path_2
                HD_all_paths_2[j][i] = HD_path_2

            bar()

    results = {}
    LD_paths = LD_all_paths_2
    HD_paths = HD_all_paths_2
    for index in range(len(LD_data)):
        for i in range(index, len(LD_paths[index])):
            LD_path = LD_paths[index][i]
            HD_path = HD_paths[index][i]
            # remove all -1
            LD_path = [x for x in LD_path if x != -1]
            HD_path = [x for x in HD_path if x != -1]
            # remove first element
            LD_path = LD_path[1:]
            HD_path = HD_path[1:]

            for j in range(1, len(LD_path)):
                # distance = intersection of the jth elements of the two paths
                distance = len(set(LD_path[:j]).intersection(set(HD_path[:j])))
                if j not in results:
                    results[j] = [distance/j]
                else:
                    results[j].append(distance/j)

    results = dict(sorted(results.items()))
    for key in results:
        tempo = np.mean(list(results[key]))

        maximum = max(list(results.keys()))
        results[key] = ((maximum + 1) * tempo - key) / \
            (maximum + 1 - key)
    print(results)
