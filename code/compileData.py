from Utils import *
import sys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="display dimension reduction")
    parser.add_argument("outputfile", help="name of the output file")
    args = parser.parse_args()
    name = LD_DATA_NAME
    # save data to a file
    outputfile = open(args.outputfile+".txt", "a")
    LD_data_reshaped = (LD_data - np.min(LD_data)) / \
        (np.max(LD_data) - np.min(LD_data)) * 100
    _, _, edges = alpha_shape(LD_data_reshaped, alpha=0.3)
    distance_matrix_LD = compute_distance_matrix(LD_data)
    graph = distance_matrix_LD.copy()
    graph = graph**2 + np.max(graph) * 100
    for i in range(len(graph)):
        graph[i][i] = 0  # remove self loop
    for edge in edges:
        graph[edge[0]][edge[1]] = distance_matrix_LD[edge[0]][edge[1]]**2
        graph[edge[1]][edge[0]] = distance_matrix_LD[edge[1]][edge[0]]**2
    # results, localisation_of_errors,_ = random_selection_path_2(
    #     HD_data, LD_data, 200, graph)
    # average = {}
    # for key in results:
    #     average[key] = sum(results[key]) / (len(results[key]))
    # average = dict(sorted(average.items()))
    # outputfile.write(name + " random \n")
    # outputfile.write(str(average))
    # outputfile.write("\n")
    xss, yss, index_enveloppes = enveloppe_of_cluster(LD_data)
    index_enveloppe = [
        item for sublist in index_enveloppes for item in sublist]

    results, localisation_of_errors, _ = ALL_path_enveloppe(
        HD_data, LD_data, graph, index_enveloppe)
    average = {}
    for key in results:
        average[key] = sum(results[key]) / (len(results[key]))
    average = dict(sorted(average.items()))
    outputfile.write(name + " ALL \n")
    outputfile.write(str(average))
    outputfile.write("\n")
    outputfile.close()
