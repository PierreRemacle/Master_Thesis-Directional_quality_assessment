
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from Utils import *
import random

# display embedding 2D

data_mnist_tsne = {1: 0.459966428572065, 2: 0.5277238570152577, 3: 0.5692405748546728, 4: 0.6015101206131168, 5: 0.6227283612820825, 6: 0.6390266090445793, 7: 0.652520791697302, 8: 0.6633790473017268, 9: 0.6733689519193145, 10: 0.6819548052490997, 11: 0.6903901768242217, 12: 0.6979062844304944, 13: 0.7052227519076996, 14: 0.7123377626684257, 15: 0.7198144790803573, 16: 0.7271675225678225, 17: 0.7344122274448169, 18: 0.741499712577518, 19: 0.748702747476414, 20: 0.7557368567301569, 21: 0.7627795591125618, 22: 0.7700045596231506, 23: 0.7771679905298526, 24: 0.7842382563466873, 25: 0.791227943959605, 26: 0.798015512216593, 27: 0.8046827424410816, 28: 0.8113920822663634, 29: 0.8178231194768601, 30: 0.824075818618329, 31: 0.8301827920620726, 32: 0.8361961844425932, 33: 0.8421572140276518, 34: 0.8479879061301394, 35: 0.8537720787273755,
                   36: 0.8593695961448975, 37: 0.8651130173952533, 38: 0.8706202275871134, 39: 0.8760994228606198, 40: 0.8815183662585924, 41: 0.8867823942984676, 42: 0.891852321451269, 43: 0.8968795802542968, 44: 0.9017328319450221, 45: 0.9066003714349286, 46: 0.9112829983005286, 47: 0.9157467519916181, 48: 0.9202075514429923, 49: 0.9243749217478403, 50: 0.928396399481447, 51: 0.9322346232292501, 52: 0.936227646712977, 53: 0.9401350443238256, 54: 0.9439324552578198, 55: 0.947636922812976, 56: 0.9510202007491797, 57: 0.9542619815210202, 58: 0.9571986639924136, 59: 0.9599066920771542, 60: 0.9622568093385212, 61: 0.9644821916942001, 62: 0.9667746726878527, 63: 0.9692881336635073, 64: 0.9718067200203252, 65: 0.9744354434009606, 66: 0.9767388167388167, 67: 0.9793080054274083, 68: 0.982811306340718, 69: 0.9855072463768119}

data_mnist_PCA = {1: 0.15023505864103898, 2: 0.23441634773833742, 3: 0.3041942954605061, 4: 0.3614470162268514, 5: 0.40804743952714556, 6: 0.4469046079140103, 7: 0.4817840398329682, 8: 0.5128071389492921, 9: 0.540898124860746, 10: 0.5662091331531405, 11: 0.5891608702522891, 12: 0.6107153060751134, 13: 0.6306522083406435, 14: 0.6493108901398793, 15: 0.6665707636596861, 16: 0.682727054690821, 17: 0.6982214133724154, 18: 0.7126890965651871, 19: 0.7264249236661777, 20: 0.7391357140842367, 21: 0.7513940222465075, 22: 0.7628241841313027, 23: 0.773632849511805, 24: 0.7839404775177886, 25: 0.7938243862328208, 26: 0.8033923327617327, 27: 0.8123211046967492, 28: 0.8208968267148883, 29: 0.8291622938784173, 30: 0.8369471539084079, 31: 0.8443841298867681, 32: 0.8514294070737521, 33: 0.8581120686563848, 34: 0.8645309746618158, 35: 0.8707442544874874, 36: 0.8766584598739915, 37: 0.8822813668736151, 38: 0.8876679608065204,
                  39: 0.8926450449827682, 40: 0.8974413881792949, 41: 0.9020301192140776, 42: 0.9065399886914033, 43: 0.910844769754373, 44: 0.9149669454313106, 45: 0.9189210753015591, 46: 0.9225650835819709, 47: 0.926277759263972, 48: 0.9297514989267738, 49: 0.9331669765995483, 50: 0.9363998599616701, 51: 0.9393688006006355, 52: 0.9423072273920838, 53: 0.9449369683386342, 54: 0.9475864657312778, 55: 0.950120923762705, 56: 0.9525140777551212, 57: 0.9547834750225814, 58: 0.9569055367256546, 59: 0.9588335394871157, 60: 0.96093864537296, 61: 0.9629061796709312, 62: 0.9648491742744214, 63: 0.9669044032045778, 64: 0.9688286151888245, 65: 0.9708324888684171, 66: 0.9726912016672843, 67: 0.9745023898881593, 68: 0.9765950069348127, 69: 0.9785772715346269, 70: 0.980864401187727, 71: 0.9826314671951305, 72: 0.9850793650793648, 73: 0.9874694701072528, 74: 0.9896049896049895, 75: 0.9908148148148148, 76: 0.9903508771929824, 77: 0.9935064935064934}
y_coil_tsne = [0.15,      0.43333333, 0.31771435, 0.3443868, 0.29367676, 0.29259676, 0.26699653, 0.25849631, 0.24179553, 0.2331773, 0.22094118, 0.21103286, 0.20376917, 0.1975581, 0.19243447, 0.18864097, 0.18364787, 0.17856471, 0.17346407, 0.16899189, 0.1648748, 0.15908099, 0.15370576, 0.14799138, 0.1452994, 0.14243192, 0.14028726, 0.13935326, 0.13873136, 0.13885236, 0.13711642, 0.13461037, 0.12991065, 0.12546065, 0.12016168, 0.11524895,
               0.11122923, 0.10751325, 0.10521186, 0.10292702, 0.09862998, 0.09478882, 0.09207169, 0.0877909, 0.08428736, 0.08319986, 0.07797309, 0.07414966, 0.06928767, 0.06662765, 0.06324587, 0.05826388, 0.05739023, 0.06359408, 0.06935714, 0.07201049, 0.07775767, 0.08088512, 0.08355072, 0.08639831, 0.08548387, 0.0870801, 0.09378324, 0.1, 0.11173436, 0.1283893, 0.14562594, 0.15188881, 0.16873016, 0.15915493, 0.14479167, 0.11369863, 0.12567568]

y_mnist_tsne = [0.15, 0.43333333, 0.389039, 0.39039, 0.36840863, 0.35568374,
                0.3387861, 0.32468347, 0.31169517, 0.29852858, 0.28747479, 0.27669732,
                0.26761103, 0.25794555, 0.24865544, 0.24011903, 0.2327701, 0.2250106,
                0.21817128, 0.21169148, 0.20589381, 0.20031665, 0.19481915, 0.18889262,
                0.1840891, 0.17890562, 0.17424635, 0.16971082, 0.16539682, 0.1607031,
                0.15695915, 0.15383958, 0.15022478, 0.14707452, 0.14428859, 0.14151793,
                0.13897113, 0.13636465, 0.13354246, 0.13078402, 0.12869261, 0.12652728,
                0.12462295, 0.12244712, 0.12082277, 0.11908762, 0.11704609, 0.11504043,
                0.11370177, 0.11272263, 0.11194565, 0.11066841, 0.11004698, 0.10938827,
                0.10896719, 0.10896338, 0.1092025, 0.10958521, 0.10954322, 0.10944245,
                0.11030354, 0.11159437, 0.11253759, 0.11365762, 0.11140872, 0.11050613,
                0.10303242, 0.09296745, 0.08430569, 0.08046054, 0.05575397]
y_mnist_PCA = [0.15, 0.43333333, 0.33248879, 0.33048071, 0.28647787, 0.27198875,
               0.24616451, 0.22879553, 0.21253798, 0.1989166, 0.18790461, 0.17790996,
               0.1693857, 0.16216722, 0.15511217, 0.14900325, 0.14395659, 0.13888633,
               0.1347963, 0.13090547, 0.12744219, 0.12440329, 0.12162131, 0.11883916,
               0.11647684, 0.11465048, 0.11249871, 0.11068439, 0.10929621, 0.1078339,
               0.10671739, 0.1054545, 0.10456878, 0.10389813, 0.1037336, 0.10281506,
               0.10239813, 0.10171588, 0.1016505, 0.10111291, 0.10125464, 0.10129177,
               0.1012817, 0.10122922, 0.10143067, 0.10153956, 0.10170605, 0.10231811,
               0.10275003, 0.1031672, 0.10394677, 0.10415879, 0.10485045, 0.10556437,
               0.10652954, 0.10785068, 0.10863399, 0.11021352, 0.11204012, 0.11256947,
               0.11463777, 0.11593533, 0.11681453, 0.11869285, 0.12072004, 0.12251412,
               0.12616982, 0.12659897, 0.12641072, 0.12799914, 0.12642729, 0.12840096,
               0.13525014, 0.1424769, 0.14821762, 0.15018038, 0.15213675, 0.17295742,
               0.26]
y_coil_MDS = [0.15,      0.43333333, 0.32603594, 0.34418505, 0.28951246, 0.28302884,
              0.25354507, 0.24254724, 0.22325846, 0.21273622, 0.20011821, 0.19353146,
              0.18408458, 0.17909764, 0.17322213, 0.16953699, 0.16560197, 0.16336708,
              0.16032678, 0.15899254, 0.1560951, 0.15402099, 0.15120778, 0.14912989,
              0.14616038, 0.14349029, 0.14074072, 0.13922115, 0.13787996, 0.13666193,
              0.13546437, 0.13507785, 0.13549553, 0.1342552, 0.1355402, 0.13462815,
              0.13438941, 0.13370678, 0.13392828, 0.13398716, 0.13512698, 0.13694775,
              0.14075795, 0.14371838, 0.15122401, 0.15460392, 0.16672674, 0.17279085,
              0.18256942, 0.18965219, 0.19153497, 0.19104223, 0.19229841, 0.19691288,
              0.20531015, 0.21233292, 0.22060983, 0.21466754, 0.19666667, 0.15245902]

data_coil_tsne = {1: 0.19921575164564226, 2: 0.27662303910457325, 3: 0.3343560789587308, 4: 0.3768834493473521, 5: 0.4173588065372669, 6: 0.45331800865746497, 7: 0.4853081314772449, 8: 0.5162355070220744, 9: 0.5440746836210336, 10: 0.5699842272225145, 11: 0.593443731819258, 12: 0.6152388402196041, 13: 0.635461762242549, 14: 0.6541181318102941, 15: 0.6714437731636262, 16: 0.6868732582352205, 17: 0.7007651346836606, 18: 0.7126669981393674, 19: 0.7231155257660641, 20: 0.7329419856937543, 21: 0.7422456661679259, 22: 0.7510863478040137, 23: 0.7596081746319265, 24: 0.768539892446185, 25: 0.7775074622675936, 26: 0.7868322238096233, 27: 0.795933171573308, 28: 0.8050650323058284, 29: 0.8138078296422101, 30: 0.8218740169381018, 31: 0.8293446376567933, 32: 0.8366107061410654, 33: 0.8437274308365962, 34: 0.8508686671055552, 35: 0.8581279764687376,
                  36: 0.8651183295728739, 37: 0.8715961610970959, 38: 0.8773963128415609, 39: 0.8831001286885785, 40: 0.8889006957621757, 41: 0.8943963091732123, 42: 0.8997252921834461, 43: 0.9048146914016061, 44: 0.9092985749269125, 45: 0.9131538037209769, 46: 0.9165963878057131, 47: 0.9200784506006374, 48: 0.9237693828702257, 49: 0.9273365912021373, 50: 0.9303091228952, 51: 0.9318173275700058, 52: 0.9327194150996003, 53: 0.9313004191806314, 54: 0.9276591289782246, 55: 0.9240762243016186, 56: 0.9217391304347826, 57: 0.9217676397753828, 58: 0.925442161074345, 59: 0.9306801115640421, 60: 0.9367906533142584, 61: 0.9444281015163265, 62: 0.9512710547579092, 63: 0.9575097334531297, 64: 0.9633049242424242, 65: 0.9674556213017752, 66: 0.9705710955710953, 67: 0.973448546739984, 68: 0.9761764705882353, 69: 0.979836168872086, 70: 0.9816326530612246, 71: 0.9859154929577465}


def path_rnx_distance_sorted(LD_data, HD_data, LD_paths, HD_paths):

    results = {}

    for index in range(len(LD_data)):

        results[index] = {}

        for i in range(len(LD_paths[index])):

            LD_path = LD_paths[index][i]
            HD_path = HD_paths[index][i]
            # remove all -1
            LD_path = [x for x in LD_path if x != -1]
            HD_path = [x for x in HD_path if x != -1]
            # remove first element
            LD_path = LD_path[1:]
            HD_path = HD_path[1:]

            # distance = intersection of the jth elements of the two paths
            distance = levenshteinDistanceDP(LD_path, HD_path)

            if len(LD_path) not in results[index]:
                results[index][len(LD_path)] = [distance]
            else:
                results[index][len(LD_path)].append(distance)
        # sort the keys
        results[index] = dict(sorted(results[index].items()))
        results[index] = {k: np.mean(v) for k, v in results[index].items()}

    with open('results_color.pkl', 'wb') as fp:
        pickle.dump(results, fp)


def rescale(LD_data, HD_data, LD_paths, HD_paths):

    longest_path_len = 0
    ys = np.zeros([len(LD_data), 500])
    results_2 = [0] * len(LD_data)
    all_index_enveloppe = range(len(LD_data))
    # len of the longest path
    with open('results_color.pkl', 'rb') as fp:
        results = pickle.load(fp)

    for index in all_index_enveloppe:
        data = results[index]
        keys = list(data.keys())
        max_key = max(keys)
        if max_key > longest_path_len:
            longest_path_len = max_key

    for index in all_index_enveloppe:
        # print(index)

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
        length = len(results[index])

        y = np.array(list(data.values()))/list(data.keys()) - \
            ((np.array(list(data.keys())) - 1.7)/(np.array(list(data.keys()))))
        # y sould have a length of 500
        ys[index] = np.pad(-y, (0, 500-len(y)), 'constant')
        # plt.plot(list(data.keys()), -y, alpha=0.05)
    ys = ys.T
    means = np.zeros(longest_path_len)
    for i in range(longest_path_len):

        tempo = np.trim_zeros(np.sort(ys[i]))
        means[i] = np.mean(tempo)

    plt.plot(np.arange(2, len(means)+2),
             means)

    # color each point according to the value in results_2

    plt.scatter(LD_data[:, 0], LD_data[:, 1], c=np.log(results_2))
    plt.legend()

    # plt.show()
    # path_rnx_distance_sorted(
    #     LD_data, HD_data, LD_paths_2, HD_paths_2)
    return means


def rescale_multiple_alpha(LD_data, HD_data, LD_paths, HD_paths, alphas, color, true_y):

    fig, axs = plt.subplots(1, 2)
    with open('results_color.pkl', 'rb') as fp:
        results = pickle.load(fp)

    LD_data = np.array(LD_data)
    min = np.min(LD_data, axis=0)
    max = np.max(LD_data, axis=0)
    LD_data_rescale = LD_data - np.min(LD_data, axis=0)
    LD_data_rescale = LD_data_rescale / np.max(LD_data_rescale, axis=0) * 100
    already_colored = []

    axs[1].scatter(LD_data[:, 0], LD_data[:, 1])
    for alpha in alphas:

        ys = np.zeros([len(LD_data), 500])
        results_2 = [0] * len(LD_data)
        # enveloppe, all_index_enveloppe = multiple_enveloppe(LD_data, 15)
        concave_hull, edge_points, _ = alpha_shape(
            LD_data_rescale, alpha=alpha)
        all_index_enveloppe = []
        for i in range(len(concave_hull.geoms)):
            index_enveloppe = []
            for j in range(len(concave_hull.geoms[i].exterior.coords)):
                index_enveloppe.append(
                    np.where(LD_data_rescale == concave_hull.geoms[i].exterior.coords[j])[0][0])
            all_index_enveloppe.append(index_enveloppe)

        all_index_enveloppe = [
            item for sublist in all_index_enveloppe for item in sublist]
        to_color = []
        for index in all_index_enveloppe:
            # print(index)

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
            length = len(results[index])

            y = np.array(list(data.values()))/list(data.keys()) - \
                ((np.array(list(data.keys())) - 1.7)/(np.array(list(data.keys()))))
            # y sould have a length of 500
            ys[index] = np.pad(-y, (0, 500-len(y)), 'constant')
            if index not in already_colored:
                already_colored.append(index)
                to_color.append(index)

        LD_data_to_color = LD_data[to_color]
        axs[1].scatter(LD_data_to_color[:, 0], LD_data_to_color[:,
                       1], c=color[alpha], label=f'alpha = {alpha}')
        ys = ys.T
        means = np.zeros(len(true_y))
        for i in range(len(true_y)):

            tempo = np.trim_zeros(np.sort(ys[i]))
            means[i] = np.mean(tempo)
        # MSE from alpha
        print("number of points", len(all_index_enveloppe))
        # remove nan values
        true_y = true_y[~np.isnan(means)]
        means = means[~np.isnan(means)]

        print(np.mean((means - true_y)**2))
        # dashed line
        axs[0].plot(np.arange(2, len(means)+2),
                    means, label=f'alpha = {alpha}', color=color[alpha], linestyle='dashdot')

    axs[0].plot(np.arange(2, len(true_y)+2),
                true_y, label='all points')
    axs[0].legend()
    axs[1].legend()
    # plt.show()


def rescale_multiple_layers(LD_data, HD_data, LD_paths, HD_paths, n_layers, color, true_y):

    fig, axs = plt.subplots(1, 2)
    with open('results_color.pkl', 'rb') as fp:
        results = pickle.load(fp)
    already_colored = []
    axs[1].scatter(LD_data[:, 0], LD_data[:, 1])
    for n_layer in n_layers:

        ys = np.zeros([len(LD_data), 500])
        results_2 = [0] * len(LD_data)
        enveloppe, all_index_enveloppe = multiple_enveloppe(LD_data, n_layer)
        all_index_enveloppe = [
            item for sublist in all_index_enveloppe for item in sublist]
        to_color = []
        for index in all_index_enveloppe:
            # print(index)

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
            length = len(results[index])

            y = np.array(list(data.values()))/list(data.keys()) - \
                ((np.array(list(data.keys())) - 1.7)/(np.array(list(data.keys()))))
            # y sould have a length of 500
            ys[index] = np.pad(-y, (0, 500-len(y)), 'constant')
            if index not in already_colored:
                already_colored.append(index)
                to_color.append(index)

        LD_data_to_color = LD_data[to_color]
        axs[1].scatter(LD_data_to_color[:, 0], LD_data_to_color[:,
                       1], c=color[n_layer], label=f'{n_layer} layers')
        ys = ys.T
        means = np.zeros(len(true_y))
        for i in range(len(true_y)):

            tempo = np.trim_zeros(np.sort(ys[i]))
            means[i] = np.mean(tempo)
        # dashed line
        axs[0].plot(np.arange(2, len(means)+2),
                    means, label=f'{n_layer} layers', color=color[n_layer], linestyle='dashdot')
        # MSE
        print("number of points", len(all_index_enveloppe))

        true_y = true_y[~np.isnan(means)]
        means = means[~np.isnan(means)]
        print(np.mean((means - true_y)**2))

    axs[0].plot(np.arange(2, len(true_y)+2),
                true_y, label='all points')
    axs[0].legend()
    axs[1].legend()
    # plt.show()
    #


def rescale_random(LD_data, HD_data, LD_paths, HD_paths, true_y, n):

    with open('results_color.pkl', 'rb') as fp:
        results = pickle.load(fp)
    already_colored = []
    all_index_enveloppe = random.sample(range(len(LD_data)), n)

    ys = np.zeros([len(LD_data), 500])
    for index in all_index_enveloppe:
        # print(index)

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
        length = len(results[index])

        y = np.array(list(data.values()))/list(data.keys()) - \
            ((np.array(list(data.keys())) - 1.7)/(np.array(list(data.keys()))))
        # y sould have a length of 500
        ys[index] = np.pad(-y, (0, 500-len(y)), 'constant')
        if index not in already_colored:
            already_colored.append(index)

    ys = ys.T
    means = np.zeros(len(true_y))
    for i in range(len(true_y)):

        tempo = np.trim_zeros(np.sort(ys[i]))
        # tempo wmust have len of trye_y
        means[i] = np.mean(tempo)
    # replace nan with 0 in means
    for i in range(len(means)):
        if np.isnan(means[i]):
            means[i] = 0
    plt.plot(np.arange(2, len(means)+2), means)
    plt.plot(np.arange(2, len(true_y)+2), true_y)
    plt.show()
    print(means)
    print(true_y)
    print("MSE", np.mean((means - true_y)**2))
    return np.mean((means - true_y)**2)


# folders = ['Isomap_MNIST_data', 'Isomap_COIL-20_data', 'MDS_MNIST_data', 'MDS_COIL-20_data', 'PCA_MNIST_data',
#            'PCA_COIL-20_data', 't-SNE_MNIST_data', 't-SNE_COIL-20_data', 'UMAP_MNIST_data', 'UMAP_COIL-20_data']
# folders = ['t-SNE_MNIST_data']
# folders = ['MDS_COIL-20_data']
# folders = ['UMAP_COIL-20_data']
folders = ['Isomap_MNIST_data']
#
for folder in folders:
    print(folder)
# load data
    LD_data = np.load('./'+folder+'/LD_data.npy')
    HD_data = np.load('./'+folder+'/HD_data.npy')
# load paths
    HD_paths = np.load('./'+folder+'/HD_all_paths.npy')
    HD_paths_2 = np.load('./'+folder+'/HD_all_paths_2.npy')
    LD_paths = np.load('./'+folder+'/LD_all_paths.npy')
    LD_paths_2 = np.load('./'+folder+'/LD_all_paths_2.npy')
# load distance matrix
    LD_distance_matrix = np.load('./'+folder+'/LD_distance_matrix.npy')
    HD_distance_matrix = np.load('./'+folder+'/HD_distance_matrix.npy')
#     path_rnx_distance_sorted(LD_data, HD_data, LD_paths_2, HD_paths_2)
#     true_y = rescale(LD_data, HD_data, LD_paths_2, HD_paths_2)
#     print("MSE from alphas (0.3, 0.5, 0.7, 0.9)")
#     rescale_multiple_alpha(LD_data, HD_data, LD_paths_2,
#                            HD_paths_2, [0.3, 0.5, 0.7, 0.9], {0.3: 'r', 0.5: 'b', 0.7: 'g', 0.9: 'orange'}, true_y)
#     print("MSE from layers (1, 3, 5, 10)")
#     rescale_multiple_layers(LD_data, HD_data, LD_paths_2,
#                             HD_paths_2, [1, 3, 5, 10], {1: 'r', 3: 'b', 5: 'g', 10: 'orange'}, true_y)
    size = len(LD_data)//30 - 1
    path_rnx_distance_sorted(LD_data, HD_data, LD_paths_2, HD_paths_2)
    true_y = rescale(LD_data, HD_data, LD_paths_2, HD_paths_2)
    plt.show()
    rmse_results = {}
    for i in range(2, 29):
        print(i)
        for j in range(10):
            rmse = rescale_random(LD_data, HD_data, LD_paths_2,
                                  HD_paths_2, true_y, i)
            if i not in rmse_results:
                rmse_results[i] = [rmse]
            else:
                rmse_results[i].append(rmse)
    for i in range(size):
        print((i+1)*30)
        for j in range(20):
            rmse = rescale_random(LD_data, HD_data, LD_paths_2,
                                  HD_paths_2, true_y, (i+1)*30)
            if (i+1)*30 not in rmse_results:
                rmse_results[(i+1)*30] = [rmse]
            else:
                rmse_results[(i+1)*30].append(rmse)
    rmse_worst = {}
    rmse_best = {}
    for key in rmse_results:
        rmse_worst[key] = np.max(rmse_results[key])
        rmse_best[key] = np.min(rmse_results[key])
        rmse_results[key] = np.mean(rmse_results[key])
    print(rmse_results)
    print(rmse_worst)
    print(rmse_best)
    plt.plot(list(rmse_results.keys()), list(rmse_results.values()),
             label="average RMSE on 10 random sample of size x")
    plt.plot(list(rmse_worst.keys()), list(rmse_worst.values()),
             label="worst RMSE", linestyle='dashed')
    plt.plot(list(rmse_best.keys()), list(rmse_best.values()),
             label="best RMSE", linestyle='dashed')

    # color the areas between the worst and best RMSE
    plt.fill_between(list(rmse_worst.keys()), list(rmse_worst.values()), list(
        rmse_best.values()), color='gray', alpha=0.5)
    plt.xlabel('Number of points')
    plt.ylabel('RMSE')
    plt.yscale('log')

    # mnist t-SNE
    # # 8.73 ∗ 10−5 (178) 6.65 ∗ 10−5 (541) 1.33 ∗ 10−5 (1424) 3.59 ∗ 10−6 (1902)
    # x = [178, 541, 1424, 1902]
    # y = [8.73 * 10**-5, 6.65 * 10**-5, 1.33 * 10**-5, 3.59 * 10**-6]
    # plt.scatter(x, y, label='RMSE using alpha shape', color='r')
    # # 7.86 ∗ 10−5 (29) 5.19 ∗ 10−5 (91) 2.19 ∗ 10−5 (150) 2.07 ∗ 10−5 (309)
    # x = [29, 91, 150, 309]
    # y = [7.86 * 10**-5, 5.19 * 10**-5, 2.19 * 10**-5, 2.07 * 10**-5]
    # plt.scatter(x, y, label='RMSE using multiple layers', color='g')
    ##
    # MDS COIL-20
    ##
    # 2.80 ∗ 10−4 (125) 1.15 ∗ 10−4 (236) 1.68 ∗ 10−4 (631) 2.08 ∗ 10−4 (770)
    # x = [125, 236, 631, 770]
    # y = [2.80 * 10**-4, 1.15 * 10**-4, 1.68 * 10**-4, 2.08 * 10**-4]
    # plt.scatter(x, y, label='RMSE using alpha shape', color='r')
    # 1.68 ∗ 10−4 (10) 4.78 ∗ 10−4 (53) 2.99 ∗ 10−4 (95) 5.80 ∗ 10−5 (207)
    # x = [10, 53, 95, 207]
    # y = [1.68 * 10**-4, 4.78 * 10**-4, 2.99 * 10**-4, 5.80 * 10**-5]
    # plt.scatter(x, y, label='RMSE using multiple layers', color='g')
    ##
    # Isomap mnist
    #
    # 2.42 ∗ 10−4(127) 1.68 ∗ 10−4(254) 7.73 ∗ 10−5(497) 4.24 ∗ 10−5(716)
    x = [127, 254, 497, 716]
    y = [2.42 * 10**-4, 1.68 * 10**-4, 7.73 * 10**-5, 4.24 * 10**-5]
    plt.scatter(x, y, label='RMSE using alpha shape', color='r')
    # 1.95 ∗ 10−4(13) 2.75 ∗ 10−4(45) 2.40 ∗ 10−4(74) 1.56 ∗ 10−4(174)
    x = [13, 45, 74, 174]
    y = [1.95 * 10**-4, 2.75 * 10**-4, 2.40 * 10**-4, 1.56 * 10**-4]
    plt.scatter(x, y, label='RMSE using multiple layers', color='g')
    #
    # # UMAP COIL-20
    # #
    # # 2.05 ∗ 10−5 (446) 1.04 ∗ 10−5 (561) 4.80 ∗ 10−6 (758) 3.59 ∗ 10−6 (852)
    # x = [446, 561, 758, 852]
    # y = [2.05 * 10**-5, 1.04 * 10**-5, 4.80 * 10**-6, 3.59 * 10**-6]
    # plt.scatter(x, y, label='RMSE using alpha shape', color='r')
    # # 1.57 ∗ 10−4 (18) 1.86 ∗ 10−4 (53) 2.17 ∗ 10−4 (77) 1.83 ∗ 10−4 (143)
    # x = [18, 53, 77, 143]
    # y = [1.57 * 10**-4, 1.86 * 10**-4, 2.17 * 10**-4, 1.83 * 10**-4]
    # plt.scatter(x, y, label='RMSE using multiple layers', color='g')
    #
    plt.legend()
    plt.show()
