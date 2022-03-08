import numpy as np
import pandas
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from mlxtend.feature_selection import SequentialFeatureSelector as sequential_feature
from imblearn.over_sampling import SMOTE
from seaborn import heatmap
import matplotlib.pyplot as plot


def load_dataset():
    data = pandas.read_csv("data/diabetes.csv", delimiter=";")
    features = data.columns[1:len(data.columns)]
    x = pandas.DataFrame(data[features])
    y = pandas.DataFrame(data["Diabetes_binary"])
    y.replace({False: 0, True: 1}, inplace=True)
    return x, y


def remove_columns(data_arg, elements):
    data = data_arg[:]
    for element in elements:
        data = data.drop(element, axis="columns")
    return data


def pred(x_arg, y_arg):
    predictor = DecisionTreeClassifier(random_state=1)
    return round(cross_val_score(predictor, x_arg, y_arg).mean(), 3)


def discretize(x_train_arg, x_test_arg):
    elements = ["HOUR_I", "SPDLIM_H", "MONTH"]
    bins = [2, 4, 5, 10]

    accuracy_mean = pred(x_train_arg, y_train, x_test_arg, y_test)
    print("Accuracy: " + str(round(accuracy_mean, 3)))

    for variable in elements:
        for bin_i in bins:
            x_train_disc = x_train_arg[:]

            x_train_disc[variable] = pandas.cut(x_train_disc[variable], labels=range(bin_i), bins=bin_i)
            accuracy_mean_discr = pred(x_train_disc, y_train, x_test_arg, y_test)

            if bin_i == 2 and variable == "HOUR_I":
                x_train_disc_final = x_train_disc

            print("Accuracy Discretize with variable/bin: " + variable + "/"
            + str(bin_i) + " : " + str(round(accuracy_mean_discr, 3)))

    return x_train_disc_final


def loss_values():
    x_mode = x.copy()
    x_mean = x.copy()
    x_char = x.copy()
    y_char = y.copy()
    unknown = -999

    for variable in x:
        x_mean[variable] = x_mean[variable].replace(unknown, None)

        mean = x_mean[variable].mean()
        x_mean[variable].fillna(value=mean)

        mode = x_mode[variable].mode()[0]
        x_mode[variable].replace(unknown, mode, inplace=True)

    for i in x.columns:
        #x_char = x_char[(x_char[i] != unknown)]
        x_char = x_char.columns[(x_char[i].columns != unknown)]

    s = x_char.columns[x_char.columns == unknown]
    missing_values = x_char.columns[(x_char == unknown).any()]
    y_char = remove_columns(y_char, missing_values)

    return x_mean, x_mode, x_char, y_char


def plot_correlation():
    correlation_matrix = x.corr()
    heatmap(correlation_matrix)
    plot.show()


#https://www.analyticsvidhya.com/blog/2021/04/backward-feature-elimination-and-its-implementation/
def selection():
    s = sequential_feature(GaussianNB(), k_features=10, forward=False)
    s = s.fit(x, np.ravel(y_train))
    features = list(s.k_feature_names_)

    not_selected = list(set(x.columns) - set(features))

    x_sel = remove_columns(x, not_selected)
    x_test_sel = remove_columns(x_test, not_selected)
    removed_features = pred(x_sel, y_train, x_test_sel, y_test)

    print("Accuracy All Characteristics: " + str(all))
    print("Features: " + str(features))
    print("Accuracy Removed Features: " + str(removed_features))

    return x_sel, x_test_sel


if __name__ == "__main__":
    x, y = load_dataset()
    # x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6)

    x_mean, x_mode, x_char, y_char = loss_values()
    plot_correlation()
    print(pred(x_mean, y))
    print(pred(x_mode, y))
    print(pred(x_char, y_char))
    #
    # #Ejercicio 1
    # x_train_discr = discretize(x_train, x_test)
    # print("------------------------------------------------------------------")
    #
    # #Ejercicio 2
    # best, mode = loss_values()
    #
    # print("El mejor es " + mode)
    # print("------------------------------------------------------------------")
    #
    # #Ejercicio 3
    # x_test_not_imputed = remove_columns(x_test, not_imputed)
    # x_sel_char, x_test_sel_char = selection(x_train_discr, x_test_not_imputed)
    # x_sample = pandas.DataFrame(x_sel_char)
    # x_sample = x_sample.sample(frac=0.3, random_state=1)
    # y_train = y_train.loc[x_sample.index]
    # x_test_sample = remove_columns(x_test, set(x_train.columns) - set(x_sample.columns))
    #
    # print("Muestreo del 30%")
    # acc_res_sample = 0
    # acc_res_oversample = 0
    #
    # acc_res_sample = pred(x_sample, y_train, x_test_sample, y_test)
    # print("Accuracy sample: " + str(acc_res_sample))
    #
    # x_oversample, y_train_oversample = SMOTE().fit_resample(x_sample, y_train)
    # acc_res_oversample = pred(x_oversample, y_train_oversample, x_test_sample, y_test)
    # print("Accuracy oversampling: " + str(acc_res_oversample))


