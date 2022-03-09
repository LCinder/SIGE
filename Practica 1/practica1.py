import numpy
import numpy as np
import pandas
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mlxtend.feature_selection import SequentialFeatureSelector as sequential_feature
from imblearn.over_sampling import SMOTE
from seaborn import heatmap
import matplotlib.pyplot as plot
from IPython.display import display

def load_dataset():
    data = pandas.read_csv("data/diabetes.csv", delimiter=";")
    features = data.columns[1:len(data.columns)]
    x = pandas.DataFrame(data[features])
    y = pandas.DataFrame(data["Diabetes_binary"])
    y.replace({False: 0, True: 1}, inplace=True)
    normalize(x)
    return x, y

def normalize(x):
    for column in x.columns:
        x[column] = x[column] / x[column].max()

def remove_columns(data_arg, elements):
    data = data_arg.copy()
    for element in elements:
        data = data.drop(element, axis="columns")
    return data


def pred(x_arg, y_arg):
    predictor = DecisionTreeClassifier(random_state=1)
    return round(cross_val_score(predictor, x_arg, y_arg).mean(), 3)


def discretize(x_arg, y_arg):
    elements = [ "BMI", "Age", "Income"]
    bins = [2, 4, 6]

    accuracy_mean = pred(x_arg, y_arg)
    print("Accuracy: " + str(round(accuracy_mean, 3)))
    x_train_disc_aux = x_arg.copy()

    accuracy_max = 0
    for variable in elements:
        for bin_i in bins:
            x_train_disc = x_arg.copy()

            x_train_disc_aux[variable] = pandas.cut(x_train_disc[variable], labels=range(bin_i), bins=bin_i)
            accuracy_mean_discr = pred(x_train_disc_aux, y_arg)
            if accuracy_mean_discr > accuracy_max:
                x_train_disc[variable] = x_train_disc_aux[variable]

            print("Accuracy Discretize with variable/bin: " + variable + "/"
            + str(bin_i) + " : " + str(round(accuracy_mean_discr, 3)))

    return x_train_disc


def loss_values():
    unknown = -999
    x.replace(unknown, numpy.NAN, inplace=True)
    x_mode = x.copy()
    x_mean = x.copy()
    x_instances = x.copy()
    x_char = x.copy()
    y_instances = y.copy()

    # Mean and mode
    for variable in x:
        x_mean[variable] = x_mean[variable].replace(numpy.NAN, None)

        mean = x_mean[variable].mean()
        x_mean[variable].fillna(value=mean)

        mode = x_mode[variable].mode()[0]
        x_mode[variable].replace(numpy.NAN, mode, inplace=True)

    # Instances
    indexes = []
    for index, row in x_instances.iterrows():
        if row.isnull().any():
            indexes.append(index)
    y_instances.drop(indexes, inplace=True)
    x_instances = x_instances.dropna()

    # Characteristics
    x_char = x_char.dropna(axis=1)

    return x_mean, x_mode, x_char, x_instances, y_instances


def plot_correlation():
    correlation_matrix = x.corr()
    heatmap(correlation_matrix)
    plot.show()


#https://www.analyticsvidhya.com/blog/2021/04/backward-feature-elimination-and-its-implementation/
def selection(x_arg, y_arg):
    predictor = RandomForestClassifier(random_state=1)
    predictor.fit(x_arg, numpy.ravel(y_arg))
    features_importances = predictor.feature_importances_
    features_names = predictor.feature_names_in_
    #not_selected = list(set(x_arg.columns) - set(features))

    feature_importance = pandas.DataFrame({"Feature": features_names,
    "Importance": features_importances}).sort_values("Importance", ascending=False)

    display(feature_importance)
    plot.savefig("img/feature_importances.png")

    x_sel = remove_columns(x_arg, not_selected)
    removed_features = pred(x_sel, y_arg)

    print("Accuracy All Characteristics: " + str(all))
    print("Features: " + str(features))
    print("Accuracy Removed Features: " + str(removed_features))

    return x_sel


if __name__ == "__main__":
    x, y = load_dataset()
    # x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6)

    x_mean, x_mode, x_char, x_instances, y_instances = loss_values()
    plot_correlation()
    print(pred(x_mean, y))
    print(pred(x_mode, y))
    print(pred(x_char, y))
    print(pred(x_instances, y_instances))

    #
    # #Ejercicio 1
    print("------------------------------------------------------------------")
    x_mean = discretize(x_mean, y)
    print("------------------------------------------------------------------")
    #
    print("------------------------------------------------------------------")
    x_sel = selection(x_mean, y)
    print("------------------------------------------------------------------")
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


