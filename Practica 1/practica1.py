import numpy
import numpy as np
import pandas
import seaborn
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
    unknown = -999
    x.replace(unknown, numpy.NAN, inplace=True)
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
    predictor = RandomForestClassifier(random_state=1)
    return round(cross_val_score(predictor, x_arg, numpy.ravel(y_arg)).mean(), 3)


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


def loss_values(x):
    x = remove_columns(x, ["NoDocbcCost"])
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
    all_features = pred(x_arg, y_arg)

    features_importances = predictor.feature_importances_
    features_names = predictor.feature_names_in_
    #not_selected = list(set(x_arg.columns) - set(features))

    feature_importance = pandas.DataFrame({"Feature": features_names,
    "Importance": features_importances}).sort_values("Importance", ascending=False)

    display(feature_importance)
    plot.savefig("img/feature_importances.png")

    selected_features = [3, 5, 7, 10, 15]
    for element in selected_features:
        best_features = feature_importance[0:element]
        x_sel = x_arg.copy()

        x_sel = remove_columns(x_sel, best_features.Feature)
        removed_features = pred(x_sel, y_arg)

        print("--------------------------------------------------")
        print("Number: {}".format(element))
        print("Accuracy All Characteristics: " + str(all_features))
        print("Features: " + str(best_features.Feature))
        print("Accuracy Removed Features: " + str(removed_features))

    return x_sel

def resample():
    dataf = pandas.read_csv("data/diabetes.csv", delimiter=";")
    occurrences_0, occurrences_1 = dataf.Diabetes_binary.value_counts()

    dataf_0 = dataf[dataf.Diabetes_binary == 0]
    dataf_1 = dataf[dataf.Diabetes_binary == 1]

    dataf_0 = dataf_0.sample(occurrences_1)
    diabetes_count = [numpy.count_nonzero(dataf_0.Diabetes_binary == 0), numpy.count_nonzero(dataf_1.Diabetes_binary == 1)]
    plot.pie(diabetes_count, labels=["No Diabetes", "Diabetes"], autopct="%1.1f%%",
    colors=seaborn.color_palette("pastel"))
    plot.title(label="Samples proportion (Diabetes)")
    plot.savefig("proportion_2.png")
    plot.show()

    dataf_resample = pandas.concat([dataf_0, dataf_1])
    features = dataf_resample.columns[1:len(dataf_resample.columns)]
    x_resample = pandas.DataFrame(dataf_resample[features])
    y_resample = pandas.DataFrame(dataf_resample["Diabetes_binary"])
    y_resample.replace({False: 0, True: 1}, inplace=True)
    normalize(x_resample)

    return x_resample, y_resample


def plot_EDA():
    diabetes_count = [numpy.count_nonzero(y == 0), numpy.count_nonzero(y == 1)]
    plot.pie(diabetes_count, labels=["No Diabetes", "Diabetes"], autopct="%1.1f%%", colors=seaborn.color_palette("pastel"))
    plot.title(label="Samples proportion (Diabetes)")
    plot.savefig("proportion.png")
    plot.show()

    plot.figure(figsize=(20, 10))
    plot.bar(x.columns, x.isna().sum(), color=seaborn.color_palette("pastel"))
    plot.xticks(rotation=45)
    plot.title(label="Number of missing values")
    plot.xlabel("Columns")
    plot.ylabel("missing values")
    plot.savefig("img/missing_values.png")
    plot.show()

    x_plot = x.copy()
    x_plot = x_plot.dropna()

    for column, i in zip(x_plot.columns, range(x_plot.columns.shape[0])):
        if len(numpy.unique(x_plot[column])) != 2:
            plot.hist(x_plot[column], color=seaborn.color_palette("pastel")[i%len(seaborn.color_palette("pastel"))])
        else:
            column_count = [numpy.count_nonzero(x_plot[column] == 0), numpy.count_nonzero(x_plot[column] == 1)]
            plot.pie(column_count, colors=seaborn.color_palette("pastel"), labels=["No (0)", "Si (1)"], autopct="%1.1f%%")
        plot.title(label="Distribucion {}".format(column))
        plot.savefig("img/{}.png".format(column))
        plot.show()

    ##############################################################################
    ###########################All plots##########################################
    ##############################################################################



if __name__ == "__main__":
    x, y = load_dataset()
    #plot_EDA()


    x_mean, x_mode, x_char, x_instances, y_instances = loss_values(x)
    plot_correlation()
    print(pred(x_mean, y))
    print(pred(x_mode, y))
    print(pred(x_char, y))
    print(pred(x_instances, y_instances))

    #
    # #Ejercicio 1
    print("------------------------------------------------------------------")
    x_mean = discretize(x_mode, y)
    print("------------------------------------------------------------------")
    #
    print("------------------------------------------------------------------")
    x_sel = selection(x_mode, y)
    print("------------------------------------------------------------------")

    x_sample = x_mode.copy()
    x_sample = x_sample.sample(frac=0.3, random_state=1)
    y_sample = y.loc[x_sample.index]

    print("Muestreo del 30%")
    acc_res_sample = 0
    acc_res_oversample = 0

    acc_res_sample = pred(x_sample, y_sample)
    print("Accuracy sample: " + str(acc_res_sample))

    x_oversample, y_train_oversample = SMOTE().fit_resample(x_sample, y_sample)
    acc_res_oversample = pred(x_oversample, y_train_oversample)
    print("Accuracy oversampling: " + str(acc_res_oversample))

    x_resample, y_resample = resample()
    acc_res_undersampling = pred(x_resample, y_resample)
    print("Accuracy undersampling: " + str(acc_res_undersampling))




