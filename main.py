import random as random
from classifiers.income.income_classifier import IncomeClassifier
from classifiers.digits.digits import NeuralNetwork
from synthesizer import Synthesizer
import pandas as pd
import numpy as np
import pickle
import csv
import matplotlib.pyplot as plt
import math as math
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from pandas import DataFrame
import warnings
warnings.filterwarnings("ignore")

#CONSTANTS
num_of_shadow_models = 250
number_of_records_to_synthesize = 1000

############################
###### INCOME DATASET ######
############################


def incomeRandomizeFunction(k, x, c):
    x_age = random.randint(0, 100)
    x_education = random.randint(0, 16)
    x_marital_status = random.randint(0, 1)
    x_sex = random.randint(0, 1)
    x_gain = random.randint(0, 100000)
    x_loss = random.randint(0, 100000)
    x_hpw = random.randint(0, 100)
    x_temp = [x_age, x_education, x_marital_status, x_sex, x_gain, x_loss, x_hpw]

    if len(x) == 0:
        return x_temp

    selected_features = random.sample(range(len(x_temp)), k)

    for argfeature in selected_features:
        x[argfeature] = x_temp[argfeature]

    return x


print "Loading Income Classifier Data\n"
with open('income_classifier', 'rb') as f:
    income_classifier = pickle.load(f)
dataset = pd.read_csv("classifiers/income/adult.csv")
print "Intializing Classifier\n"
income_classifier = IncomeClassifier(dataset)
with open('income_classifier', 'wb') as f:
    pickle.dump(income_classifier, f)

print "Intializing Training Data Synthesis\n"

synthesizer = Synthesizer(
    2, 7, 1, 100, 0.9, 5, incomeRandomizeFunction, income_classifier.predict_probabilities)
    #c,kmax,kmin,iter_max,conf_min,rej_max,randomizeFunction,predictProb
synthesizer.synthesize(number_of_records_to_synthesize) #number of records



## ATTACK MODEL ###

print "Loading Attack Model Training Data\n"
#Combine and shuffle Data
dataset = DataFrame()
for c in range(2):
    temp = DataFrame.from_records(list(csv.reader(open("training_class_"+str(c)+".csv"))))
    temp.insert(7, "Class", c)
    dataset = dataset.append(temp, ignore_index=True)
dataset = dataset.sample(frac=1).reset_index(drop=True)
dataset = dataset.convert_objects(convert_numeric=True)

print "Creating shadow models\n"
# Create shadow models and subsequently the Attack Training Data
records_per_model = len(dataset)/ num_of_shadow_models
training_dataset_target = DataFrame()
for k in range(num_of_shadow_models):
    shadow_data = dataset[k*records_per_model: (k+1)*records_per_model]
    array = shadow_data.values
    X = array[:,0:7]
    Y = array[:,7]
    validation_size = 0.50
    seed = 7
    X_train, X_validation, Y_train, Y_validation = train_test_split(X,Y,
        test_size=validation_size,random_state=seed)
    shadow_model = RandomForestClassifier(n_estimators=250,max_features=5)
    shadow_model.fit(X_train, Y_train)
    prob = shadow_model.predict_proba(X_train)
    in_data = DataFrame()
    in_data.insert(0, "Class", Y_train)
    for c in range(0,2):
        in_data.insert(1+c, "Prob" + str(c), prob[:,c])
    in_data.insert(1+2, "In/Out", 1)
    prob = shadow_model.predict_proba(X_validation)
    training_dataset_target = training_dataset_target.append(in_data, ignore_index=True)
    out_data = DataFrame()
    out_data.insert(0, "Class", Y_validation)
    for c in range(0,2):
        out_data.insert(1+c, "Prob" + str(c), prob[:,c])
    out_data.insert(1+2, "In/Out", 0)
    training_dataset_target = training_dataset_target.append(out_data, ignore_index=True)

print "Creating Attack Model For Each class\n"
#Create predictor for each class
predictors = []
for c in range(0, 2):
    class_data = training_dataset_target[training_dataset_target["Class"]==c]
    array = class_data.values
    X = array[:,0:3]
    Y = array[:,3]
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = train_test_split(X,Y,
        test_size=validation_size,random_state=seed)
    predictor = RandomForestClassifier(n_estimators=250,max_features=3)
    predictor.fit(X_train, Y_train)
    predictors.append(predictor)


# Calculate accuracy
print "Calculating Accuracy of Attack Model"
correct = 0
dataset = list(csv.reader(open("X_train.csv")))[:6000]
total = len(dataset)

probs = income_classifier.random_forest.predict_proba(dataset)
input_data = DataFrame()
input_data.insert(0, "Class", [np.argmax(i) for i in probs])
for c in range(0,2):
    input_data.insert(1+c, "Prob" + str(c), probs[:,c])

for c in range(0,2):
    labeled_input = input_data[input_data["Class"] == c]
    result = predictors[c].predict_proba(labeled_input)
    correct += [np.argmax(i) for i in result].count(1)

dataset = list(csv.reader(open("X_validation.csv")))[:6000]
total = total + len(dataset)

probs = income_classifier.random_forest.predict_proba(dataset)
input_data = DataFrame()
input_data.insert(0, "Class", [np.argmax(i) for i in probs])
for c in range(0,2):
    input_data.insert(1+c, "Prob" + str(c), probs[:,c])

for c in range(0,2):
    labeled_input = input_data[input_data["Class"] == c]
    result = predictors[c].predict_proba(labeled_input)
    correct += [np.argmax(i) for i in result].count(0)
print correct, total
print "\n\nAccuracy: " + str(correct*1.0/total)

"""

###########################
###### DIGITS DATASET #####
###########################

print "Loading MNIST Classifier Data\n"
with open("classifiers/digits/pickled_mnist.pkl", "rb") as fh:
    data = pickle.load(fh)
train_imgs = data[0]
test_imgs = data[1]
train_labels = data[2]
test_labels = data[3]
train_labels_one_hot = data[4]
test_labels_one_hot = data[5]
image_size = 28  # width and length
no_of_different_labels = 10  # i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size

print "Initialising NN Classifier\n"

ANN = NeuralNetwork(no_of_in_nodes=image_pixels,
                    no_of_out_nodes=10,
                    no_of_hidden_nodes=100,
                    learning_rate=0.1)

for i in range(len(train_imgs)):
    ANN.train(train_imgs[i], train_labels_one_hot[i])


### Synthesis Code ###
print "Intializing Training Data Synthesis\n"

init_x = {}
for digit in range(0, 10):
    init_x[digit] = [i for i, x in enumerate(test_labels) if x == digit][:10]

def digitsRandomizeFunction(k, x, c):
    fac = 0.99 / 255

    if len(x) == 0:
        x = np.asfarray(
            test_imgs[init_x[c][random.randint(0, 9)]]) * fac + 0.01

    gray_features = [i for i, feature in enumerate(x) if feature > np.min(x)]

    # There may be less gray pixels i.e. less than k. For this we check how many
    # 'extra features' there are and repeat the process of manipulating feature
    # values to compensate this scarcity

    extra_features = 0
    selected_gray_features = gray_features
    if len(gray_features) >= k:
        selected_gray_features = random.sample(gray_features, k)
    else:
        extra_features = k - len(gray_features)
    
    #we define a stride i.e. the distance from the selected pixel that can be altered
    stride = 1

    for grayfeature in selected_gray_features:
        # choose a nearby pixel. stride pixels away from the selected gray feature
        row_number = int(grayfeature/28)
        column_number = grayfeature%28
        new_row = random.randint(max(row_number - stride, 0), min(row_number +  stride, 27))
        new_col = random.randint(max(column_number - stride, 0), min(column_number +  stride, 27))
        indice_to_change = new_row * 28 + new_col
        x[indice_to_change] = random.randint(0, 255) * fac + 0.01

    #Repeat the process for the extra features
    for extra_feature in range(extra_features):
        grayfeature = selected_gray_features[random.randint(0, len(selected_gray_features) - 1)]
        row_number = int(grayfeature/28)
        column_number = grayfeature%28
        new_row = random.randint(max(row_number - stride, 0), min(row_number +  stride, 27))
        new_col = random.randint(max(column_number - stride, 0), min(column_number +  stride, 27))
        indice_to_change = new_row * 28 + new_col
        x[indice_to_change] = random.randint(0, 255) * fac + 0.01

    return x

def digitsPredictProb(x):
    output_vector = np.array(ANN.run(x))
    return output_vector/sum(output_vector)


synthesizer = Synthesizer(10, 28*28, 1, 200, 0.95, 200,
                          digitsRandomizeFunction, digitsPredictProb)
                          #c, kmax, kmin, iter_max, conf_min, rej_max,
                          #randomizeFunction, predictProb

print synthesizer.synthesize(10)

print "Testing probabilities..."
for c in range(3,10):
    record = [float(i) for i in np.array(list(csv.reader(open("training_class_"+str(c)+".csv")))[0])]
    print "\n\nProb dist for class" + str(c)
    img = np.array(record).reshape((28,28))
    plt.imshow(img, cmap="Greys")
    plt.show()
    print digitsPredictProb(record)"""
