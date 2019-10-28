from modules.ModelExtraction.QuerySynthesis import Synthesizer
import pandas as pd
import numpy as np
import random as random
import pickle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score
from IPython.display import Image  
import pydotplus

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

#########################################################
#### LOAD DATA AND DO PREPROCESSING FOR TARGET MODEL ####
#########################################################

#Loading dataset
dataset = pd.read_csv('data/income/adult.csv')
dataset = dataset.fillna(np.nan)
# Reformat Column We Are Predicting: 0 means less than 50K. 1 means greater than 50K.
dataset['income']=dataset['income'].map({'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1})
Y = np.array(dataset['income'])
results = open("results/income/RESULTS.txt","a") 

# Fill Missing Category Entries
dataset["workclass"] = dataset["workclass"].fillna("?")
dataset["occupation"] = dataset["occupation"].fillna("?")
dataset["native.country"] = dataset["native.country"].fillna("United-States")
dataset = pd.concat([dataset,pd.get_dummies(dataset['sex'], prefix='sex')],axis=1)
dataset = pd.concat([dataset,pd.get_dummies(dataset['workclass'], prefix='workclass')],axis=1)
dataset = pd.concat([dataset,pd.get_dummies(dataset['education'], prefix='education')],axis=1)
dataset = pd.concat([dataset,pd.get_dummies(dataset['marital.status'], prefix='marital.status')],axis=1)
dataset = pd.concat([dataset,pd.get_dummies(dataset['occupation'], prefix='occupation')],axis=1)
dataset = pd.concat([dataset,pd.get_dummies(dataset['relationship'], prefix='relationship')],axis=1)
dataset = pd.concat([dataset,pd.get_dummies(dataset['race'], prefix='race')],axis=1)
dataset = pd.concat([dataset,pd.get_dummies(dataset['native.country'], prefix='native.country')],axis=1)
dataset.drop(labels=["income", "education.num", "sex", "workclass", "education", "marital.status", "occupation", "relationship", "race", "native.country"], axis = 1, inplace = True)
X = np.array(dataset)

#Splitting of dataset into Training set and testing set (80% and 20% respectively)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

############################
#### TRAIN TARGET MODEL ####
############################

#Single layer architecture
model = Sequential()
model.add(Dense(107,input_shape=(X_train.shape[1],)))
model.add(Activation('relu'))
model.add(Dense(64,input_shape=(X_train.shape[1],)))
model.add(Activation('relu'))
model.add(Dense(64,input_shape=(X_train.shape[1],)))
model.add(Activation('relu'))
model.add(Dense(128,input_shape=(X_train.shape[1],)))
model.add(Activation('relu'))
model.add(Dense(128,input_shape=(X_train.shape[1],)))
model.add(Activation('relu'))
model.add(Dense(256,input_shape=(X_train.shape[1],)))
model.add(Activation('relu'))
model.add(Dense(256,input_shape=(X_train.shape[1],)))
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='sparse_categorical_crossentropy',
              optimizer="sgd",metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=128, epochs=10, verbose=1, validation_data=(X_test, Y_test))

loss, accuracy = model.evaluate(X_test,Y_test, verbose=0)
predictionsNN = [np.argmax(x) for x in model.predict(np.array(X_test))]
print("\n\nAccuracy of Neural Network (Target BB Model): %s%%\n" % str(accuracy*100.0))
results.write("\n\nAccuracy of Neural Network (Target BB Model): %s%%\n" % str(accuracy*100.0))

dtree = DecisionTreeClassifier(random_state=0)
dtree.fit(X_train,Y_train)

## Test accuracy and similarity
predictionsDT = dtree.predict(X_test)
print("Accuracy of DT (trained and tested on original data): %s%%\n" % (100*accuracy_score(Y_test, predictionsDT)))
results.write("Accuracy of DT (trained and tested on original data): %s%%\n" % (100*accuracy_score(Y_test, predictionsDT)))

print("Classification similarity of NN and DT trained on target model dataset: %s%%\n" % (np.sum(predictionsDT==predictionsNN)*1.0/len(predictionsDT)*100))
results.write("Classification similarity of NN and DT trained on target model dataset: %s%%\n" % (np.sum(predictionsDT==predictionsNN)*1.0/len(predictionsDT)*100))


#Visualisation
dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data, 
                feature_names=list(dataset.columns),
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.write_png("visualisations/income/income_dtree_trained_original.png"))

###########################
##### SYNTHESIZE DATA #####
###########################

print "Intializing Training Data Synthesis\n"

def incomeRandomizeFunction(k, x, c):
    x_age = random.randint(0, 100)
    x_fnlwgt = random.randint(12285, 1484705)
    x_capital_gain = random.randint(0, 99999)
    x_capital_loss = random.randint(0, 4356)
    x_hours_per_week = random.randint(0, 99)
    x_one_hot = [random.randint(0, 1) for i in range(102)]
    x_temp = [x_age, x_fnlwgt, x_capital_gain, x_capital_loss, x_hours_per_week] + x_one_hot

    if len(x) == 0:
        return x_temp

    selected_features = random.sample(range(len(x_temp)), k)

    for argfeature in selected_features:
        x[argfeature] = x_temp[argfeature]

    return x

def incomePredictProb(x):
    return list(model.predict(np.array([x]))[0])


synthesizer = Synthesizer(2, 107, 1, 100, 0.6, 100,
                          incomeRandomizeFunction, incomePredictProb)
                          #c, kmax, kmin, iter_max, conf_min, rej_max,
                          #randomizeFunction, predictProb

synthesizer.synthesize(1000, "data/income/income_synthesized_data.csv")

#####################################################
#### CREATE SHADOW MODEL IN FORM OF DECISION TREE ###
#####################################################

## Load Data
dataset = pd.read_csv("data/income/income_synthesized_data.csv",  header=None)
array = dataset.values
X = array[:,0:len(dataset.columns)-1]
Y = array[:,len(dataset.columns)-1]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X,Y,
    test_size=validation_size,random_state=seed)

## Create Decision Tree trained on synthesized data
dtree=DecisionTreeClassifier()
dtree.fit(X_train,Y_train)

## Test accuracy of Decision Tree trained on synthesized and tested on synthesized
predictions = dtree.predict(X_validation)
print("Classification similarity of NN and DT trained on synthesized dataset: %s%%\n" % (100*accuracy_score(Y_validation, predictions)))
results.write("Classification similarity of NN and DT trained on synthesized dataset: %s%%\n" % (100*accuracy_score(Y_validation, predictions)))

## Test accuracy of Decision Tree trained on synthesized and tested on original
predictions = dtree.predict(X_test)
print("Accuracy of DT (trained on synthesized and tested on original): %s%%\n" % (100*accuracy_score(Y_test, predictions)))
results.write("Accuracy of DT (trained on synthesized and tested on original): %s%%\n" % (100*accuracy_score(Y_test, predictions)))

## Visualisation
dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data, 
                feature_names=dataset.columns[:len(dataset.columns)-1],
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.write_png("visualisations/income/income_dtree_trained_synthesized.png"))
