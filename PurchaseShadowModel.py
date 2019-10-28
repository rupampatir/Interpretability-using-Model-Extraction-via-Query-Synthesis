from modules.ModelExtraction.QuerySynthesis import Synthesizer
import pandas as pd
import numpy as np
import random as random
import csv
import pickle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D,MaxPooling2D, Flatten
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score
from IPython.display import Image  
import pydotplus
from ast import literal_eval

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

num_of_features_variation = [20] #10, 20, 30, 40, 50, 75, 100]
class_variation = [20] #2,5,10,20,50, 75, 100]

#########################################################
#### LOAD DATA AND DO PREPROCESSING FOR TARGET MODEL ####
#########################################################

for num_of_features in num_of_features_variation:
    for num_of_classes in class_variation:
        results = open("results/purchase/RESULTS.txt","a") 

        results.write("\n\n************************************************\n\nResults for "+str(num_of_features)+" features and "+str(num_of_classes)+" classes\n")

        #Loading dataset
        data = pd.read_csv("data/purchase/"+str(num_of_features)+"f/purchase_"+str(num_of_features)+"f_"+str(num_of_classes)+"c.csv", index_col=0)
        Y = np.array(data['Label'])
        del data['Label']
        data = data.astype(bool)
        X = np.array(data)

        #Splitting of dataset into Training set and testing set (80% and 20% respectively)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        ############################
        #### TRAIN TARGET MODEL ####
        ############################

        #Single layer architecture
        model = Sequential()
        model.add(Dense(32,input_shape=(X_train.shape[1],)))
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
        model.add(Dense(num_of_classes))
        model.add(Activation('softmax'))
        model.compile(loss='sparse_categorical_crossentropy',
                    optimizer="sgd",metrics=['accuracy'])
        model.fit(X_train, Y_train, batch_size=128, epochs=100, verbose=1, validation_data=(X_test, Y_test))

        loss, accuracy = model.evaluate(X_test,Y_test, verbose=0)
        predictionsNN = [np.argmax(x) for x in model.predict(np.array(X_test))]
        print("\n\nAccuracy of Neural Network (Target BB Model): %s%%\n" % str(accuracy*100.0))
        results.write("\n\nAccuracy of Neural Network (Target BB Model): %s%%\n" % str(accuracy*100.0))
        
        # Train Random Forest
        RF = RandomForestClassifier(n_estimators=(num_of_classes+num_of_features))
        RF.fit(X_train, Y_train)
        estimator = RF

        ## Test accuracy and similarity
        predictionsRF = RF.predict(X_test)
        print("Accuracy of RF (trained and tested on original data): %s%%\n" % (100*accuracy_score(Y_test, predictionsRF)))
        print("Classification similarity of NN and RF trained on target model dataset: %s%%\n" % (np.sum(predictionsRF==predictionsNN)*1.0/len(predictionsRF)*100))
        results.write("Accuracy of RF (trained and tested on original data): %s%%\n" % (100*accuracy_score(Y_test, predictionsRF)))
        results.write("Classification similarity of NN and RF trained on target model dataset: %s%%\n" % (np.sum(predictionsRF==predictionsNN)*1.0/len(predictionsRF)*100))
    
        ###########################
        ##### SYNTHESIZE DATA #####
        ###########################

        print "Intializing Training Data Synthesis\n"

        def purchaseRandomizeFunction(k, x, c):
            x_temp = [random.randint(0, 1) for iter in range(num_of_features)]
            if len(x) == 0:
                return x_temp
            selected_features = random.sample(range(len(x_temp)), k)
            for argfeature in selected_features:
                x[argfeature] = x_temp[argfeature]
            return x

        def purchasePredictProb(x):
            return list(model.predict(np.array([x]))[0])



        synthesizer = Synthesizer(num_of_classes, num_of_features, 1, 100, 0.7, 100,
                                purchaseRandomizeFunction, purchasePredictProb)
                                #c, kmax, kmin, iter_max, conf_min, rej_max,
                                #randomizeFunction, predictProb

        synthesizer.synthesize(1000, "data/purchase/"+str(num_of_features)+"f/synthesized_data/purchase_"+str(num_of_features)+"f_"+str(num_of_classes)+"c_synthesized_data.csv")

        #####################################################
        #### CREATE SHADOW MODEL IN FORM OF DECISION TREE ###
        #####################################################

        ## Load Data
        dataset = pd.read_csv("data/purchase/"+str(num_of_features)+"f/synthesized_data/purchase_"+str(num_of_features)+"f_"+str(num_of_classes)+"c_synthesized_data.csv",  header=None)
        array = dataset.values
        X = array[:,0:len(dataset.columns)-1]
        Y = array[:,len(dataset.columns)-1]
        validation_size = 0.20
        seed = 7
        X_train, X_validation, Y_train, Y_validation = train_test_split(X,Y,
            test_size=validation_size,random_state=seed)

        ## Create Random Forest trained on synthesized data
        RF = RandomForestClassifier(n_estimators=(num_of_classes+num_of_features))
        RF.fit(X_train, Y_train)
        estimator = RF

        ## Test accuracy of Decision Tree trained on synthesized and tested on synthesized
        predictions = RF.predict(X_validation)
        print("Classification similarity of NN and Random Forest trained on synthesized dataset: %s%%\n" % (100*accuracy_score(Y_validation, predictions)))
        results.write("Classification similarity of NN and Random Forest trained on synthesized dataset: %s%%\n" % (100*accuracy_score(Y_validation, predictions)))

        ## Test accuracy of Decision Tree trained on synthesized and tested on original
        predictions = RF.predict(X_test)
        print("Accuracy of Random Forest (trained on synthesized and tested on original): %s%%\n" % (100*accuracy_score(Y_test, predictions)))
        results.write("Accuracy of Random Forest (trained on synthesized and tested on original): %s%%\n" % (100*accuracy_score(Y_test, predictions)))

        results.close()