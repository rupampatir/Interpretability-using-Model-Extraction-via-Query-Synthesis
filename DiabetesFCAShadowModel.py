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
import eli5
from eli5.sklearn import PermutationImportance
import pydotplus
from tabulate import tabulate
from rulefit import RuleFit
from sklearn.ensemble import RandomForestRegressor


#ignore warnings
import warnings
warnings.filterwarnings('ignore')
results = open("results/diabetes/RESULTS.txt","a") 

#########################################################
#### LOAD DATA AND DO PREPROCESSING FOR TARGET MODEL ####
#########################################################

#Loading dataset
data = pd.read_csv('data/diabetes/diabetes.csv')
Y = np.array(data['Outcome'])
del data['Outcome']
data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
data['Glucose'].fillna(data['Glucose'].mean(), inplace = True)
data['BloodPressure'].fillna(data['BloodPressure'].mean(), inplace = True)
data['SkinThickness'].fillna(data['SkinThickness'].median(), inplace = True)
data['Insulin'].fillna(data['Insulin'].median(), inplace = True)
data['BMI'].fillna(data['BMI'].median(), inplace = True)
X = np.array(data)
column_names = data.columns
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
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='sparse_categorical_crossentropy',
              optimizer="sgd",metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=128, epochs=100, verbose=1, validation_data=(X_test, Y_test))

with open('diabetes_classifier_NN', 'wb') as f:
    pickle.dump(model, f)

loss, accuracy = model.evaluate(X_test,Y_test, verbose=0)
predictionsNN = [np.argmax(x) for x in model.predict(np.array(X_test))]
print("\n\nAccuracy of Neural Network (Target BB Model): %s%%\n" % str(accuracy*100.0))
results.write("\n\nAccuracy of Neural Network (Target BB Model): %s%%\n" % str(accuracy*100.0))


dtree = DecisionTreeClassifier(random_state=0)
dtree.fit(X_train,[np.argmax(x) for x in model.predict(np.array(X_train))])
predictionsDT = dtree.predict(X_test)
print("Similarity of NN and DT (trained on original data and NN predictions): %s%%\n" % (np.sum(predictionsDT==predictionsNN)*1.0/len(predictionsDT)*100))
results.write("Similarity of NN and DT (trained on original data and NN predictions): %s%%\n" % (np.sum(predictionsDT==predictionsNN)*1.0/len(predictionsDT)*100))

## Test accuracy and similarity

dtree = DecisionTreeClassifier(random_state=0)
dtree.fit(X_train,Y_train)
predictionsDT = dtree.predict(X_test)
print("Accuracy of DT (trained and tested on original data): %s%%\n" % (100*accuracy_score(Y_test, predictionsDT)))
results.write("Accuracy of DT (trained and tested on original data): %s%%\n" % (100*accuracy_score(Y_test, predictionsDT)))
print("Classification similarity of NN and DT trained on target model dataset: %s%%\n" % (np.sum(predictionsDT==predictionsNN)*1.0/len(predictionsDT)*100))
results.write("Classification similarity of NN and DT trained on target model dataset: %s%%\n" % (np.sum(predictionsDT==predictionsNN)*1.0/len(predictionsDT)*100))

#Visualisation
dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data, 
                feature_names=['Pregnancies', 'Glcose', 'BloodPressre', 'SkinThickness',
       'Inslin', 'BMI', 'DiabetesPedigreeFnction', 'Age'],
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.write_png("visualisations/diabetes/diabetes_dtree_trained_original.png"))

###########################
##### SYNTHESIZE DATA #####
###########################

print "Intializing Training Data Synthesis\n"

def diabetesRandomizeFunction(k, x, c):
    x_pregnancies = random.randint(0, 17)
    x_glucose = random.randint(0, 200)
    x_blood_pressure = random.randint(0, 130)
    x_skin_thickness = random.randint(0, 100)
    x_insulin = random.randint(0, 900)
    x_bmi = round(random.uniform(15, 70),2)
    x_dpf = round(random.uniform(0, 3),2)
    x_age = random.randint(20, 100)
    x_temp = [x_pregnancies, x_glucose, x_blood_pressure, x_skin_thickness, x_insulin, x_bmi, x_dpf, x_age]

    if len(x) == 0:
        return x_temp

    selected_features = random.sample(range(len(x_temp)), k)

    for argfeature in selected_features:
        x[argfeature] = x_temp[argfeature]

    return x

def diabetesPredictProb(x):
    return list(model.predict(np.array([x]))[0])


synthesizer = Synthesizer(2, 8, 1, 100, 0.9, 100,
                          diabetesRandomizeFunction, diabetesPredictProb)
                          #c, kmax, kmin, iter_max, conf_min, rej_max,
                          #randomizeFunction, predictProb

synthesizer.synthesize(1000, "data/diabetes/diabetes_synthesized_data.csv")


#####################################################
#### CREATE SHADOW MODEL IN FORM OF DECISION TREE ###
#####################################################

## Load Data
dataset = pd.read_csv("data/diabetes/diabetes_synthesized_data.csv",  header=None)
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
                feature_names=column_names,
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.write_png("visualisations/diabetes/diabetes_dtree_trained_synthesized.png"))


##################################################################################
#### CREATE SHADOW MODEL IN FORM OF RANDOM FOREST AND RUN PERMUATION REGRESSOR ###
##################################################################################

## Random Forest Regressor for permutation importance

rf = RandomForestRegressor(n_estimators = 100,
                           n_jobs = -1,
                           oob_score = True,
                           bootstrap = True,
                           random_state = 42)
rf.fit(X_train, Y_train)

perm = PermutationImportance(rf, random_state=1).fit(X_validation,Y_validation)
results.write('\n\n\nRANDOM FOREST REGRESSOR PERMUTATION IMPORTANCE\n\n\n')
print( eli5.format_as_text(eli5.explain_weights(perm, feature_names = data.columns.tolist())))

results.write( eli5.format_as_text(eli5.explain_weights(perm, feature_names = data.columns.tolist())))

##########################################################
#### CREATE SHADOW MODEL IN FORM OF RULE FIT ALGORITHM ###
##########################################################

rf = RuleFit()
rf.fit(X_train,[int(i) for i in Y_train], feature_names=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])
rules = rf.get_rules()
rules = rules[rules.coef != 0].sort_values("support", ascending=False)
print('\n\nRule Fit aglorithm rules\n'+str(rules))
results.write('\n\nRule Fit aglorithm rules\n'+str(rules))

##############################################################
#### CREATE SHADOW MODEL IN FORM OF Formel Concept Lattice ###
##############################################################

from concepts import Context

## Reading Data
dataset = pd.read_csv('data/diabetes/diabetes_synthesized_data.csv')
dataset.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
y = dataset["Outcome"]
dataset = dataset.drop(["Outcome"],axis = 1)

## Test/Train split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(dataset,y,test_size=0.3,random_state=42, stratify=y)

model = None
with open('diabetes_classifier_NN', 'rb') as f:
    model = pickle.load(f)

## Create one-hot-encodings
def create_diabetes_one_hot_encoding(data):
    diabetes_data = data.copy(deep = True)
    diabetes_data.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    diabetes_data['BP1'] = 0
    diabetes_data.loc[diabetes_data.BloodPressure < 60, 'BP1'] = 1
    diabetes_data['BP2'] = 0
    diabetes_data.loc[diabetes_data.BloodPressure.between(60,90,True), 'BP2'] = 1
    diabetes_data['BP3'] = 0
    diabetes_data.loc[diabetes_data.BloodPressure > 90, 'BP3'] = 1
    diabetes_data = diabetes_data.drop(["BloodPressure"],axis = 1)
    diabetes_data['BMI1'] = 0
    diabetes_data.loc[diabetes_data.BMI < 25, 'BMI1'] = 1
    diabetes_data['BMI2'] = 0
    diabetes_data.loc[diabetes_data.BMI.between(25,30, True), 'BMI2'] = 1
    diabetes_data['BMI3'] = 0
    diabetes_data.loc[diabetes_data.BMI>30, 'BMI3'] = 1
    diabetes_data = diabetes_data.drop(["BMI"],axis = 1)
    diabetes_data['Insulin1'] = 0
    diabetes_data.loc[diabetes_data.Insulin < 16, 'Insulin1'] = 1
    diabetes_data['Insulin2'] = 0
    diabetes_data.loc[diabetes_data.Insulin.between(16,166,True), 'Insulin2'] = 1
    diabetes_data['Insulin3'] = 0
    diabetes_data.loc[diabetes_data.Insulin > 166, 'Insulin3'] = 1
    diabetes_data = diabetes_data.drop(["Insulin"],axis = 1)
    diabetes_data['Glucose1'] = 0
    diabetes_data.loc[diabetes_data.Glucose < 140, 'Glucose1'] = 1
    diabetes_data['Glucose2'] = 0
    diabetes_data.loc[diabetes_data.Glucose.between(140,200, True), 'Glucose2'] = 1
    diabetes_data['Glucose3'] = 0
    diabetes_data.loc[diabetes_data.Glucose > 200, 'Glucose3'] = 1
    diabetes_data = diabetes_data.drop(["Glucose"],axis = 1)
    diabetes_data['SkinThickness1'] = 0
    diabetes_data.loc[diabetes_data.SkinThickness < 23, 'SkinThickness1'] = 1
    diabetes_data['SkinThickness2'] = 0
    diabetes_data.loc[diabetes_data.SkinThickness > 23, 'SkinThickness2'] = 1
    diabetes_data['SkinThickness3'] = 0
    diabetes_data.loc[diabetes_data.SkinThickness == 23, 'SkinThickness3'] = 1
    diabetes_data = diabetes_data.drop(["SkinThickness"],axis = 1)
    diabetes_data['Pregnancy1'] = 0
    diabetes_data.loc[diabetes_data.Pregnancies < 1, 'Pregnancy1'] = 1
    diabetes_data['Pregnancy2'] = 0
    diabetes_data.loc[diabetes_data.Pregnancies.between(1, 3, True), 'Pregnancy2'] = 1
    diabetes_data['Pregnancy3'] = 0
    diabetes_data.loc[diabetes_data.Pregnancies>3, 'Pregnancy3'] = 1
    diabetes_data = diabetes_data.drop(["Pregnancies"],axis = 1)
    diabetes_data['Age1'] = 0
    diabetes_data.loc[diabetes_data.Age <20 , 'Age1'] = 1
    diabetes_data['Age3'] = 0
    diabetes_data.loc[diabetes_data.Age > 60, 'Age3'] = 1
    diabetes_data['Age2'] = 0
    diabetes_data.loc[diabetes_data.Age.between(20,60, True), 'Age2'] = 1
    diabetes_data = diabetes_data.drop(["Age"],axis = 1)
    diabetes_data['DPF1'] = 0
    diabetes_data.loc[diabetes_data.DiabetesPedigreeFunction <0.3 , 'DPF1'] = 1
    diabetes_data['DPF2'] = 0
    diabetes_data.loc[diabetes_data.DiabetesPedigreeFunction.between(0.3, 0.6, True) , 'DPF2'] = 1
    diabetes_data['DPF3'] = 0
    diabetes_data.loc[diabetes_data.DiabetesPedigreeFunction >0.6 , 'DPF3'] = 1
    diabetes_data = diabetes_data.drop(["DiabetesPedigreeFunction"],axis = 1)
    return  diabetes_data

X_train_one_hot = create_diabetes_one_hot_encoding(X_train)
X_test_one_hot = create_diabetes_one_hot_encoding(X_test)

# Creating and save context for implication rules
X_train_one_hot['Class'] = y_train
X_train_Class_split = pd.concat([X_train_one_hot,pd.get_dummies(X_train_one_hot['Class'], prefix='Class')],axis=1)
X_train_Class_split = X_train_Class_split.drop(["Class"], axis=1).drop_duplicates()
objects = X_train_Class_split.index.values
objects = [str(oi) for oi in objects] 
properties = X_train_Class_split.columns.values
bools = list(X_train_Class_split.astype(bool).itertuples(index=False, name=None))
cxt = Context(objects, properties, bools)
cxt.tofile('diabetes_context.cxt', frmat='cxt', encoding='utf-8')

## Create concepts lattices for each class
c = {}
l = {}
no_of_classes = 2
X_train_one_hot['Class'] = y_train
X_train_one_hot = X_train_one_hot.drop_duplicates()

for i in range(0, no_of_classes):
    X_temp =X_train_one_hot.copy(deep = True)
    X_temp = X_temp[X_temp['Class'] == i].drop(["Class"],axis = 1)
    objects = X_temp.index.values
    objects = [str(oi) for oi in objects] 
    properties = X_temp.columns.values
    bools = list(X_temp.astype(bool).itertuples(index=False, name=None))
    c[i] = Context(objects, properties, bools)
    l[i] = c[i].lattice

## Render Concept Lattice Graphs
"""for lat in range(len(l)):
    print  'creating lattice ' + str(lat)
    l[lat].graphviz().render('lattice'+str(lat)+'.gv', view=True)"""

## Remove intersections
class_intents_sets = []
for i in range(0, no_of_classes):
    set_temp = set()
    for extent, intent in l[i]:
        set_temp.add(intent)
    class_intents_sets.append(set_temp)

## Define FCA classifier
def predict_fca(s):
    properties = s.index.values
    objects = [str(s.name)]
    bools = tuple(s.astype(bool))
    s_lattice = Context(objects, properties, [bools])
    s_intents = set()
    for extent_s, intent_s in s_lattice.lattice:
        s_intents.add(intent_s)
    sets = set(list(s_intents)[1])
    probs = []
    for i in range(0, no_of_classes):
        for intent_c in class_intents_sets[i]:
            setc = set(intent_c)
            if sets.issubset(setc):
                probs.append(i)
    if len(probs) == 0:
        return -1
    return max(probs,key=probs.count)

## Calculate accuracy of fca classification = 0
correct = 0
attempted = 0
for si in range(0, len(X_test_one_hot)):
    s = X_test_one_hot.iloc[si]
    p = predict_fca(s)
    if (p == y_test.iloc[si]):
        correct+=1
    attempted+=1

print 'Fidelity of FCA shadow model: ', round(float(correct)/attempted*100.0, 2), '%'
results.write( 'Fidelity of FCA shadow model: ' + str( round(float(correct)/attempted*100.0, 2)) + '%')

print 'Determined by FCA classification: ', round(float(attempted)/len(X_test_one_hot)*100.0, 2), '%'
results.write('\n\nDetermined by FCA classification: ' + str(round(float(attempted)/len(X_test_one_hot)*100.0, 2)) + '%')

## Explanation Generators

def modifySample(si, feature):
    original = X_test_one_hot.iloc[si].copy(deep=True)
    features_to_modify = [original[feature * 3], original[(feature * 3) + 1], original[(feature * 3) + 2]]
    randi = random.randint(0,2)
    while features_to_modify[randi] == 1:
        randi = random.randint(0,2)
    original[feature * 3] = 0
    original[(feature * 3) + 1] = 0
    original[(feature * 3) + 2] = 0
    original[(feature * 3) + randi] = 1
    return original

def explanation_generator(samples):
    Pml = []
    Pfca = []
    E = []
    for si in range(0, len(samples)):
        s = samples.iloc[si]
        p = predict_fca(X_test_one_hot.iloc[si])
        Pfca.append(p)
        Pml.append(np.argmax(model.predict(np.array([samples.iloc[si]]))))
        for feature in range(0, len(s)):
            temp_s = modifySample(si, feature)
            p = predict_fca(temp_s)
            if Pml[-1] == Pfca[-1]:
                if p != Pfca[-1]:
                    E.append([feature, 1, Pml[-1]])
                else:
                    E.append([feature, 0, Pml[-1]])
            else:
                if p == Pfca[-1]:
                    E.append([feature, 1, Pfca[-1]])
                else:
                    E.append([feature, 0, Pfca[-1]])
    return E

print '\nGenerating Explanations'
E = explanation_generator(X_test)

features = ['BloodPressure', 'BMI', 'Insulin', 'Glucose',  'SkinThickness', 'Pregnancies', 'Age', 'DiabetesPedigreeFunction']
table = []
records_diabetes = [item for item in E if item[2] == 1]
records_no_diabetes = [item for item in E if item[2] == 0]

for prop in range(0, len(features)):
    prop_records_diabetes = [item for item in records_diabetes if ((item[0] == prop))]
    prop_records_diabetes_change = len([item for item in prop_records_diabetes if ((item[1] == 1))])
    prop_score_diabetes = round(float(prop_records_diabetes_change)/float(len(prop_records_diabetes))*100,2)
    prop_records_no_diabetes = [item for item in records_no_diabetes if ((item[0] == prop))]
    prop_records_no_diabetes_change = len([item for item in prop_records_no_diabetes if ((item[1] == 1))])
    prop_score_no_diabetes = round(float(prop_records_no_diabetes_change)/float(len(prop_records_no_diabetes))*100,2)
    table.append([features[prop], prop_score_diabetes, prop_score_no_diabetes, float(prop_score_diabetes + prop_score_no_diabetes)/2])

print(tabulate(pd.DataFrame(table).sort_values( by=3, ascending=False).values, headers=['Feature', 'Diabetes', 'No Diabetes', 'Average']))
results.write('\n\nFeature Importance as determined by FCA:\n\n' + tabulate(table, headers=['Feature', 'Diabetes', 'No Diabetes']))
"""
## Check FCA on target model dataset

data = pd.read_csv('data/diabetes/diabetes.csv')
data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
data['Glucose'].fillna(data['Glucose'].mean(), inplace = True)
data['BloodPressure'].fillna(data['BloodPressure'].mean(), inplace = True)
data['SkinThickness'].fillna(data['SkinThickness'].median(), inplace = True)
data['Insulin'].fillna(data['Insulin'].median(), inplace = True)
data['BMI'].fillna(data['BMI'].median(), inplace = True)
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
y = data["Outcome"]
data = data.drop(["Outcome"],axis = 1)

#Splitting of data into Training set and testing set (80% and 20% respectively)
X_train,X_test,y_train,y_test = train_test_split(data,y,test_size=0.3,random_state=42, stratify=y)


predictionsNN_test = [np.argmax(x) for x in model.predict(np.array(X_test))]
y_test = pd.Series(predictionsNN_test, index=X_test.index.values, name="Outcome")
predictionsNN_train = [np.argmax(x) for x in model.predict(np.array(X_train))]
y_train = pd.Series(predictionsNN_train, index=X_train.index.values, name="Outcome")

X_train_one_hot = create_diabetes_one_hot_encoding(X_train)
X_test_one_hot = create_diabetes_one_hot_encoding(X_test)

## Create concepts lattices for each class
c = {}
l = {}
no_of_classes = 2
X_train_one_hot['Class'] = y_train
X_train_one_hot = X_train_one_hot.drop_duplicates()
for i in range(0, no_of_classes):
    X_temp =X_train_one_hot.copy(deep = True)
    X_temp = X_temp[X_temp['Class'] == i].drop(["Class"],axis = 1)
    objects = X_temp.index.values
    objects = [str(oi) for oi in objects] 
    properties = X_temp.columns.values
    bools = list(X_temp.astype(bool).itertuples(index=False, name=None))
    c[i] = Context(objects, properties, bools)
    l[i] = c[i].lattice

## Remove intersections
class_intents_sets = []
for i in range(0, no_of_classes):
    set_temp = set()
    for extent, intent in l[i]:
        set_temp.add(intent)
    class_intents_sets.append(set_temp)

## Calculate accuracy of fca classification = 0
correct = 0
attempted = 0
for si in range(0, len(X_test_one_hot)):
    s = X_test_one_hot.iloc[si]
    p = predict_fca(s)
    if (p == y_test.iloc[si]):
        correct+=1
    attempted+=1

print 'Fidelity of FCA trained on target training data with NNs predictions: ', round(float(correct)/attempted*100.0, 2), '%'
results.write('Fidelity of FCA trained on target training data with NNs predictions: ' + str(round(float(correct)/attempted*100.0, 2)) + '%')
print 'Determined by FCA classification: ', round(float(attempted)/len(X_test_one_hot)*100.0, 2), '%'
results.write('\n\nDetermined by FCA classification: ' + str(round(float(attempted)/len(X_test_one_hot)*100.0, 2)) + '%')
"""