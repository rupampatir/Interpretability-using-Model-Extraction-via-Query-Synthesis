import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pydotplus
from ast import literal_eval
import numpy as np

## Load Data
data_0 = pd.read_csv("training_class_0.csv")
data_0.columns = ['age', 'education', 'marital.status', 'sex', 'capital.gain',
       'capital.loss', 'hours.per.week']
data_0["Label"] = 0
data_1 = pd.read_csv("training_class_1.csv")
data_1.columns = ['age', 'education', 'marital.status', 'sex', 'capital.gain',
       'capital.loss', 'hours.per.week']
data_1["Label"] = 1
dataset = pd.concat([data_0, data_1], ignore_index=True)
"""
#Split categorical values to hot binary features
education = dataset['education']

# get the unique activities
edcols = np.unique([a for a in education])

# assemble the desired one hot array from the activities
edarr = np.array([np.in1d(edcols, al) for al in education])
eddf = pd.DataFrame(edarr, columns=edcols)

# stick the dataframe with the one hot array onto the main dataframe
dataset = pd.concat([dataset.drop(columns='education'), eddf], axis=1)


"""
array = dataset.values
X = array[:,0:len(dataset.columns)-1]
Y = array[:,len(dataset.columns)-1]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X,Y,
    test_size=validation_size,random_state=seed)

## Create Decision Tree
dtree=DecisionTreeClassifier()
dtree.fit(X_train,Y_train)

## Test accuracy
predictions = dtree.predict(X_validation)
print("Accuracy: %s%%" % (100*accuracy_score(Y_validation, predictions)))

dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data, 
                feature_names=dataset.columns[:len(dataset.columns)-1],
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.write_png("dtree.png"))


"""import os
       from sklearn.tree import export_graphviz
       from sklearn import tree
       i_tree = 0
       for tree_in_forest in estimator.estimators_:
       dot_data = StringIO()
       export_graphviz(tree_in_forest, out_file=dot_data, 
                     feature_names=[i for i in range(num_of_features)],
                     filled=True, rounded=True,
                     special_characters=True)
       graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
       Image(graph.write_png("data/purchase/"+str(num_of_features)+"f/visualisations/dtree_trained_synthesized/"+str(num_of_classes)+"c_"+ str(i_tree)+".png"))
       i_tree +=1"""