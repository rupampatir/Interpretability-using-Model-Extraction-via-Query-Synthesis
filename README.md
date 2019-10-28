# Interpretability Using Model Extraction via Query Synthesis

To run the generator and get the results and visualisations run one of the *ShadowModel.py files depending on the dataset.

The results and visualisations for each dataset are saves in the results and visualisations folders respectively

# Datasets used

Animal: Classifying type of animal (https://www.kaggle.com/uciml/zoo-animal-classification)

Diabetes: Classifying whether the person has diabetes or not (https://www.kaggle.com/uciml/pima-indians-diabetes-database)

Income: Classifying whether a person earns more or less than 50K a year (https://www.kaggle.com/uciml/adult-census-income)

Mobile: Classifying price range (https://www.kaggle.com/iabhishekofficial/mobile-price-classification)

Purchase: 
Applying kNN to label the data. This is a one-hot represntation with columns representing a product and rows representing users. The values are 1 if a user has bought the product, and 0 otherwise. The code to create the one hot representation is in _data/purchase/purchase_create_dataset.py_

(https://www.kaggle.com/c/acquire-valued-shoppers-challenge/data)

For the purchase dataset to reduce the size of the dataset, the following can be used to use 10 million records instead of the whole dataset (the whole dataset is 22GB large!). The following splits the csv without having to open or read it first.

   __split -l 100000000 transactions.csv__

# Link to Research Proposal and Outline: 

Research Proposal: https://docs.google.com/document/d/17_u0Vubl3z87NdKAiQVP9mTQeJL7jZlwAGzJ9hmL41E/edit?usp=sharing
Algoritm Outline: https://docs.google.com/document/d/15U_2DY6j2JXXbOaHWcMmKyFLiQntQ7AXZPQCP7Fxk-k/edit?usp=sharing