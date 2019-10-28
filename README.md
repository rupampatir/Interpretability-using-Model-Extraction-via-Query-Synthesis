# Interpretability Using Model Extraction via Query Synthesis

The modules/ModelExtraction/QuerySynthesisr.py file contains the code implementation of the algorithm in the paper by Shokri et al.

# Datasets used
Animal: Classifying type of animal (https://www.kaggle.com/uciml/zoo-animal-classification)
Diabetes: Classifying whether the person has diabetes or not (https://www.kaggle.com/uciml/pima-indians-diabetes-database)
Income: Classifying whether a person earns more or less than 50K a year (https://www.kaggle.com/uciml/adult-census-income)
Mobile: Classifying price range (https://www.kaggle.com/iabhishekofficial/mobile-price-classification)
Purchase: Applying kNN to label the data. This is a one-hot represntation with columns representing a product and rows representing users. The values are 1 if a user has bought the product, and 0 otherwise. The code to create the one hot representation is in data/purchase/10M/purchase_create_dataset.py (https://www.kaggle.com/c/acquire-valued-shoppers-challenge/data)

To run the generator and get the results and visualisations run one of the __ShadowModel.py files depending on the dataset.

The results and visualisations for each dataset are saves in the results and visualisations folders respectively
