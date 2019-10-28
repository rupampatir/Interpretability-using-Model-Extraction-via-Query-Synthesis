import pandas as pd
import random as r
from sklearn.cluster import KMeans

## INITIALISATION OF VARIABLES ##
num_of_features_variation = [10, 20, 30, 40, 50, 75, 100]
class_variation = [2,5,10,20,50, 75, 100]

## Read dataset ##
data = pd.read_csv("data/purchase/10M/xaa")
#categories = list(data['category'].value_counts()[0:600].index.values) #if using most frequently transacted products
for num_of_features in num_of_features_variation:
    print "num of features ",
    print num_of_features
    categories = r.sample(list(data['category'].value_counts().index.values), num_of_features)
    category_dict = {}
    for i in range(num_of_features):
        category_dict[categories[i]] = i
    data["category"] = data["category"].replace(category_dict)
    customers = list(data["id"].unique())
    df = []
    for customer in customers:
        cust_trans = data[data["id"] == customer]
        cust_products = list(cust_trans["category"].unique())
        temp = [0 for i in range(num_of_features)]
        for product in cust_products:
            if product < num_of_features:
                temp[product] = 1
        df.append(temp)

    # Cluster using k-means for every number of class labels
    for num_of_classes in class_variation:
        data_cust_vs_prod = pd.DataFrame(df)
        kmeans = KMeans(n_clusters=num_of_classes)
        kmeans.fit(data_cust_vs_prod)
        y_kmeans = kmeans.predict(data_cust_vs_prod)
        data_cust_vs_prod["Label"] = y_kmeans
        data_cust_vs_prod.to_csv("purchase_"+str(num_of_features)+"f_"+str(num_of_classes)+"c.csv")