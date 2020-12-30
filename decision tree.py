import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

filename = "decisiontreedata.csv"
yName = "Drug"

df = pd.read_csv(filename, delimiter = ",")
x_titles = [col for col in df.columns if col != yName]
x = df[x_titles].values

#categorical variables need to be converted to numerical
from sklearn import preprocessing

#the following block generalizes preprocessing to any file via iteration
#the rest of the file should be pretty identical to the lab
for colNum in range(x.shape[1]):
    values = []
    if type(x[:,colNum][0]) not in [int, float]:
        for value in x[:,colNum]:
            if value not in values:
                values.append(value)
        newCol = preprocessing.LabelEncoder()
        newCol.fit(values)
        x[:,colNum] = newCol.transform(x[:,colNum])

y = df[yName]

from sklearn.model_selection import train_test_split

x_trainset, x_testset, y_trainset, y_testset = train_test_split(x, y, test_size=0.3, random_state=3)

newTree = DecisionTreeClassifier(criterion="entropy", max_depth = len(x_titles))

newTree.fit(x_trainset, y_trainset)

predTree = newTree.predict(x_testset)

from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))

#visualize

from  io import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree

dot_data     = StringIO()
filename     = "tree.png"
featureNames = df.columns[0:len(x_titles)]
targetNames  = df[yName].unique().tolist()

out          = tree.export_graphviz(newTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)  
graph        = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img          = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')
