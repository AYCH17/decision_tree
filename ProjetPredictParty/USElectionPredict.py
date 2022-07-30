import graphviz
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import numpy as np

col_names = ["Populations / Million","PIB( %)","Population urbaine en Million",

"Export - biens / services(Billion of US $)" ,   "Import - biens / services(Billion of US $)"  ,
"Natalité brut / 1000 habitants",
"Taux d'emploie",	"Taux de Chomage(%)",
"Paricipation aux elections %",
"Revenu personnel médian réel USD"	,
"Criminalite/100.000Hbt" ,"Parti gouvernant",	"Vainqueur"]

# load dataset
learning_set = pd.read_csv("donnees_entree_csv.csv", header=None, names=col_names)

learning_set.head()

#split dataset in features and target variable
feature_cols = ["Populations / Million","PIB( %)","Population urbaine en Million",

"Export - biens / services(Billion of US $)" ,   "Import - biens / services(Billion of US $)"  ,
"Natalité brut / 1000 habitants",
"Taux d'emploie",	"Taux de Chomage(%)",
"Paricipation aux elections %",
"Revenu personnel médian réel USD"	,
"Criminalite/100.000Hbt" ,"Parti gouvernant"]

X0 = learning_set[feature_cols].values[1:17].astype(np.float) # Features
y0 = learning_set.Vainqueur.values[1:16]# Target variable

X = X0[:, :3]  # we only take the first three features.
y=[0,1,1,0,0,1,0,0,0,1,1,0,0,1,1] #0 for republicans and 1 for democrats

fig = plt.figure(1, figsize=(8, 6))
ax0 = Axes3D(fig, elev=-150, azim=110)
ax0.scatter(X[1:16, 0], X[1:16, 1], X[1:16, 2], c=y,
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax0.set_title("First three directions")
ax0.set_xlabel('population')
ax0.w_xaxis.set_ticklabels([])
ax0.set_ylabel('pib(%)')
ax0.w_yaxis.set_ticklabels([])
ax0.set_zlabel("population urbaine")
ax0.w_zaxis.set_ticklabels([])
# To get a better understanding of interaction of the dimensions
# plot the first three PCA dimensions
fig = plt.figure(2, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(X0)
ax.scatter(X_reduced[1:16, 0], X_reduced[1:16, 1], X_reduced[1:16, 2], c=y
           , edgecolor='k', s=40)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

plt.show()



# Create Decision Tree classifer object
clf = DecisionTreeClassifier()
# Train Decision Tree Classifer
clf = clf.fit(X_reduced[1:16],y0)

dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=['eigenV1','eigenV2','eigenV3'],
                                class_names=['Rep','Democ'],
                                filled=True)

# Draw graph
graph = graphviz.Source(dot_data, format="png")

graph.render("decision_tree")


X_test_reduced = X_reduced[15].reshape(1,3) # le dernier élément est la donnée du test


#Predict the response for test dataset
y_pred = clf.predict(X_test_reduced)

print(y_pred)
