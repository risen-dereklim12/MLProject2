from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot
import matplotlib.pyplot as plt
import pickle
import os.path

class DecisionTree:
    def __init__(self, n_trees=100, max_depth=7):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.dt = DecisionTreeClassifier(criterion='gini')

    def fit(self, X_train, y_train, X_test, y_test):
        self.dt.fit(X_train, y_train)
        print("score :",self.dt.score(X_test, y_test))
        print("depth :",self.dt.get_depth())
        print("leaves :",self.dt.get_n_leaves())

    def cross_val(self, X_train, y_train, k=5):
        score = []
        for d in range(2,16):
            dt= DecisionTreeClassifier(max_depth=d, criterion = 'gini')
            val_scores = cross_val_score(dt, X_train, y_train, cv=5)
            score.append(val_scores.mean())
        pyplot.plot(list(range(2,16)),score)
        pyplot.legend(['CV score'], loc='upper left')
        pyplot.show()

    def predict(self, X_test, y_test):
        y_pred = self.dt.predict(X_test) #this is the trained decision tree
        print("score :", self.dt.score(X_test,y_test))
        loaded_model_score = 0.0
        if os.path.isfile('models/decision_tree.pkl'):
                with open('models/decision_tree.pkl', 'rb') as file:
                    loaded_model = pickle.load(file)
                    loaded_model_score = loaded_model.score(X_test,y_test)
        if self.dt.score(X_test,y_test) > 0.7 and loaded_model_score < self.dt.score(X_test,y_test):
            with open('models/decision_tree.pkl', 'wb') as file:
                pickle.dump(self.dt, file)
        else:
            print('Previous model is better than the current model. Not saving the current model.')
        cr = classification_report(y_test,y_pred)
        print (cr)
        return y_pred
    
    def confusion_matrix(self, X_test, y_test):
        disp = ConfusionMatrixDisplay.from_estimator(self.dt, X_test, y_test, cmap=plt.cm.Blues)
        print(disp.confusion_matrix)
        plt.show()
