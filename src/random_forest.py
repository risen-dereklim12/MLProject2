from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pickle
import os.path

class RandomForest:
    def __init__(self, val_score =[], train_score = [], trees = [2,5,10,15,20]+list(range(30,110,10))+[200, 300, 400]):
        self.val_score = val_score
        self.train_score = train_score
        self.trees = trees
        self.opt_trees = 0

    def fit(self, X_train, y_train, X_test, y_test):
        for n in self.trees:
            rf = RandomForestClassifier(n_estimators=n)
            rf.fit(X_train,y_train)
            self.train_score.append(rf.score(X_train, y_train))
            self.val_score.append(rf.score(X_test, y_test))
        plt.plot(self.trees, self.train_score,'r-x',label='train')
        plt.plot(self.trees, self.val_score,'g-x', label='val')
        plt.xlabel("No of trees in RF")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()
        ind = self.val_score.index(max(self.val_score))
        self.opt_trees = self.trees[ind]
        print("Optimum number of trees :", self.opt_trees)

    def predict(self, X_train, y_train, X_test, y_test):
        rf = RandomForestClassifier(n_estimators=self.opt_trees)
        rf.fit(X_train,y_train)
        y_pred = rf.predict(X_test)
        print("score :", rf.score(X_test,y_test))
        loaded_model_score = 0.0
        if os.path.isfile('models/random_forest.pkl'):
                with open('models/random_forest.pkl', 'rb') as file:
                    loaded_model = pickle.load(file)
                    loaded_model_score = loaded_model.score(X_test,y_test)
        if rf.score(X_test,y_test) > 0.7 and loaded_model_score < rf.score(X_test,y_test):
            with open('models/random_forest.pkl', 'wb') as file:
                pickle.dump(rf, file)
        else:
            print('Previous model is better than the current model. Not saving the current model.')
        cr = classification_report(y_test,y_pred)
        print (cr)
        return y_pred