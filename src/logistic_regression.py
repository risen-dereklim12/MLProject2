from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class LogRegression:
    def __init__(self):
        self.itr = 1000

    def fit(self, X_train, y_train, X_test, y_test):
        clf = LogisticRegression(max_iter=self.itr)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("Accuracy: %.2f"
                % accuracy_score(y_pred, y_test))
        if accuracy_score(y_pred, y_test) < 0.7:
            self.scale(X_train, y_train, X_test, y_test)
    
    def scale(self, X_train, y_train, X_test, y_test):
        scaler = StandardScaler()
        X1_train = scaler.fit_transform(X_train)
        X1_test = scaler.transform(X_test)
        clf1 = LogisticRegression(max_iter=self.itr)
        clf1.fit(X1_train, y_train)
        y1_pred = clf1.predict(X1_test)
        print("Scaled Accuracy: %.2f"
            % accuracy_score(y1_pred, y_test))
