from sklearn.linear_model import LogisticRegression

def build_model():
    return LogisticRegression(tol=0.1, solver="lbfgs")

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model
