from sklearn.datasets import load_wine
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

DATA = load_wine()
df = pd.DataFrame(DATA['data'], columns=DATA['feature_names'])
y = DATA['target']  # cultivator

def train_model(X, y):

    # print(DATA['DESCR'])

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    print(f'Accuracy: {model.score(X, y)}')
    return model

if __name__ == '__main__':
    train_model(df, y)
