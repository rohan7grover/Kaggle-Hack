import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier

if len(sys.argv) < 2:
    print("Usage: myprogram.py train.csv [test.csv]")
    sys.exit(1)

train_file = sys.argv[1]
test_file = sys.argv[2] if len(sys.argv) == 3 else None

data = pd.read_csv(train_file)
X = data.drop('target', axis=1)
y = data['target']

if test_file:
    test_data = pd.read_csv(test_file)
    X_train, y_train = X, y
    X_test = test_data.drop('target', axis=1)
    y_test = test_data['target']
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42)

base_models = [("random_forest", RandomForestClassifier(random_state=10)),
               ("gradient_boosting", GradientBoostingClassifier(random_state=5)),
               ("ada_boost", AdaBoostClassifier(random_state=2)),
               ("ridge", RidgeClassifier(random_state=0)),
               ("knn", KNeighborsClassifier()),
               ("extra_trees", ExtraTreesClassifier(random_state=42)),
               ("gradient_boosting2", GradientBoostingClassifier(random_state=29)),
               ("lgbm", LGBMClassifier(random_state=7))]

meta_model = LogisticRegression(solver='lbfgs', max_iter=500)

pipeline = StackingClassifier(estimators=base_models,
                              final_estimator=meta_model,
                              cv=5)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"{accuracy:.4f}")
