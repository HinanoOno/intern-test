import numpy as np
import pandas as pd
from pandas import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt

##決定木
from sklearn.datasets import fetch_covtype
dataset=fetch_covtype()
x=dataset.data
y=dataset.target
df = DataFrame(x,columns = dataset.feature_names).assign(cover_type=np.array(y))
from sklearn.tree import DecisionTreeClassifier # 決定木（分類）
from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1234)
model = DecisionTreeClassifier(random_state = 1234)
model.fit(X_train,y_train)
model.score(X_test, y_test)

#GridSearch
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier # 決定木（分類）
import warnings # 実行に関係ない警告を無視
warnings.filterwarnings('ignore')
parameters = {'criterion':['gini', 'entropy'], 'max_depth':[i for i in range(1, 11)],'max_features':['auto','sqrt','log2'], 'min_samples_leaf':[i for i in range(1, 11)],'random_state':[1234]} # ここを編集する
model = DecisionTreeClassifier(random_state=1234)
clf = GridSearchCV(model, parameters, cv=3)
clf.fit(X_train, y_train)
print(clf.best_params_, clf.best_score_)

#ベストなパラメータでモデル構築
clf = DecisionTreeClassifier(**clf.best_params_)
clf.fit(X_train, y_train)

#精度検証
clf.score(X_test,y_test)

#特徴量選択
#特徴量選択のためにインポート
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
estimator = LassoCV(cv = 10, random_state = 1234)
sfm = SelectFromModel(estimator, threshold = 1e-5)
sfm.fit(X_train,y_train)
#選択された特徴量で訓練データを上書き
X_train_selected = sfm.transform(X_train)
X_test_selected = sfm.transform(X_test)
clf = DecisionTreeClassifier(random_state=1234)
clf.fit(X_train_selected,y_train)

#テストデータで精度確認
clf.score(X_test_selected, y_test)

#ランダムフォレスト
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=10, random_state=0)
clf.fit(X_train, y_train)
print("score=", clf.score(X_test, y_test))

#訓練データの中から1.5割をサンプリング
#SVMのためにインポート
from sklearn.svm import SVC
X_train_sample = pd.DataFrame(X_train).sample(frac = 0.15, random_state=1234)
y_train_sample = pd.DataFrame(y_train).sample(frac = 0.15, random_state=1234)

#モデル構築
clf = SVC(random_state=1234)
clf.fit(X_train_sample, y_train_sample) 
clf.score(X_test,y_test)

#標準化のためにインポート
from sklearn.preprocessing import StandardScaler
#標準化
stdsc = StandardScaler()
X= stdsc.fit_transform(x)
#y = stdsc.fit_transform(y)

#データ分割

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
X_train_sample = pd.DataFrame(X_train).sample(frac = 0.15, random_state=1234)
y_train_sample = pd.DataFrame(y_train).sample(frac = 0.15, random_state=1234)

#モデル構築
clf = SVC(random_state=1234)
clf.fit(X_train_sample, y_train_sample)

#精度評価
clf.score(X_test, y_test)

#特徴量選択
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
import warnings
estimator = LassoCV( cv = 10, random_state = 1234)
sfm = SelectFromModel(estimator, threshold = 1e-5)
sfm.fit(X_train,y_train)

#選択された特徴量で訓練データを上書き
X_train_selected = sfm.transform(X_train)
X_test_selected = sfm.transform(X_test)

X_train_grid = pd.DataFrame(X_train_selected).sample(frac = 0.03,random_state=1234)
y_train_grid = pd.DataFrame(y_train).sample(frac = 0.03,random_state=1234)

#ハイパーパラメータチューニング
parameters = {'kernel':['linear', 'rbf'], 'C':[0.001, 0.01,0.1,1,10]} # ここを編集する
model = SVC(random_state=1234)
clf = GridSearchCV(model, parameters, cv=2,return_train_score=False)
clf.fit(X_train_grid, y_train_grid)
print(clf.best_params_, clf.best_score_)

X_train_sample = pd.DataFrame(X_train_selected).sample(frac = 0.15, random_state=1234)
y_train_sample = pd.DataFrame(y_train).sample(frac = 0.15, random_state=1234)

clf = SVC(**clf.best_params_,random_state=1234)
clf.fit(X_train_sample, y_train_sample) 

#テストデータで精度評価
clf.score(X_test_selected, y_test)
