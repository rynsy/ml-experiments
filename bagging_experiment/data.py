from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

test_ids = test.id
test.drop(["id"], axis=1, inplace=True)

x, y = train.drop(["species", "id"], axis=1), train["species"]
le = LabelEncoder().fit(y)
y = le.transform(y)

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=59)

clf = BaggingClassifier(base_estimator=RandomForestClassifier(), n_estimators=100)
# clf = clf.fit(x_train, y_train)
# print("Score: ", clf.score(x_test, y_test))
# print("Accuracy: ", cross_val_score(clf, x, y, cv=10).mean())

print("Train shape ", x.shape)
print("test shape ", test.shape)

clf = clf.fit(x,y)

predictions = clf.predict_proba(test)
p_df = pd.DataFrame(predictions, columns=list(le.classes_))
p_df.insert(0, "id", test_ids)
p_df.reset_index()
p_df.to_csv('submit.csv', index=False)
print(p_df.head())