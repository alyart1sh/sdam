import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, matthews_corrcoef
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('sdam.csv')
df_stabi = pd.read_csv('stabi.csv')
label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Label'])
df_stabi['Label'] = label_encoder.transform(df_stabi['Label'])
df_c = pd.concat([df, df_stabi], ignore_index=True)
print(df_c.isnull().sum())
print(df_c['PolyPhen2_class'].value_counts())
df_c['PolyPhen2_score'].plot(kind='hist',bins=20)
plt.show()
df_c = df_c.fillna(df_c[['RASA','PolyPhen2_score', 'mCSM', 'ICM']].median())
df_c = df_c.fillna(df_c[['PolyPhen2_class','TAPASS', 'ArchCandy']].mode().iloc[0])
df_c = pd.get_dummies(df_c, columns=['PolyPhen2_class'])
df_c = pd.get_dummies(df_c, columns=['TAPASS'])
df_c = pd.get_dummies(df_c, columns=['ArchCandy'])
X = df_c.drop(columns=['Label', 'Mutation'])
y = df_c['Label']
n = 75
results = []
for i in range(n):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, )
    model = ExtraTreesClassifier(n_estimators=100,random_state=64)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_prob)
    mcc = matthews_corrcoef(y_test, y_pred)

    results.append({'F1 ': f1, 'Accuracy': accuracy, 'AUROC': auroc, 'MCC': mcc})

results_df = pd.DataFrame(results)

print(results_df.describe())
plt.figure(figsize=(12, 6))
sns.violinplot(data=results_df, inner="box",linewidth=1)
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()
