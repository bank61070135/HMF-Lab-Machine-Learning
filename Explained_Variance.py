import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

Dataset = pd.read_excel('Data.xlsx')

y = Dataset[['Glucose con', '5-HMF yield', 'Select']]
x = Dataset.drop(['Catalysts', 'Glucose con', '5-HMF yield', 'Select'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

n = min(x_train.shape[0], x_train.shape[1])
pca = PCA(n_components = n)

x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

explained_var = pca.explained_variance_ratio_*100
cumulative_explained_var = explained_var.cumsum()

plt.figure(figsize=(15, 10))
plt.bar(range(1, n+1), explained_var, color='b', alpha=0.7, label='Individual Explained Variance')
plt.plot(range(1, n+1), cumulative_explained_var, marker='o', linestyle='-', color='r', label='Cumulative Explained Variance')
for i, txt in enumerate(cumulative_explained_var):
    plt.annotate(f'{txt:.2f}%', (i + 1, txt), textcoords="offset points", xytext=(0,10), ha='center')
plt.title('Individual and Cumulative Explained Variance')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance (%)')
plt.xticks(range(1, n+1))
plt.legend()
plt.grid()
plt.savefig('explained_variance_plot.png')
plt.show()