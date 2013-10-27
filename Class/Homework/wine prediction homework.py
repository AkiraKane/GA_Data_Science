




##### Wine
    
    
    
wine = pd.read_csv('C:\Wine.csv')
X = wine.iloc[:,1:]
y =wine.iloc[:,0]

X_transposed = X.T
row_means = [np.mean(i) for i in X_transposed]
data_transposed_scaled = np.array([X_transposed[i][0] - row_means[i] for i in range(4)])

pca = PCA()
pca.fit(data_transposed_scaled)


variance = pca.explained_variance_ratio_
readable_variance = variance * (1/variance[0])
plt.plot(range(4), readable_variance)
plt.show()


colors = ('red', 'blue', 'green', 'orange')
for i in range(4):
    plt.plot(range(100), pca.components_[i], c=colors[i])
    