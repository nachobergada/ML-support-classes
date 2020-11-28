class FaissKNeighbors:
    '''
    FaissKNeighbors
    FAISS Class to execute Knn algorithms
    https://github.com/j-adamczyk/Towards_Data_Science/issues/1 --> issues with series, dataframes and numpy arrays solved
    '''

    def __init__(self, k=5):
        self.index = None
        self.y = None
        self._y_np = None
        self.k = k

    def fit(self, X, y):
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X.astype(np.float32))
        self.y = y
        self._y_np = np.array(y, dtype=np.int)

    def predict(self, X):
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        # votes = self.y[indices]
        votes = self._y_np[indices]
        predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
        return predictions
