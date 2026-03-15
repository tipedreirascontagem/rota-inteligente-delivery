import pandas as pd
from sklearn.cluster import KMeans

data = {
    "latitude": [-15.78, -15.79, -15.80, -15.81, -15.77],
    "longitude": [-47.92, -47.91, -47.93, -47.90, -47.89]
}

df = pd.DataFrame(data)

kmeans = KMeans(n_clusters=2)

df["grupo"] = kmeans.fit_predict(df)

print(df)
