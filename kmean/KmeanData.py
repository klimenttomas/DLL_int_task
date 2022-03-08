from sklearn.cluster import KMeans as KMeansModel


class DataClass:

    def __init__(self):
        self.__kmeans_model: KMeansModel = None             # K-means algorithm model
        self.__clusters = None                              # Customer affiliation to clusters

    # Property decorators ----------------------------------------------------------------------------------------------
    @property
    def kmeans_model(self) -> KMeansModel:
        return self.__kmeans_model

    @kmeans_model.setter
    def kmeans_model(self, value: KMeansModel):
        self.__kmeans_model = value

    @property
    def clusters(self):
        return self.__clusters

    @clusters.setter
    def clusters(self, value):
        self.__clusters = value
