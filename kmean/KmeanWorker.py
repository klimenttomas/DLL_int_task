from preprocess import PreprocessClass
from .KmeanData import DataClass as KmeansOutputClass
from sklearn.cluster import KMeans as KMeansModel


class WorkerClass:

    def __init__(self, data: PreprocessClass):
        self.__data: PreprocessClass = data                                 # Inpute data for cluster algorithm
        self.__kmeans_results: KmeansOutputClass = KmeansOutputClass()      # Output data
        self.__kmeans_model: KMeansModel = None                             # K-means algorithm model

    # Property decorators ----------------------------------------------------------------------------------------------
    @property
    def kmeans_results(self) -> KmeansOutputClass:
        return self.__kmeans_results

    # Methods ----------------------------------------------------------------------------------------------------------
    def do_clusters(self):
        self.__create_model()
        self.__fit_and_predict()
        self.__safe_model()

    # K-means model creation
    def __create_model(self):
        self.__kmeans_model = KMeansModel(n_clusters=7, init='k-means++', max_iter=3000, random_state=20)

    # Fitting and predicting
    def __fit_and_predict(self):
        self.__kmeans_results.clusters = self.__kmeans_model.fit_predict(self.__data.prep_df.iloc[:, 4:])

    # Save model
    def __safe_model(self):
        self.__kmeans_results.kmeans_model = self.__kmeans_model
