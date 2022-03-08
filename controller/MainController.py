# the core of the code is based on key libraries: scikit-learn, pandas, numpy, matplotlib, seaborn
from preprocess.Preprocess import PreprocessClass
from kmean.KmeanData import DataClass as KmeanData
from kmean.KmeanWorker import WorkerClass as KmeanWorker
from view.View import ViewClass
import matplotlib.pyplot as plt


class Controller:

    def __init__(self, source_path: list):
        self.__source_path: list = source_path

    def do_it(self):
        # Object for preprocessing of the raw data
        prep: PreprocessClass = PreprocessClass(self.__source_path)
        # Preprocessing of the raw data
        prep.preprocess()

        # Kmeans clustering - data objects and workers objects ---------------------------------------------------------
        worker_kmeans: KmeanWorker = KmeanWorker(prep)
        # K-means model creating, fitting and predicting
        worker_kmeans.do_clusters()
        # Getting data object which holds trained classifier and result metrics
        results_kmeans: KmeanData = worker_kmeans.kmeans_results

        # Plotting object creation
        vw: ViewClass = ViewClass()
        # Plotting the histograms of the variables skewed and unskewed
        vw.show_histograms(prep.prep_df)
        # Plotting the results clusters
        vw.show_clusters(prep.prep_df, results_kmeans.clusters)
        # Plotting the squared distances vs k
        vw.calculate_and_show_elbow(prep.prep_df, num_k=14)
        plt.show()
