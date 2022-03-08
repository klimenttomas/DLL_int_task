from sklearn.cluster import KMeans as KMeansModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class ViewClass:

    def __init__(self):
        pass

    # Plotting the histograms of the variables skewed and unskewed
    def show_histograms(self, prep_data: pd.DataFrame):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Data processing')

        # Revenue
        sns.distplot(prep_data['revenue'], kde=True, ax=axes[0, 0])
        axes[0, 0].set_title("Average purchase price")

        # Credit
        sns.distplot(prep_data['credit'], kde=True, ax=axes[0, 1])
        axes[0, 1].set_title("Average credit status")

        # Products
        sns.distplot(prep_data['products'], kde=True, ax=axes[0, 2])
        axes[0, 2].set_title("Num. of unique products purchased")

        # Preprocessed Revenue
        sns.distplot(prep_data['prep_revenue'], kde=True, ax=axes[1, 0])
        #axes[1, 0].set_title("Preproc. Revenue")

        # Preprocessed Credit
        sns.distplot(prep_data['prep_credit'], kde=True, ax=axes[1, 1])
        #axes[1, 1].set_title("Preprocessed Credit")

        # Preprocessed Products
        sns.distplot(prep_data['prep_products'], kde=True, ax=axes[1, 2])
        #axes[1, 2].set_title("Preprocessed Products")

    # Squared distances vs. k-value plotting
    def calculate_and_show_elbow(self, prep_data: pd.DataFrame, num_k=15):

        y_results = self.__make_list_of_K(num_k, prep_data.iloc[:, 4:])
        x_results = list(range(num_k))

        fig = plt.figure()
        plt.plot(x_results, y_results)
        plt.xlabel("K-values")
        plt.ylabel("Squared distances")

    # Result clusters plotting
    def show_clusters(self, prep_data: pd.DataFrame, clusters):
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        zdata = prep_data["revenue"].values
        xdata = prep_data["credit"].values
        ydata = prep_data["products"].values
        ax.scatter3D(xdata, ydata, zdata, c=clusters)
        ax.set_xlabel('Credit status')
        ax.set_ylabel('Number of unique products purchased')
        ax.set_zlabel('Average purchase price')

    # Making the inertia values list for appropriate K
    @staticmethod
    def __make_list_of_K(K, dataframe):
        cluster_values = list(range(1, K + 1))
        inertia_values = []

        for c in cluster_values:
            model = KMeansModel(n_clusters=c, init='k-means++', max_iter=500, random_state=42)
            model.fit(dataframe)
            inertia_values.append(model.inertia_)

        return inertia_values
