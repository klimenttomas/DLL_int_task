import numpy as np
import pandas as pd
import dython as dy


class PreprocessClass:

    def __init__(self, source_path: list):
        self.__source_path: list = source_path      # path to source
        self.__orig_df: pd.DataFrame = None         # original dataframe from source
        self.__unique_users_list: list = []         # list of unique users
        self.__prep_df: pd.DataFrame = None         # preprocessed dataframe

    # Property decorators ----------------------------------------------------------------------------------------------
    @property
    def source_path(self) -> list:
        return self.__source_path

    @property
    def orig_df(self) -> pd.DataFrame:
        return self.__orig_df

    @orig_df.setter
    def orig_df(self, value: pd.DataFrame):
        self.__orig_df = value

    @property
    def prep_df(self) -> pd.DataFrame:
        return self.__prep_df

    @prep_df.setter
    def prep_df(self, value: pd.DataFrame):
        self.__prep_df = value

    # Methods ----------------------------------------------------------------------------------------------------------
    # Reading from source and dataframe creating
    def __read_from_source(self):
        df_list: list = []
        for item in self.__source_path:
            df_list.append(pd.read_csv(item))

        self.__orig_df = pd.concat(df_list, axis=0)
        #self.__orig_df = self.__orig_df.iloc[:4000]

    # Getting the list of users
    def __get_unique_users(self):
        self.__unique_users_list = self.__orig_df["user_id"].unique().tolist()

    # Applying the log1p transformation for to manage skewness of distribution
    def __apply_log1p_transformation(self):
        self.__prep_df["prep_revenue"] = np.log1p(self.__prep_df["revenue"])
        self.__prep_df["prep_credit"] = self.__prep_df["credit"]
        self.__prep_df["prep_products"] = np.log1p(self.__prep_df["products"])

    # Creating new features for unique users
    def __create_new_feature_dataframe(self):
        df_list: list = []

        for user in self.__unique_users_list:
            temp_list: list = None
            user_df: pd.DataFrame = self.__orig_df[self.__orig_df["user_id"] == user]

            user_id = user
            cred = user_df["credit_status_cd"].mean()
            products = user_df["prod_id"].nunique()
            revenue = user_df["revenue_usd"].sum() / user_df["prod_id"].count()

            temp_list = [user_id, revenue, cred, products]
            df_list.append(temp_list)

        self.__prep_df = pd.DataFrame(df_list, columns=["customer", "revenue", "credit", "products"])

    #Centering of the data
    def __center_dataframe(self, columns: list):
        for column in columns:
            self.__prep_df[column] = self.__prep_df[column] - self.__prep_df[column].mean()

    # Scaling the data
    def __scale_dataframe(self, columns: list):
        for column in columns:
            self.__prep_df[column] = self.__prep_df[column] / self.__prep_df[column].std()

    # Determine correlation of orginal data
    def __cor(self):
        cat_data = ["user_id", "prod_id", "sex", "age_cat", "credit_status_cd", "edcution_cat", "years_in_residence", "car_ownership", "prod_cat_1"]
        cont_data = ["revenue_usd"]

        corr_dict: dict = dy.nominal.associations(self.__orig_df.drop(columns=["prod_cat_2", "prod_cat_3"]), nominal_columns=cat_data, numerical_columns=cont_data, figsize=(20, 10), mark_columns=True)
        for key in corr_dict:
            print(f"{key}: {corr_dict[key]}")

    # Preprocessing the raw data
    def preprocess(self):
        self.__read_from_source()
        self.__get_unique_users()
        self.__create_new_feature_dataframe()
        self.__apply_log1p_transformation()
        self.__center_dataframe(["prep_revenue", "prep_credit", "prep_products"])
        self.__scale_dataframe(["prep_revenue", "prep_credit", "prep_products"])

        # Correlation of the dataset
        self.__cor()
