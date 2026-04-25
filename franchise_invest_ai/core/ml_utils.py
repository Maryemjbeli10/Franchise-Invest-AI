import joblib
import os

class MLProvider:
    _regression_model = None
    _recommender = None
    _segmentation = None

    @classmethod
    def get_regression(cls):
        if cls._regression_model is None:
            cls._regression_model = joblib.load('models/best_model.pkl')
        return cls._regression_model

    @classmethod
    def get_recommender(cls):
        if cls._recommender is None:
            # On charge ta classe FranchiseRecommender v3
            cls._recommender = joblib.load('models/franchise_recommender.pkl')
        return cls._recommender
    @classmethod
    def get_segmentation(cls):
        if cls._segmentation is None:
            # On charge ta classe FranchiseRecommender v3
            cls._segmentation = joblib.load('models/kmeans_segmentation.pkl')
        return cls._recommender    