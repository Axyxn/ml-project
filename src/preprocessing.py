"""All learned transformations live inside the training pipeline."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.utils.validation import check_is_fitted

from src.config import FEATURES, TrainingConfig
from src.data import stratified_sample


class MutualInformationSelector(TransformerMixin, BaseEstimator):
    """SelectKBest with a bounded, stratified training-only MI estimation sample.

    Sampling does not change the class prior. It only limits the expensive ranking
    calculation. The selected columns are then applied to every real training row.
    Each call to fit (including each CV fold) learns a new ranking from that fold.
    """
    def __init__(self, k=20, max_rows=60_000, random_state=42):
        self.k = k
        self.max_rows = max_rows
        self.random_state = random_state

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("MI selector expects named features from the pandas preprocessor.")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        X_sample, y_sample = stratified_sample(X, y, self.max_rows, self.random_state)
        self.ranking_rows_ = len(X_sample)
        self.ranking_frauds_ = int(np.sum(y_sample))
        scores = mutual_info_classif(
            X_sample, y_sample, discrete_features=False, random_state=self.random_state)
        # SelectKBest provides a standard support mask; the scores were computed above.
        self.selector_ = SelectKBest(k=self.k, score_func=lambda features, target: scores)
        self.selector_.fit(X_sample, y_sample)
        # Do not persist an unpicklable lambda inside a fitted joblib artifact.
        self.selector_.score_func = mutual_info_classif
        self.scores_ = scores
        self.selected_features_ = self.feature_names_in_[self.selector_.get_support()].tolist()
        return self

    def transform(self, X):
        check_is_fitted(self, "selected_features_")
        return X.loc[:, self.selected_features_].copy()

    def get_support(self):
        check_is_fitted(self, "selector_")
        return self.selector_.get_support()

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "selected_features_")
        return np.asarray(self.selected_features_, dtype=object)


def make_preprocessing_steps(config=None):
    config = config or TrainingConfig()
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).set_output(transform="pandas")
    scaler = ColumnTransformer(
        [("time_amount", RobustScaler(), ["Time", "Amount"]),
         ("pca", StandardScaler(), [f"V{i}" for i in range(1, 29)])],
        remainder="drop", verbose_feature_names_out=False).set_output(transform="pandas")
    selector = MutualInformationSelector(
        k=config.selected_k, max_rows=config.selection_max_rows, random_state=config.seed)
    return [("imputer", imputer), ("scaler", scaler), ("select", selector)]
