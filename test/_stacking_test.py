import unittest

import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

import stacker


class _BadPredictor(object):
        """
        help regressor that is overfitted by design.
        the regressor can therefore mislead a regressor that uses its outputs if correct stacking is not installed.
        """
        def fit(self, x, y):
            self._y = y
            return self
        
        def get_params(self, deep=True):
            return {}
        
        def predict(self, x):
            return self._y[:, 0]


class _StackerTest(unittest.TestCase):

    def _default_cv_fun(self):
        return sklearn.model_selection.KFold(n_splits=10)

    def _bad_cv_fun(self):
        return sklearn.model_selection.ShuffleSplit(
            n_splits=2, test_size=.25, random_state=0)

    def test_basic_operation(self):
        """"test basic usage of the stacking ensemble"""

        from sklearn.datasets import load_boston
        X, y = load_boston(return_X_y=True)

        tr, te = self._default_cv_fun().split(X).next()
        X_train, X_test = X[tr], X[te]

        stck = stacker.Stacker(
            first_level_preds=[
                Pipeline([
                    ('pca', PCA()), ('dtr', DecisionTreeRegressor(random_state=1))]),
                LinearRegression()],
            stacker_pred=SVR(),
            cv_fn=self._default_cv_fun(),
            n_jobs=-1)

        stck.fit(X[tr], y[tr])
        y_hat = stck.predict(X[te])
        score = sklearn.metrics.r2_score(y[te], y_hat)

        self.assertGreater(score, 0)

    def test_bad_coverage_cv(self):
        """test the case where a the input contains a cross validation strategy that fails to cover all
           of the input indices"""

        _make_stacker = lambda: stacker.Stacker(
            first_level_preds=[
                Pipeline([
                    ('pca', PCA()), ('dtr', DecisionTreeRegressor(random_state=1))]),
                LinearRegression()],
            stacker_pred=SVR(),
            cv_fn=self._bad_cv_fun(),
            n_jobs=-1)

        self.assertRaises(
            ValueError,
            _make_stacker()
        )

    def test_overfitting_resilience(self):
        """test that the stacker is robust to first level estimators that demonstrate overfitting
           (the stacker is expected to give low weight to the overfitted member)"""

        from sklearn.datasets import load_boston
        X, y = load_boston(return_X_y=True)

        tr, te = self._default_cv_fun().split(X).next()
        X_train, X_test = X[tr], X[te]

        stck = stacker.Stacker(
            first_level_preds=[
                DecisionTreeRegressor(random_state=1),
                _BadPredictor()],
            stacker_pred=LinearRegression(),
            cv_fn=self._default_cv_fun(),
            n_jobs=-1)

        stck.fit(X[tr], y[tr])

        self.assertGreater(
            abs(stck._stacker_pred.coef_[0]),
            abs(stck._stacker_pred.coef_[1]))

        
if __name__ == '__main__':
    unittest.main()
