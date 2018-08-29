import unittest

import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

import stacker


class BadPredictor(object):

        def fit(self, x, y):
            self._y = y
            return self
        
        def get_params(self, deep=True):
            return {}
        
        def predict(self, x):
            return self._y[:, 0]


class _StackerTest(unittest.TestCase):

    data = np.array([
        [71, 17, 14],
        [97, 20, 32],
        [53, 13, 15],
        [6, 6, 9],
        [71, 43, 71],
        [2, 1, 1],
        [89, 14, 11],
        [52, 11, 21],
        [75, 14, 28],
        [148, 36, 36],
        [53, 9, 43],
        [53,  42, 17],
        [57, 15,  8],
        [14, 0, 0],
        [14, 2, 0],
        [47, 11, 5],
        [14, 1, 0],
        [51, 27, 22],
        [7, 1, 0],
        [24, 1, 0]])

    target = np.array([
        [7], [14], [7], [7], [0], [0], [35], [7], [35], [35],
        [0], [0], [14], [14], [7], [35], [7], [14], [0], [14]])
    
    def _default_cv_fun(self):
        return sklearn.model_selection.KFold(n_splits=10)

    def _bad_cv_fun(self):
        return sklearn.model_selection.ShuffleSplit(
            n_splits=2, test_size=.25, random_state=0)

    # def test_basic_stacking(self):
    #     """test the stacking """
    #     fun = self._default_cv_fun()
    #
    #     stck = stacker.Stacker(
    #         first_level_preds=[
    #             Pipeline([
    #                 ('pca', PCA()), ('dtc', DecisionTreeClassifier(random_state=1))]),
    #             LinearRegression()],
    #         stacker_pred=SVR(),
    #         cv_fn=fun,
    #         n_jobs=-1)
    #
    #     stck.fit(self.data[0:16], self.target[0:16])
    #     res = stck.predict(self.data[17:20])
    #     np.testing.assert_almost_equal(res, np.array(
    #              [7.77900296,  6.88169834,  8.57800039]), decimal=3)
    #
    # def test_bad_predictor(self):
    #     """test that the stacker behaves logically and gives low coefficient to a bad predictor"""
    #     fun = self._default_cv_fun()
    #
    #     stck = stacker.Stacker(
    #         first_level_preds=[
    #             DecisionTreeClassifier(random_state=1),
    #             BadPredictor()],
    #         stacker_pred=LinearRegression(),
    #         cv_fn=fun,
    #         n_jobs=-1)
    #
    #     stck.fit(self.data[0:16], self.target[0:16])
    #     stck.predict(self.data[0:16])
    #     np.testing.assert_almost_equal(
    #         stck._stacker_pred.coef_, np.array([[0.39485935, -0.12336237]]), decimal=3)
    #
    # def test_bad_coverage_cv(self):
    #     """test the case where a the input contains a cv method that does
    #     not cover all of x's rows"""
    #     fun = self._bad_cv_fun()
    #
    #     self.assertRaises(
    #         ValueError, stacker.Stacker,
    #         [DecisionTreeClassifier(random_state=1), BadPredictor()],
    #         LinearRegression(), fun, -1)

    def test_basic_uperation(self):
        from sklearn.datasets import load_boston
        X, y = load_boston(return_X_y=True)

        stck = stacker.Stacker(
            first_level_preds=[
                Pipeline([
                    ('pca', PCA()), ('dtr', DecisionTreeRegressor(random_state=1))]),
                LinearRegression()],
            stacker_pred=SVR(),
            cv_fn=self._default_cv_fun(),
            n_jobs=-1)

        tr, te = self._default_cv_fun().split(X).next()
        X_train, X_test =X[tr], X[te]

        stck.fit(X[tr], y[tr])
        y_hat = stck.predict(X[te])
        score = sklearn.metrics.r2_score(y[te], y_hat)

        self.assertGreater(score, 0)

        



if __name__ == '__main__':
    unittest.main()
