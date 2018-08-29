import numpy as np
import unittest
from day_two import unittest_

if False:
    class _GradientBoostingRegressorTest(unittest_.TestCase):
        def test_bootstrap_method(self):
            import day_two

            gb_regr = day_two.sklearn_.ensemble._gradient_boosting.GradientBoostingRegressor_()

            n_samples = 100
            train_indices, test_indices = gb_regr\
                ._bootstrap_data_indices(n_samples)

            self.assertEqual(
                len(train_indices) + len(test_indices),
                n_samples,
                "size of train and test data set must sum to n_samples")

            self.assertEqual(
                len(test_indices),
                len(np.unique(test_indices)),
                "test data set is not supposed to be bootstrapped in any way")

            self.assertEqual(
                len(np.intersect1d(train_indices, test_indices)),
                0,
                "test and train data must have zero intersection")

        def test_single_stage(self):
            import day_two

            # generate data
            t = np.linspace(0, 1, 100)
            y = t + 0.1*np.random.randn()
            X = np.column_stack((t, t, t))

            gb_regr = day_two.sklearn_.ensemble._gradient_boosting.GradientBoostingRegressor_(n_stage_estimators=5)

            residual_rate = 0.1
            estimator, y_pred, score = gb_regr._fit_stage(
                X, y, np.mean(y), residual_rate)

            self.assertTrue(
                score >= 0,
                "estimators with R < 0 should be discarded")

            if estimator is not None:
                selected_features_columns = estimator._wh

                self.assertTrue(
                    len(selected_features_columns) > 0,
                    "num of selected features at each stage must be > 0")

                y_pred_next = np.mean(y) + residual_rate * estimator.predict(X)\
                    .reshape(len(y), )
                np.testing.assert_array_equal(
                    y_pred_next,
                    y_pred,
                    "stage prediction does not represent stagewise addetive model")

        def test_fit(self):
            import day_two

            # generate data
            t = np.linspace(0, 1, 100)
            y = t + 0.1*np.random.randn()
            X = np.column_stack((t, t, t))

            gb_regr = day_two.sklearn_.ensemble._gradient_boosting.GradientBoostingRegressor_(n_max_stages=3)
            gb_regr.fit(X, y)

            self.assertEqual(
                len(gb_regr.estimators_),
                gb_regr.n_stages + 1,
                "number of used estimators should be equal to #stages + 1")

        def test_predict(self):
            import day_two

            # generate data
            t = np.linspace(0, 1, 100)
            y = t + 0.1*np.random.randn()
            X = np.column_stack((t, t, t))

            gb_regr = day_two.sklearn_.ensemble._gradient_boosting.GradientBoostingRegressor_(n_max_stages=3)
            gb_regr.fit(X, y)
            y_pred = gb_regr.predict(X)

            self.assertTrue(
                len(y) == len(y_pred),
                "dimention of y and y_pred must be the same")

    if __name__ == '__main__':
        unittest_.main()
