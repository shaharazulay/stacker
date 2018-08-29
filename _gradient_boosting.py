"""Gradient Boosted Neural Network

This module contains methods for fitting gradient boosted neural network for
both regression only.

The module structure is the following:

- ``GradientBoostingRegressor_`` implements gradient boosting for
  regression problems.
"""

import numpy as np
from math import exp
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation
from day_two.sklearn_.metrics import r2_score
from day_two.sklearn_.preprocessing import ColumnSelectionApplier
from sknn.mlp import Regressor


class GradientBoostingRegressor_(object):
    """ customization of the GradientBoostingRegressor class
    to use sknn neural network regressor
    Note:
    This iterative regressor actually uses Forward Stagewize Additive model
    rather that using actual gradient of the loss function for residual
    calcluations.

    Attributes
    ----------
    # Tmp Shahar - TBD

    Parameters
    ----------
    residual_rate: float, optional (default=0.1)
        the rate shrinks the contribution of each neural network stage
        through out the gradient_boosting procedure. residual_rate can also
        be refered to as eplsilon in gradient_boosting.
    n_max_stages: int, optional (default=100)
        The max number of boosting stages to perform. Gradient boosting
        is fairly robust to over-fitting so a large number usually
        results in better performance.
    scorer: score function reference, optional
        (default ='day2.sklearn_.metrics.r2_score')
        The score function used to estimate the performance of each stage
    n_stage_estimators: int, optional (default=10)
        The number of random estimators between which to compare at each
        boosting stage. Each stage will chose the best estimator from the
        tested n_estimators.

    neural network meta parametes:
        according to sknn.mlp documentation.
        see 'http://scikit-neuralnetwork.readthedocs.org\
        /en/latest/module_mlp.html'
        for more details
    """

    def __init__(
        self,
        n_max_stages=100,
        residual_rate=0.1,
        scorer=r2_score,
        n_stage_estimators=10,
        **kwargs):

        self._n_max_stages = n_max_stages
        self._residual_rate = residual_rate
        self._scorer = scorer
        self._n_stage_estimators = n_stage_estimators
        self._kwargs = kwargs
        
        self.estimators_ = []  # list of fitted estimators
        self.train_scores_ = []  # list of train scores of all stages
        self.n_stages = 0  # number of final gradient_boosting stages

        self._residual_rates = []  # list of used residual rates per stage

    def _fit_stage(self, X, y, y_pred, residual_rate):
        """Fit another stage of nn to the boosting model.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Features matrix, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values of train data set
        y_pred : array-like, shape = [n_samples]
            Predicted values (by previous stage)
        residual_rate: float
            The residual learning rate (epsilon) to use in the curernt
            stage. determins the relative contribution of the current
            stage to the entire iterative boosting process

        Returns
        ----------
        nn_regr: estimator of type neural netwrok
            Fitted estimator (fitted to the curret stage residual)
        y_pred: array-like, shape = [n_samples]
            The prediction values of current stage at the boosting
        score: float
            The score value of the current stage
        """
        n_samples, n_features = X.shape

        # current stage will try and predict the residual
        residual = y - y_pred

        # split and bootstrap the data set
        train_indices, test_indices = self._bootstrap_data_indices(n_samples)
        Xtrain, Xtest = X[train_indices], X[test_indices]
        ytrain, ytest = residual[train_indices], residual[test_indices]

        # generate n_stage_estimators new estimators
        score_list = []
        stage_estimators = []
        for _ in range(self._n_stage_estimators):

            nn_regr = self._get_base_regressor(n_features)
            # select random feature subset to train regressor
            feature_subset = self._choose_feature_subset(n_features)
            # create holistic regressor (with feature selection)
            current_stage = ColumnSelectionApplier(feature_subset, nn_regr)
            # fit and score the stage
            current_stage.fit(Xtrain, ytrain)
            y_hat = current_stage.predict(Xtest)
            y_hat = y_hat.reshape(len(y_hat), )
            score = self._scorer(
                ytest,
                y_hat,
                np.mean(ytrain))
            if (score > 0):  # if R < 0 throw the predictor
                score_list.append(score)
                stage_estimators.append(current_stage)

        # choose best regressor
        if len(score_list) == 0:
            return None, [], 0
        i_stage = score_list.index(max(score_list))
        current_stage = stage_estimators[i_stage]

        y_pred_current_stage = current_stage.predict(X)
        y_pred_current_stage = y_pred_current_stage.reshape(
            len(y_pred_current_stage, ))

        y_pred += residual_rate * y_pred_current_stage

        accumulated_score = self._scorer(
            y,
            y_pred,
            np.mean(y))

        return current_stage, y_pred, accumulated_score

    def _fit_stages(self, X, y, y_pred, stop_condition=None):
        """Iteratively fit the nn stages to the boosting model.

        creats self.estimators_ - a list[] of estimators of type neural netwrok
        list of fitted estimator (fitted to the curret stage residual)

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Features matrix, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values of train data set
        y_pred : array-like, shape = [n_samples]
            Predicted values (by init estimator - intial guesss)
        stop_condition: float, optional (default=None)
            The stop condition of the iterative boosting model.
            This condition is represented as the R^2 of the aggregated
            predicted y at each stage

        Returns
        ----------
        n_stages: int
            The number of final stages the boosting model used for fitting
            the training data
        """
        n_stages = 0
        residual_rate = self._residual_rate
        i_stage = 0

        if stop_condition is None:
            stop_condition = 1  # no effective stopping condition

        while (i_stage < self._n_max_stages):

            stage, y_pred_next, score = self._fit_stage(
                X,
                y,
                y_pred,
                residual_rate)

            if stage is None:  # no good estimator found in stage
                i_stage += 1
                continue

            self.estimators_.append(stage)
            self.train_scores_.append(score)
            self._residual_rates.append(residual_rate)

            y_pred = y_pred_next

            i_stage += 1
            n_stages += 1

            if score > stop_condition:
                break

            # Tmp Shahar - fix decaying residual_rate here
            residual_rate = self._residual_rate * exp(-score)

        return n_stages

    def fit(self, X, y, sample_weight=None):
        """Fit the gradient boosting model.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values of the data set
        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        self : object
            Returns self.
        """

        if sample_weight is not None:
            raise NotImplementedError()
        # Tmp Shahar - currently not using stop_condition

        # validity check
        if len(y.shape) != 1:
            raise ValueError(
                'you must use single dimension shape of shape=(N,).'
                '  received shape is: %s' % str(y.shape))

        # initial guess - starting with simple linear estimator
        init_regr = LinearRegression()
        init_regr.fit(X, y)
        y_pred = init_regr.predict(X)
        y_pred = y_pred.reshape(len(y_pred), )

        # add init regressor to list of used estimators
        self.estimators_ = []
        self.train_scores_ = []
        self._residual_rates = []

        self.estimators_.append(init_regr)
        self.train_scores_.append(self._scorer(
            y,
            y_pred,
            np.mean(y)))
        self._residual_rates.append(1)

        # fit the boosting stages
        self.n_stages = self._fit_stages(X, y, y_pred, stop_condition=None)

        return self

    def predict(self, X):
        """Predict y according to input features X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y: array of shape = [n_samples,]
            The predicted values.
        """
        if len(self.estimators_) == 0:
            raise RuntimeError('Regressor was not fitted, or no good'
                               'fit was found')

        y_pred = 0
        for stage, residual_rate in zip(self.estimators_, self._residual_rates):
            y_pred += residual_rate * stage.predict(X).reshape(len(X), )

        return y_pred

    def _bootstrap_data_indices(self, n_samples):
        """Perform the following operation on the data indices:
           1. split the data indices into train, test using SuffleSplit
           2. bootstrap the data indices (replace=True) using np.random.choice

        Parameters
        ----------
        n_samples: int
            The number of samlpes in the the data set
            (X & y have n_samples raws)

        Returns
        ---------
        train_indices: array-like, shape = [m_train_samples,]
            The indices of the train data set indicating which samples
            are chosen to build the train data set.
        test_indices: array-like, shape = [m_test_samples,]
            The indices of the test data set indicating which samples
            are chosen to build the test data set.
        """

        sh_spl = cross_validation.ShuffleSplit(
            n_samples, n_iter=1, test_size=0.1)
        train_indices, test_indices = next(iter(sh_spl))

        # bootstraping the train indices
        train_indices = np.random.choice(
            train_indices,
            len(train_indices),
            replace=True)

        return train_indices, test_indices

    def _get_base_regressor(self, n_features):
        """Creates the instance of our base neural network regressor"""

        nn_regr = Regressor(**self._kwargs)
        return nn_regr

    def _choose_feature_subset(self, n_features):
        "Returns random column list to use as feature selection transformer"""

        # choice_ = np.random.choice(n_features, n_features, replace=True)
        choice_ = np.random.choice(n_features, n_features, replace=False)
        return list(np.unique(choice_))
