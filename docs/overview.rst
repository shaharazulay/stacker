.. _adapting:

The Stacking Ensemble
===================

Stacking (sometimes called stacked generalization or bagging) is an ensemble meta-algorithm that attempts to improve a model's
predictive power by harnessing multiple models (perferably different in nature) to a unified pipeline.

The Stacking method is a very general name that is sometimes used to describe different methods to crete the unfied pipeline.
Here, we focus on a Stacking ensemble which uses the multiple models predict the target, while unifing them using a 
meta-level regressor - which learns how to annotate proper weights to the predictions of the models under it.

A simpler type of Stacking might have been to average the predictions of the different models (similar to Random Forest, 
but perhaps without the limitation of a single-type model).

In true Stacking the "stacker" or the meta-level regressor can also perform learning, where models which are proven to be
less efficient in predicting the data are provided lower weight in the final prediction.

.. image:: _static/figure_001.jpg

*[1] high-level description of the stacking ensemble*

involves training a learning algorithm to combine the predictions of several other learning algorithms. First, all of the other algorithms are trained using the available data, then a combiner algorithm is trained to make a final prediction using all the predictions of the other algorithms as additional inputs. If an arbitrary combiner algorithm is used, then stacking can theoretically represent any of the ensemble techniques described in this article, although, in practice, a logistic regression model is often used as the combiner.
Stacking typically yields performance better than any single one of the trained models.[23] It has been successfully used on both supervised learning tasks (regression,[24] classification and distance learning [25]) and unsupervised learning (density estimation).[26] It has also been used to estimate bagging's error rate.[3][27] It has been reported to out-perform Bayesian model-averaging.[28] The two top-performers in the Netflix competition utilized blending, which may be considered to be a form of stacking.[29]
The ``frame`` function takes an estimator, and returns an `adapter <https://en.wikipedia.org/wiki/Adapter_pattern>`_ of that estimator. This adapter does the 
same thing as the adapted class, except that:

1. It expects to take :class:`pandas.DataFrame` and :class:`pandas.Series` objects, not :class:`numpy.array` objects. It performs verifications on these inputs, 
    and processes the outputs, as described in :ref:`verification_and_processing`.

2. It supports two additional operators: ``|`` for pipelining (see :ref:`pipeline`), and ``+`` for feature unions (see :ref:`feature_union`).

Suppose we start with:

    >>> from sklearn import linear_model 
    >>> from sklearn import preprocessing
    >>> from sklearn import base
    >>> from ibex import frame

We can use ``frame`` to adapt an object:

    >>> prd = frame(linear_model.LinearRegression())
    >>> prd
    Adapter[LinearRegression](copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

We can use ``frame`` to adapt a class:

    >>> PDLinearRegression = frame(linear_model.LinearRegression)
    >>> PDStandardScaler = frame(preprocessing.StandardScaler)

Once we adapt a class, it behaves pretty much like the underlying one. We can construct it in whatever ways it the underlying class supports, for example:

    >>> PDLinearRegression()
    Adapter[LinearRegression](copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
    >>> PDLinearRegression(fit_intercept=False)
    Adapter[LinearRegression](copy_X=True, fit_intercept=False, n_jobs=1, normalize=False)

It has the same name as the underlying class:

    >>> PDLinearRegression.__name__
    'LinearRegression'

It subclasses the same mixins of the underlying class:

    >>> isinstance(PDLinearRegression(), base.RegressorMixin)
    True
    >>> isinstance(PDLinearRegression(), base.TransformerMixin)
    False
    >>> isinstance(PDStandardScaler(), base.RegressorMixin)
    False
    >>> isinstance(PDStandardScaler(), base.TransformerMixin)
    True

As can be seen above, though, the string and representation is modified, to signify this is an adapted type:

    >>> PDLinearRegression()
    Adapter[LinearRegression](copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
    >>> linear_model.LinearRegression()
    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

|
|

Of course, the imposition to decorate every class (not to mention object) via ``frame``, can become annoying.

.. image:: _static/got_frame.jpeg

If a library is used often enough, it might pay to wrap it once. Ibex does this (nearly completely) automatically for :mod:`sklearn` (see :ref:`sklearn`).

