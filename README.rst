Stacker
=========

**Ensemble learning combining multiple regression models into a single regression pipeline**

Shahar Azulay, Ariel Hanemann

There three so-called "meta-algorithms" / approaches to combine several machine learning techniques into one predictive model in order to:
   1. decrease the variance (bagging)
   2. decrease the bias (boosting)
   3. or improving the predictive force (stacking).

Stacking is an ensemble learning method in which one applys several models to your original data at the first stage.
Each of the models are trained to fit and best estimate the data in a predictive manner.

At the second stage, the models are not combined in a standard way (e.g. averaging their outputs) but rather meta-level model is introduced
to estimate the input together with outputs of every model to estimate the weight each model should get or, in other words, determine which
models perform better than others on the certain types of input data.

Stacking is tricky and should be performed with care.
Without a careful implementation stacking can be easily drawn into a very overfitted model, that fails to generalize
well on the input data.

The Stacker module allows for easy use of the stacking apporoach in an sklearn-based environment.
