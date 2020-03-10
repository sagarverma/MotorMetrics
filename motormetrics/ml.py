import math

import numpy as np

from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import mean_squared_log_error


def flatten_extra_dims(quant):
    '''
    Flatten array with dimensions more than 1
    All metrics are independent of dimension
    '''
    return quant.flatten()


def r2(y_true, y_pred):
    '''
    R-squared(coefficient of determination)
    https://en.wikipedia.org/wiki/Coefficient_of_determination
    y = y_true
    f = y_pred
    \bar{y} = 1/n \sum_{i=1}^{n} y_i
    SS_{tot} = \sum_{i=1}^{n} (y_i - \bar{y})^2
    SS_{reg} = \sum_{i=1}^{n} (f_i - \bar{y})^2
    SS_{res} = \sum_{i=1}^{n} (y_i - f_i)^2
    R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
    '''
    assert y_true.shape == y_pred.shape

    y_true = flatten_extra_dims(y_true)
    y_pred = flatten_extra_dims(y_pred)
    return r2_score(y_true, y_pred)


def rmsle(y_true, y_pred):
    '''
    Root mean square log error
    https://medium.com/analytics-vidhya/root-mean-square-log-error-rmse-vs-rmlse-935c6cc1802a
    y = y_true
    f = y_pred
    rmsle(y, f) = \sqrt(1/n \sum_{i=1}^{n} (log(y_i + 1) - log(f_i + 1))^2)
    '''
    assert y_true.shape == y_pred.shape
    assert y_true.min() >= -1 and y_true.max() <= 1
    assert y_pred.min() >= -1 and y_pred.max() <= 1

    y_true = flatten_extra_dims(y_true)
    y_pred = flatten_extra_dims(y_pred)
    terms_to_sum = (np.log(y_pred + 1) - np.log(y_true + 1))**2
    return (sum(terms_to_sum) * (1.0/len(y_true))) ** 0.5

def rmse(y_true, y_pred):
    '''
    Root mean square error
    https://en.wikipedia.org/wiki/Root-mean-square_deviation
    y = y_true
    f = y_pred
    rmse(y, f) = \frac{1}{n} \sqrt(mse(y, f))
    '''
    assert y_true.shape == y_pred.shape

    y_true = flatten_extra_dims(y_true)
    y_pred = flatten_extra_dims(y_pred)

    return np.sqrt(((y_pred - y_true) ** 2).mean())


def mae(y_true, y_pred):
    '''
    Mean absolute error
    https://en.wikipedia.org/wiki/Mean_absolute_error
    y = y_true
    f = y_pred
    mae(y, f) = \frac{1}{n} \sum_{i=1}^{n} |y_i - f_i|
    '''
    assert y_true.shape == y_pred.shape

    y_true = flatten_extra_dims(y_true)
    y_pred = flatten_extra_dims(y_pred)
    return mean_absolute_error(y_true, y_pred)


def smape(y_true, y_pred):
    '''
    Symmetric mean absolute percentage error
    https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
    y = y_true
    f = y_pred
    smape(y, f) = \frac{100%}{n} \sum_{i=1}^{n} \frac{|y_i - f_i|}{(|y_i|+|f_i|)/2}
    '''
    assert y_true.shape == y_pred.shape

    y_true = flatten_extra_dims(y_true)
    y_pred = flatten_extra_dims(y_pred)
    return 100.0/ len(y_true) * np.sum(2.0 * np.abs(y_pred - y_true) / \
           (np.abs(y_true) + np.abs(y_pred) + 0.00001))


def sc(signal):
    '''
    Signal complexity/mean total variation
    https://en.wikipedia.org/wiki/Total_variation
    sc(y) = \frac{1}{n-1} \sum{i=1}{n-1} |y_{i+1} - y_i|
    '''
    signal = flatten_extra_dims(signal)
    return np.mean(abs(signal[1:] - signal[:-1]))


def smape_vs_sc(y_true, y_pred, window):
    '''
    Compute smape and sc over a rolling window.
    '''
    assert window >= 1
    assert y_true.shape == y_pred.shape

    y_true = flatten_extra_dims(y_true)
    y_pred = flatten_extra_dims(y_pred)
    smape_vs_sc_all_windows = []

    for i in range(0, y_true.shape[0]):
        if i + window + 1 < y_true.shape[0]:
            smape_val = smape(y_true[i: i + window], y_pred[i: i + window])
            sc_val = sc(y_true[i : i + window])
            smape_vs_sc_all_windows.append([smape_val, sc_val])

    return np.asarray(smape_vs_sc_all_windows)


def sc_mse(y_pred, y_true):
    '''
    Total variation weighted mean square error
    '''
    assert y_true.shape == y_pred.shape

    sc_y_true = np.mean(np.abs(y_true[:,:,1:] - y_true[:,:,:-1]), axis=2)
    mse = np.mean((y_pred - y_true) ** 2.0, axis=2)
    loss = sc_y_true * mse
    loss = np.mean(loss)
    return loss
