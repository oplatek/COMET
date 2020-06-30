# -*- coding: utf-8 -*-
r"""
Regression Metrics
==============
    Metrics to evaluate regression quality of estimator models.
"""
import warnings

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr

from pytorch_lightning.metrics import Metric


class RegressionReport(Metric):
    def __init__(self):
        super().__init__(name="regression_report")
        self.metrics = [Pearson(), Kendall(), Spearman()]

    def forward(self, x: np.array, y: np.array) -> float:
        """ Computes Kendall correlation.

        :param x: predicted scores.
        :param x: ground truth scores.
        
        Return: 
            - Kendall Tau correlation value.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return {metric.name: metric(x, y) for metric in self.metrics}


class Kendall(Metric):
    def __init__(self):
        super().__init__(name="kendall")

    def forward(self, x: np.array, y: np.array) -> float:
        """ Computes Kendall correlation.

        :param x: predicted scores.
        :param x: ground truth scores.
        
        Return: 
            - Kendall Tau correlation value.
        """
        return kendalltau(x, y)[0]


class Pearson(Metric):
    def __init__(self):
        super().__init__(name="pearson")

    def forward(self, x: np.array, y: np.array) -> float:
        """ Computes Pearson correlation.

        :param x: predicted scores.
        :param x: ground truth scores.
        
        Return: 
            - Pearson correlation value.
        """
        return pearsonr(x, y)[0]


class Spearman(Metric):
    def __init__(self):
        super().__init__(name="spearman")

    def forward(self, x: np.array, y: np.array) -> float:
        """ Computes Spearman correlation.

        :param x: predicted scores.
        :param x: ground truth scores.
        
        Return: 
            - Spearman correlation value.
        """
        return spearmanr(x, y)[0]
