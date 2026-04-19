"""
Evaluation functions for time series data.
"""
# Stylized facts
from .fisher_kurtosis import fisher_kurtosis
from .autocorrelation import acf_squared

__all__ = [
    'fisher_kurtosis',
    'acf_squared',
]
