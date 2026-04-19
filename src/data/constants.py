"""
Data-layer constants shared across the project.

These constants define the COVID-19 data-gap boundaries detected in the
ACN datasets.  During this period stations reported zero energy — these
are artificial zeros from ``asfreq`` fill, not real observations.
"""

# Detected via _warn_zero_gaps: 105 consecutive zero-value days
# in ACN_JPL from 2020-08-05 to 2020-11-17.
COVID_START = '2020-08-05'
COVID_END = '2020-11-17'

EXCLUDE_COVID_DATA = True
