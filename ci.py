
import numpy as np
import pandas as pd

import scipy.stats as st
from scipy.stats import ttest_ind
from scipy.special import stdtr

en_file = pd.read_csv("/Users/dmurugan/Downloads/pay_survey.txt",
                      names=["Sex", "year", "pay"],
                      header=None, sep=r"\s*")

'''
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(en_file)
'''


def con_interval(data, confidence=0.95):
    n = len(data)
    t = st.t.ppf(confidence, df=n - 1)
    m = np.mean(data)
    sd = np.std(data)
    moe = t * sd / np.sqrt(n)
    print("t-score %f" % t)
    print("mean %f" % m)
    print("standard deviation %f" % sd)
    print("number of records %f" % n)
    print("margin of error %f" % moe)
    return m + moe, m - moe


num_df = en_file.values

print("************Stats for overall data************")
ci = con_interval(num_df[:, 2])
print("Confidence Interval")
print(ci)

male_data = en_file.loc[en_file['Sex'] == "Male"].values

female_data = en_file.loc[en_file['Sex'] == "Female"].values

print("************Stats for Male************")
ci = con_interval(male_data[:, 2])
print("Confidence Interval")
print(ci)

print("************Stats for Female************")
ci = con_interval(female_data[:, 2])
print("Confidence Interval")
print(ci)

# t test by predefined funciton
t, p = ttest_ind(male_data[:, 2], female_data[:, 2], equal_var=True)
print("ttest_ind:            t = %g  p = %g" % (t, p))

# t test using pooled variance
mmean = np.mean(male_data[:, 2])
mvar = np.var(male_data[:, 2])
mn = len(male_data[:, 2])

fmean = np.mean(female_data[:, 2])
fvar = np.var(female_data[:, 2])
fn = len(female_data[:, 2])

# Pooled Variance

pool_var = (((mn - 1) * mvar) + ((fn - 1) * fvar)) / (mn + fn - 2)

# Test Statistics

ts = (mmean - fmean) / np.sqrt((pool_var / mn) + (pool_var / fn))

print("test-statistics:: %f" % ts)

dof = mn + fn - 2


pf = 2*stdtr(dof, -np.abs(ts))

print(pf)


print("End of the Program")
