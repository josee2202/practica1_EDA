#!/usr/bin/env python
"""Synthetic stand-in for data/raw_data/application_data.csv.

Same 122 columns, dtypes, category labels and (roughly) missing-value rates as the
real file, but every value is a random draw. The target comes from a hand-written
logistic rule. Its only purpose is to let CI execute the notebooks end to end
without the real dataset, which cannot be redistributed. Metrics computed on this
data say nothing about the real problem.

The rule gives weight to the columns that notebooks 04 and 05 reference by name
(e.g. DAYS_EMPLOYED, FLAG_EMP_PHONE, AMT_GOODS_PRICE), so that the data-driven
feature selection in notebook 03 keeps them, as it does on the real data.

    python scripts/make_synthetic_data.py --rows 6000 --out data/raw_data/application_data.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES = {
    'NAME_CONTRACT_TYPE': (['Cash loans', 'Revolving loans'], [0.9, 0.1]),
    'CODE_GENDER': (['F', 'M', 'XNA'], [0.658, 0.341, 0.001]),
    'FLAG_OWN_CAR': (['N', 'Y'], [0.66, 0.34]),
    'FLAG_OWN_REALTY': (['N', 'Y'], [0.31, 0.69]),
    'NAME_TYPE_SUITE': (['Unaccompanied', 'Family', 'Spouse, partner', 'Children', 'Other_B', 'Other_A',
                         'Group of people'], [0.81, 0.13, 0.037, 0.011, 0.006, 0.003, 0.003]),
    'NAME_INCOME_TYPE': (['Working', 'Commercial associate', 'Pensioner', 'State servant', 'Unemployed',
                          'Student', 'Businessman', 'Maternity leave'], [0.516, 0.233, 0.18, 0.068, 0.001,
                                                                         0.001, 0.0005, 0.0005]),
    'NAME_EDUCATION_TYPE': (['Secondary / secondary special', 'Higher education', 'Incomplete higher',
                             'Lower secondary', 'Academic degree'], [0.71, 0.243, 0.033, 0.0125, 0.0015]),
    'NAME_FAMILY_STATUS': (['Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow',
                            'Unknown'], [0.639, 0.148, 0.097, 0.064, 0.0515, 0.0005]),
    'NAME_HOUSING_TYPE': (['House / apartment', 'With parents', 'Municipal apartment', 'Rented apartment',
                           'Office apartment', 'Co-op apartment'], [0.887, 0.048, 0.036, 0.016, 0.009, 0.004]),
    'OCCUPATION_TYPE': (['Laborers', 'Sales staff', 'Core staff', 'Managers', 'Drivers', 'High skill tech staff',
                         'Accountants', 'Medicine staff', 'Security staff', 'Cooking staff', 'Cleaning staff',
                         'Private service staff', 'Low-skill Laborers', 'Waiters/barmen staff', 'Secretaries',
                         'Realty agents', 'HR staff', 'IT staff'], None),
    'WEEKDAY_APPR_PROCESS_START': (['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY',
                                    'SUNDAY'], [0.16, 0.17, 0.17, 0.16, 0.16, 0.11, 0.07]),
    'ORGANIZATION_TYPE': (['Advertising', 'Agriculture', 'Bank', 'Business Entity Type 1',
                           'Business Entity Type 2', 'Business Entity Type 3', 'Cleaning', 'Construction',
                           'Culture', 'Electricity', 'Emergency', 'Government', 'Hotel', 'Housing',
                           'Industry: type 1', 'Industry: type 10', 'Industry: type 11', 'Industry: type 12',
                           'Industry: type 13', 'Industry: type 2', 'Industry: type 3', 'Industry: type 4',
                           'Industry: type 5', 'Industry: type 6', 'Industry: type 7', 'Industry: type 8',
                           'Industry: type 9', 'Insurance', 'Kindergarten', 'Legal Services', 'Medicine',
                           'Military', 'Mobile', 'Other', 'Police', 'Postal', 'Realtor', 'Religion',
                           'Restaurant', 'School', 'Security', 'Security Ministries', 'Self-employed',
                           'Services', 'Telecom', 'Trade: type 1', 'Trade: type 2', 'Trade: type 3',
                           'Trade: type 4', 'Trade: type 5', 'Trade: type 6', 'Trade: type 7',
                           'Transport: type 1', 'Transport: type 2', 'Transport: type 3', 'Transport: type 4',
                           'University', 'XNA'], None),
    'FONDKAPREMONT_MODE': (['reg oper account', 'reg oper spec account', 'not specified', 'org spec account'],
                           [0.75, 0.13, 0.06, 0.06]),
    'HOUSETYPE_MODE': (['block of flats', 'specific housing', 'terraced house'], [0.98, 0.01, 0.01]),
    'WALLSMATERIAL_MODE': (['Panel', 'Stone, brick', 'Block', 'Wooden', 'Mixed', 'Monolithic', 'Others'],
                           [0.44, 0.43, 0.06, 0.035, 0.015, 0.012, 0.008]),
    'EMERGENCYSTATE_MODE': (['No', 'Yes'], [0.986, 0.014]),
}

# Missing-value share per column. Building columns above 0.66 are the ones notebook 01
# drops (it removes columns with more than 66% missing values), as on the real data.
MISSING = {
    'AMT_GOODS_PRICE': 0.001, 'NAME_TYPE_SUITE': 0.01, 'OWN_CAR_AGE': 0.60, 'OCCUPATION_TYPE': 0.31,
    'CNT_FAM_MEMBERS': 0.001, 'EXT_SOURCE_1': 0.56, 'EXT_SOURCE_2': 0.002, 'EXT_SOURCE_3': 0.20,
    'FONDKAPREMONT_MODE': 0.70, 'HOUSETYPE_MODE': 0.50, 'WALLSMATERIAL_MODE': 0.50, 'EMERGENCYSTATE_MODE': 0.47,
    'TOTALAREA_MODE': 0.30,
}
BUILDING = ['APARTMENTS', 'BASEMENTAREA', 'YEARS_BEGINEXPLUATATION', 'YEARS_BUILD', 'COMMONAREA', 'ELEVATORS',
            'ENTRANCES', 'FLOORSMAX', 'FLOORSMIN', 'LANDAREA', 'LIVINGAPARTMENTS', 'LIVINGAREA',
            'NONLIVINGAPARTMENTS', 'NONLIVINGAREA']
BUILDING_DROPPED = {'YEARS_BUILD', 'COMMONAREA', 'FLOORSMIN', 'LIVINGAPARTMENTS', 'NONLIVINGAPARTMENTS'}

FLAG_DOCUMENT_RATE = {3: 0.71, 6: 0.088, 8: 0.081, 5: 0.015, 16: 0.01, 18: 0.008, 9: 0.004, 11: 0.004,
                      13: 0.0035, 14: 0.003, 15: 0.0012}

# Weights of the synthetic logistic rule (on standardised columns)
TARGET_WEIGHTS = {
    'EXT_SOURCE_2': -0.9, 'EXT_SOURCE_3': -0.7, 'DAYS_EMPLOYED': 0.6, 'FLAG_EMP_PHONE': 0.6,
    'IS_PENSIONER': -0.7, 'OBS_30_CNT_SOCIAL_CIRCLE': 0.6, 'AMT_GOODS_PRICE': -0.6, 'AMT_CREDIT': 0.6,
    'APARTMENTS_MEDI': -0.6, 'BASEMENTAREA_MEDI': 0.6, 'ENTRANCES_MEDI': 0.6, 'BASEMENTAREA_MODE': -0.6,
}
TARGET_RATE = 0.08

COLUMNS = (
    ['SK_ID_CURR', 'TARGET', 'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
     'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'NAME_TYPE_SUITE',
     'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
     'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH',
     'OWN_CAR_AGE', 'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE',
     'FLAG_EMAIL', 'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY',
     'WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START', 'REG_REGION_NOT_LIVE_REGION',
     'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',
     'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY', 'ORGANIZATION_TYPE', 'EXT_SOURCE_1', 'EXT_SOURCE_2',
     'EXT_SOURCE_3']
    + [f'{b}_{s}' for s in ['AVG', 'MODE', 'MEDI'] for b in BUILDING]
    + ['FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'TOTALAREA_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE',
       'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE',
       'DEF_60_CNT_SOCIAL_CIRCLE', 'DAYS_LAST_PHONE_CHANGE']
    + [f'FLAG_DOCUMENT_{i}' for i in range(2, 22)]
    + [f'AMT_REQ_CREDIT_BUREAU_{p}' for p in ['HOUR', 'DAY', 'WEEK', 'MON', 'QRT', 'YEAR']]
)


def _flag(rng, n, p):
    """0/1 int column that always contains both values, like every FLAG_* column of the real file."""
    x = (rng.random(n) < p).astype(int)
    x[:2], x[2:4] = 1, 0
    return x


def _intercept_for_rate(score, rate):
    lo, hi = -20.0, 20.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(mid + score)))).mean() < rate:
            lo = mid
        else:
            hi = mid
    return mid


def make_synthetic_application_data(n_rows=6000, seed=0):
    rng = np.random.default_rng(seed)
    n = n_rows
    missing = dict(MISSING)
    d = {'SK_ID_CURR': 100002 + np.arange(n)}

    for col, (levels, probs) in CATEGORIES.items():
        d[col] = rng.choice(levels, size=n, p=probs)

    d['CNT_CHILDREN'] = np.minimum(rng.poisson(0.4, n), 12)
    d['AMT_INCOME_TOTAL'] = np.round(rng.lognormal(11.9, 0.5, n), -2)
    d['AMT_CREDIT'] = np.round(rng.lognormal(13.1, 0.6, n), -2)
    d['AMT_ANNUITY'] = np.round(d['AMT_CREDIT'] * rng.uniform(0.03, 0.08, n), 1)
    d['AMT_GOODS_PRICE'] = np.round(rng.lognormal(13.0, 0.6, n), -3)
    d['REGION_POPULATION_RELATIVE'] = np.round(rng.uniform(0.0003, 0.072, n), 6)
    d['DAYS_BIRTH'] = -rng.integers(7500, 25200, n)
    d['DAYS_EMPLOYED'] = -rng.integers(0, 17900, n)
    d['DAYS_REGISTRATION'] = -rng.integers(0, 24600, n).astype(float)
    d['DAYS_ID_PUBLISH'] = -rng.integers(0, 7200, n)
    d['OWN_CAR_AGE'] = rng.integers(0, 40, n).astype(float)
    d['FLAG_MOBIL'] = _flag(rng, n, 0.9999)
    d['FLAG_EMP_PHONE'] = _flag(rng, n, 0.82)
    d['FLAG_WORK_PHONE'] = _flag(rng, n, 0.2)
    d['FLAG_CONT_MOBILE'] = _flag(rng, n, 0.998)
    d['FLAG_PHONE'] = _flag(rng, n, 0.28)
    d['FLAG_EMAIL'] = _flag(rng, n, 0.057)
    d['CNT_FAM_MEMBERS'] = (1 + rng.poisson(1.2, n)).astype(float)
    d['REGION_RATING_CLIENT'] = rng.choice([1, 2, 3], size=n, p=[0.1, 0.74, 0.16])
    d['REGION_RATING_CLIENT_W_CITY'] = rng.choice([1, 2, 3], size=n, p=[0.11, 0.75, 0.14])
    d['HOUR_APPR_PROCESS_START'] = rng.integers(0, 24, n)
    for col, p in [('REG_REGION_NOT_LIVE_REGION', 0.015), ('REG_REGION_NOT_WORK_REGION', 0.05),
                   ('LIVE_REGION_NOT_WORK_REGION', 0.04), ('REG_CITY_NOT_LIVE_CITY', 0.08),
                   ('REG_CITY_NOT_WORK_CITY', 0.23), ('LIVE_CITY_NOT_WORK_CITY', 0.18)]:
        d[col] = _flag(rng, n, p)
    for col in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']:
        d[col] = rng.beta(4, 3.5, n)
    for b in BUILDING:
        for s in ['AVG', 'MODE', 'MEDI']:
            d[f'{b}_{s}'] = np.round(rng.beta(1.2, 8, n), 4)
            missing[f'{b}_{s}'] = 0.70 if b in BUILDING_DROPPED else 0.30
    d['TOTALAREA_MODE'] = np.round(rng.beta(1.2, 10, n), 4)
    d['OBS_30_CNT_SOCIAL_CIRCLE'] = rng.poisson(1.4, n).astype(float)
    d['DEF_30_CNT_SOCIAL_CIRCLE'] = rng.poisson(0.14, n).astype(float)
    d['OBS_60_CNT_SOCIAL_CIRCLE'] = d['OBS_30_CNT_SOCIAL_CIRCLE'] + rng.poisson(0.05, n)
    d['DEF_60_CNT_SOCIAL_CIRCLE'] = rng.poisson(0.1, n).astype(float)
    d['DAYS_LAST_PHONE_CHANGE'] = -rng.integers(0, 4300, n).astype(float)
    for i in range(2, 22):
        d[f'FLAG_DOCUMENT_{i}'] = _flag(rng, n, FLAG_DOCUMENT_RATE.get(i, 0.001))
    for period, lam in [('HOUR', 0.006), ('DAY', 0.007), ('WEEK', 0.03), ('MON', 0.27), ('QRT', 0.27),
                        ('YEAR', 1.9)]:
        d[f'AMT_REQ_CREDIT_BUREAU_{period}'] = rng.poisson(lam, n).astype(float)
    for col in ['OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE',
                'DEF_60_CNT_SOCIAL_CIRCLE']:
        missing[col] = 0.003

    df = pd.DataFrame(d)

    # Target from the complete (pre-missing) values
    features = df.assign(IS_PENSIONER=(df['NAME_INCOME_TYPE'] == 'Pensioner').astype(float))
    score = np.zeros(n)
    for col, w in TARGET_WEIGHTS.items():
        x = features[col].astype(float)
        score += w * (x - x.mean()) / x.std()
    p = 1 / (1 + np.exp(-(_intercept_for_rate(score, TARGET_RATE) + score)))
    df['TARGET'] = (rng.random(n) < p).astype(int)

    for col, share in missing.items():
        df.loc[rng.random(n) < share, col] = np.nan
    # The credit-bureau counts are missing together, as in the real file
    bureau = [c for c in df.columns if c.startswith('AMT_REQ_CREDIT_BUREAU_')]
    df.loc[rng.random(n) < 0.135, bureau] = np.nan

    return df[COLUMNS]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rows', type=int, default=6000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=Path, default=Path('data/raw_data/application_data.csv'))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = make_synthetic_application_data(args.rows, args.seed)
    df.to_csv(args.out, index=False)
    print(f'{df.shape[0]} rows x {df.shape[1]} columns, positive rate {df["TARGET"].mean():.3f} -> {args.out}')


if __name__ == '__main__':
    main()
