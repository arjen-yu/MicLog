"""
This file is part of TA-Eval-Rep.
Copyright (C) 2022 University of Luxembourg
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 3 of the License.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import print_function

import pandas as pd
from tqdm import tqdm


def evaluate_template_level(dataset, df_groundtruth, df_parsedresult, filter_templates=None):
    correct_parsing_templates = 0
    if filter_templates is not None:
        filter_identify_templates = set()
    null_logids = df_groundtruth[~df_groundtruth["EventTemplate"].isnull()].index
    df_groundtruth = df_groundtruth.loc[null_logids]
    df_parsedresult = df_parsedresult.loc[null_logids]
    series_groundtruth = df_groundtruth["EventTemplate"]
    series_parsedlog = df_parsedresult["EventTemplate"]
    series_groundtruth_valuecounts = series_groundtruth.value_counts()

    df_combined = pd.concat([series_groundtruth, series_parsedlog], axis=1, keys=["groundtruth", "parsedlog"])
    grouped_df = df_combined.groupby("parsedlog")

    for identified_template, group in tqdm(grouped_df):
        corr_oracle_templates = set(list(group["groundtruth"]))
        if filter_templates is not None and len(corr_oracle_templates.intersection(set(filter_templates))) > 0:
            filter_identify_templates.add(identified_template)
        if corr_oracle_templates == {identified_template}:
            if (filter_templates is None) or (identified_template in filter_templates):
                correct_parsing_templates += 1

    if filter_templates is not None:
        pta = correct_parsing_templates / len(filter_identify_templates)
        rta = correct_parsing_templates / len(filter_templates)
    else:
        pta = correct_parsing_templates / len(grouped_df)
        rta = correct_parsing_templates / len(series_groundtruth_valuecounts)
    fta = 0.0
    if pta != 0 or rta != 0:
        fta = 2 * (pta * rta) / (pta + rta)
    print("PTA: {:.4f}, RTA: {:.4f} FTA: {:.4f}".format(pta, rta, fta))
    t1 = len(grouped_df) if filter_templates is None else len(filter_identify_templates)
    t2 = len(series_groundtruth_valuecounts) if filter_templates is None else len(filter_templates)
    print("Identify : {}, Groundtruth : {}".format(t1, t2))
    return t1, t2, fta, pta, rta
