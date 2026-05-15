"""
Description : This file implements the function to evaluation accuracy of log parsing
Author      : LogPAI team
License     : MIT
"""

import pandas as pd
from tqdm import tqdm


def calculate_group_accuracy(df_groundtruth, df_parsedlog, filter_templates=None):
    """Evaluation function to org_benchmark log parsing accuracy."""
    null_logids = df_groundtruth[~df_groundtruth["EventTemplate"].isnull()].index
    df_groundtruth = df_groundtruth.loc[null_logids]
    df_parsedlog = df_parsedlog.loc[null_logids]
    ga, fga = get_accuracy(df_groundtruth["EventTemplate"], df_parsedlog["EventTemplate"], filter_templates)
    print("Grouping_Accuracy (GA): %.4f, FGA: %.4f," % (ga, fga))
    return ga, fga


def get_accuracy(series_groundtruth, series_parsedlog, filter_templates=None):
    """Compute accuracy metrics between log parsing results and ground truth."""
    series_groundtruth_valuecounts = series_groundtruth.value_counts()
    series_parsedlog_valuecounts = series_parsedlog.value_counts()
    df_combined = pd.concat([series_groundtruth, series_parsedlog], axis=1, keys=["groundtruth", "parsedlog"])
    grouped_df = df_combined.groupby("groundtruth")
    accurate_events = 0
    accurate_templates = 0
    if filter_templates is not None:
        filter_identify_templates = set()
    for ground_truth_id, group in tqdm(grouped_df):
        series_parsedlog_logid_valuecounts = group["parsedlog"].value_counts()
        if filter_templates is not None and ground_truth_id in filter_templates:
            for parsed_event_id in series_parsedlog_logid_valuecounts.index:
                filter_identify_templates.add(parsed_event_id)
        if series_parsedlog_logid_valuecounts.size == 1:
            parsed_event_id = series_parsedlog_logid_valuecounts.index[0]
            if len(group) == series_parsedlog[series_parsedlog == parsed_event_id].size:
                if (filter_templates is None) or (ground_truth_id in filter_templates):
                    accurate_events += len(group)
                    accurate_templates += 1
    if filter_templates is not None:
        ga = float(accurate_events) / len(series_groundtruth[series_groundtruth.isin(filter_templates)])
        pga = float(accurate_templates) / len(filter_identify_templates)
        rga = float(accurate_templates) / len(filter_templates)
    else:
        ga = float(accurate_events) / len(series_groundtruth)
        pga = float(accurate_templates) / len(series_parsedlog_valuecounts)
        rga = float(accurate_templates) / len(series_groundtruth_valuecounts)
    fga = 0.0
    if pga != 0 or rga != 0:
        fga = 2 * (pga * rga) / (pga + rga)
    return ga, fga
