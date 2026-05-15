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

import csv
import os
import time

import pandas as pd

from .GA_calculator import calculate_group_accuracy
from .PA_calculator import calculate_parsing_accuracy
from .template_level_analysis import evaluate_template_level


def prepare_results(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    result_file = "summary.csv"
    with open(os.path.join(output_dir, result_file), "w") as csv_file:
        fw = csv.writer(csv_file, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
        fw.writerow(
            ["Dataset", "parse_time", "identified_templates", "ground_templates", "GA", "PA", "FGA", "PTA", "RTA", "FTA"]
        )

    return result_file


def is_file_empty(file_path):
    with open(file_path, "r") as file:
        content = file.read()
        return len(content) == 0


def align_results(groundtruth_df, parsedresult_df):
    if "LineId" in groundtruth_df.columns and "LineId" in parsedresult_df.columns:
        groundtruth_df = groundtruth_df.copy()
        parsedresult_df = parsedresult_df.copy()
        groundtruth_df["LineId"] = groundtruth_df["LineId"].astype(str)
        parsedresult_df["LineId"] = parsedresult_df["LineId"].astype(str)
        merged = parsedresult_df.merge(
            groundtruth_df,
            on="LineId",
            how="inner",
            suffixes=("_parsed", "_groundtruth"),
        )
        parsed_cols = {
            column.replace("_parsed", ""): merged[column]
            for column in merged.columns
            if column.endswith("_parsed")
        }
        groundtruth_cols = {
            column.replace("_groundtruth", ""): merged[column]
            for column in merged.columns
            if column.endswith("_groundtruth")
        }
        parsed_cols["LineId"] = merged["LineId"]
        groundtruth_cols["LineId"] = merged["LineId"]
        parsedresult_df = pd.DataFrame(parsed_cols)
        groundtruth_df = pd.DataFrame(groundtruth_cols)
        return groundtruth_df.reset_index(drop=True), parsedresult_df.reset_index(drop=True)

    aligned_size = min(len(groundtruth_df), len(parsedresult_df))
    return (
        groundtruth_df.iloc[:aligned_size].reset_index(drop=True),
        parsedresult_df.iloc[:aligned_size].reset_index(drop=True),
    )


def evaluator(dataset, input_dir, output_dir, log_file, LogParser, param_dict, result_file, parse_time_override=None):
    print("\n=== Evaluation on %s ===" % dataset)
    indir = os.path.join(input_dir, os.path.dirname(log_file))
    log_file_basename = os.path.basename(log_file)
    groundtruth = os.path.join(indir, log_file_basename + "_structured.csv")
    parsedresult = os.path.join(output_dir, log_file_basename + "_structured.csv")
    start_time = time.time()
    if LogParser is not None:
        print("start parsing.")
        parser = LogParser(**param_dict)
        print(param_dict)
        parser.parse(log_file_basename)
        print("end parsing.")
        parse_time = time.time() - start_time
    else:
        parse_time = -1 if parse_time_override is None else parse_time_override
    print("parsing time: ", parse_time)

    if not os.path.exists(parsedresult) or is_file_empty(parsedresult):
        print("No output file generated.")
        result = (
            dataset
            + ","
            + "None,None,None,None,None,None,None,None,None\n"
        )
        with open(os.path.join(output_dir, result_file), "a") as summary_file:
            summary_file.write(result)
        return

    parsedresult = pd.read_csv(parsedresult, dtype=str)
    parsedresult.fillna("", inplace=True)
    groundtruth = pd.read_csv(groundtruth, dtype=str)
    groundtruth.fillna("", inplace=True)
    groundtruth, parsedresult = align_results(groundtruth, parsedresult)

    print("Start compute grouping accuracy")
    start_time = time.time()
    ga, fga = calculate_group_accuracy(groundtruth, parsedresult)
    ga_end_time = time.time() - start_time
    print("Grouping Accuracy calculation done. [Time taken: {:.3f}]".format(ga_end_time))

    start_time = time.time()
    pa = calculate_parsing_accuracy(dataset, groundtruth, parsedresult)
    pa_end_time = time.time() - start_time
    print("Parsing Accuracy calculation done. [Time taken: {:.3f}]".format(pa_end_time))

    start_time = time.time()
    tool_templates, ground_templates, fta, pta, rta = evaluate_template_level(dataset, groundtruth, parsedresult)
    ta_end_time = time.time() - start_time
    print("Template-level accuracy calculation done. [Time taken: {:.3f}]".format(ta_end_time))

    result = (
        dataset
        + ","
        + "{:.2f}".format(parse_time)
        + ","
        + str(tool_templates)
        + ","
        + str(ground_templates)
        + ","
        + "{:.1f}".format(ga * 100)
        + ","
        + "{:.1f}".format(pa * 100)
        + ","
        + "{:.1f}".format(fga * 100)
        + ","
        + "{:.1f}".format(pta * 100)
        + ","
        + "{:.1f}".format(rta * 100)
        + ","
        + "{:.1f}".format(fta * 100)
        + "\n"
    )

    with open(os.path.join(output_dir, result_file), "a") as summary_file:
        summary_file.write(result)
