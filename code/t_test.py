from ir_measures import *

from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

import numpy as np
import pandas as pd

from indexer.evaluation import compute_metrics_from_qrels


def compute_ir_metrics_for_ttest(benchmark_name: str, path_to_run: str, metrics) -> list:
    """
        Compute the IR metrics for a given run in TREC format in a list, suitable for running t-test.

        Args:
            benchmark_name: str, the name of the benchmark.
            path_to_run: str, the path to the run file.
            metrics: list, the list of metrics to compute.
    """
    results = compute_metrics_from_qrels(path_to_run=path_to_run, benchmark_name=benchmark_name, metrics=metrics, aggregate=False)
   
    df_results = pd.DataFrame(results, columns=['qid', 'metric', 'metric_value'])  # prepare run's dataframe 
    df_results.replace(metrics, np.arange(len(metrics)), inplace=True) # transform metric name to id (int)
    df_results.sort_values(['metric', 'qid'], inplace=True) # sort by metric id (int)
    df_results.reset_index(drop=True, inplace=True)

    return df_results['metric_value'].to_list()



def twosided_paired_ttest(run_res, baseline_res, metrics: list, alpha_level: float = 0.05, method: str = "ttest-twosided") -> tuple:
    """
        Compute the significance with a two-sided paired t-test, at significance {alpha_level}
        for the dataframe of IR effectiveness metrics computed on a query set with the compute_ir_metrics_for_ttest method.

        Args:
            run_res: metrics computed on the tested run
            baseline_res: metrics computed on the baseline run
            metrics: list of metrics to be tested
            alpha_level: significance level for t-test
            method: the method to use for significance testing (default: ttest-twosided)
    """
    n_queries = int(len(baseline_res) / len(metrics))

    ttest_res_dict = {}
    pvalues_res_dict = {}
    
    for metric_id, metric in enumerate(metrics):
        start_id = metric_id*n_queries
        end_id = start_id + n_queries
        base_vals = baseline_res[start_id:end_id]
        run_vals = run_res[start_id:end_id]
        assert len(base_vals) == n_queries and len(run_vals) == n_queries, f"Error. Baseline values={len(base_vals)}, Run values={len(run_vals)}, queries={n_queries}"
        
        if method == "ttest-twosided": # two-sided paired t-test
            ttest_res = stats.ttest_rel(run_vals, base_vals, nan_policy='raise')
            significant_test = ttest_res.pvalue < alpha_level
            ttest_res_dict[str(metric)] = significant_test
            pvalues_res_dict[str(metric)] = ttest_res.pvalue
        else:
            raise ValueError("Invalid method. Supported methods: 'ttest-twosided'")
    
    ttest_df = pd.DataFrame.from_dict(ttest_res_dict.items()).T
    ttest_df.columns = [str(metric) for metric in metrics] # add header
    ttest_df = ttest_df[1:] # drop first row (the old header)
    pvalues_df = pd.DataFrame.from_dict(pvalues_res_dict.items()).T
    pvalues_df = pvalues_df[1:]
    pvalues_df.columns = [str(metric) for metric in metrics] # add header
 
    return ttest_df, pvalues_df


def twosided_paired_general_ttest(run_vals: list, base_vals: list, metric_name: str, alpha_level: float = 0.05, method: str = "wilcoxon-similar") -> tuple:
    """
        Compute the significance with a two-sided paired t-test, at significance {alpha_level} for two given lists of values.

        Args:
            run_vals: list of values computed on the tested run
            base_vals: list of values computed on the baseline run
            metric_name: the name of the metric
            alpha_level: significance level for t-test
            method: the method to use for significance testing (default: ttest-twosided)
    """
    ttest_res_dict = {}
    pvalues_res_dict = {}
    assert len(base_vals) == len(run_vals), f"Error. Baseline contains {len(base_vals)} values, Run contains {len(run_vals)} values."
    
    if method == "ttest-twosided":
        ttest_res = stats.ttest_rel(run_vals, base_vals, nan_policy='raise')
        significant_test = ttest_res.pvalue < alpha_level
        ttest_res_dict[metric_name] = significant_test
        pvalues_res_dict[metric_name] = ttest_res.pvalue
    else:
        raise ValueError("Invalid method. Supported methods: 'ttest-twosided'")
    
    ttest_df = pd.DataFrame.from_dict(ttest_res_dict.items()).T
    ttest_df.columns = [metric_name] # add header
    ttest_df = ttest_df[1:] # drop first row (the old header)
    pvalues_df = pd.DataFrame.from_dict(pvalues_res_dict.items()).T
    pvalues_df = pvalues_df[1:]
    pvalues_df.columns = [metric_name] # add header
   
    return ttest_df, pvalues_df



def evaluate_ir_significance_difference(run_path: str, baseline_run_path: str, benchmark_name: str, metrics: list, method: str = "ttest-twosided", alpha_level: float = 0.05) -> tuple:
    """
        Compute the significance with a two-sided paired t-test for two given runs in TREC format.

        Args:
            run_path: str, the path to the run file of the tested run.
            baseline_run_path: str, the path to the run file of the baseline run.
            benchmark_name: str, the name of the benchmark.
            metrics: list, the list of metrics to compute.
            method: the method to use for significance testing (default: ttest-twosided)
            alpha_level: significance level for t-test
    """
    base_res = compute_ir_metrics_for_ttest(benchmark_name, baseline_run_path, metrics=metrics)
    run_res = compute_ir_metrics_for_ttest(benchmark_name, run_path, metrics=metrics)
    return twosided_paired_ttest(run_res, base_res, metrics, method=method, alpha_level=alpha_level)


def twosided_ztest_proportion(run_success: int, base_success: int, sample_size: int, alpha_level: float = 0.05) -> tuple:  
    """
        Compute the significance with a two-sided z-test for two given proportions.

        Args:
            run_success: int, the number of successes in the tested run.
            base_success: int, the number of successes in the baseline run.
            sample_size: int, the sample size.
            alpha_level: significance level for z-test
    """
    z_score, p_value = proportions_ztest(count=np.array([run_success, base_success]), nobs=np.array([sample_size, sample_size]), alternative='two-sided')
    significant_test = p_value < alpha_level
    return significant_test, p_value
