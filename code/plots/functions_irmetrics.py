import matplotlib.pyplot as plt
import pandas as pd
import re

import indexer.evaluation as ir_eval

from plots.plot_config import *

def _filter_metric_from_results(df_res: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    df = df_res
    print("df_run: ", df['run'])
    df['experiment'] = df['run'].apply(lambda x: x.split('-')[0])

    df['limit'] = df['run'].apply(lambda x: x.split('_')[1]).apply(lambda x: int(re.search(r'\d+', x).group()))
    return df[["experiment", "limit", f"{metric_name}",]]


def _evaluate_at_many_limits(tested_exps: dict, limits: list, benchmark_name: str, reranked: bool | str) -> pd.DataFrame:
    run_files = []
    run_names = []
    
    if reranked in [True, False]:
        reranked_str = 'reranked-' if reranked else ''

        for exp_name, print_name in tested_exps.items():
            for limit in limits:
                run_files.append(f"./../../data/runs/{exp_name}/limit_{limit}/{benchmark_name}/{reranked_str}limit_{limit}.tsv")
                run_names.append(f"{print_name}-limit_{limit}")
    elif reranked == "all":
        for exp_name, print_name in tested_exps.items():
            for limit in limits:
                for rerank_prefix, rerank_str in [("", ""), ("reranked-", " reranked")]:
                    run_files.append(f"./../../data/runs/{exp_name}/limit_{limit}/{benchmark_name}/{rerank_prefix}limit_{limit}.tsv")
                    run_names.append(f"{print_name}{rerank_str}-limit_{limit}")
    else:
        raise ValueError("reranked must be True, False or 'all'.")

    eval_df = ir_eval.evaluate_multiple_runs(run_files, run_names, benchmark_name)

    return eval_df


def _plot_metric_limit(df: pd.DataFrame, metric_name: str, grid: bool = False, markers: list = MARKERS, xmillions: bool = True, x_limit: bool = True, logy: bool = False):
    fig, ax = plt.subplots(figsize=(8, 6))
    linestyle = "-"
    reranked_linestyle = "--"

    for exp_num, experiment in enumerate(df['experiment'].unique()):
        exp_df = df[df['experiment'] == experiment]
        limit = exp_df['limit']
        metric_val = exp_df[metric_name]
       
        ax.plot(limit, metric_val, label=experiment, marker=markers[exp_num], color=COLORS[exp_num], linestyle=linestyle)

        if 'reranked' in exp_df.columns:
            reranked = exp_df['reranked']
            ax.plot(limit, reranked, label=experiment+" reranked", marker=markers[exp_num], color=COLORS[exp_num], linestyle=reranked_linestyle)


    ax.set_xlabel('Number of crawled pages'+(' (M)' if xmillions else ''))
    ax.set_ylabel(f"{metric_name}")

    if x_limit:
        x_lim_max = df['limit'].max() + df['limit'].min()
        ax.set_xlim(0, x_lim_max)
    if xmillions:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(millions_nounit))

    if logy:
        ax.set_yscale('log')

    ax.legend(loc='lower right')

    ax.grid(grid)

    return fig

def _multiple_plot_metric_limit(all_dfs_dict: dict, benchmarks_dict: dict, metric_name: str, grid: bool = False, markers: list = MARKERS, xmillions: bool = True, x_limit: bool = True, sharey: bool = True, no_legend: bool = False, logy: bool = False, figsize: tuple = (7,4), ttest: bool = False):

    n_benchmarks = len(all_dfs_dict)

    figsize_x = figsize[0] * n_benchmarks
    figsize_y = figsize[1]

    fig, axes = plt.subplots(1, n_benchmarks, figsize=(figsize_x, figsize_y), sharey=sharey)
    if n_benchmarks == 1:
        axes = [axes]

    linestyle = "-"
    reranked_linestyle = "--"

    for idx_plot, (benchmark_name, benchmark_df) in enumerate(all_dfs_dict.items()):
        ax = axes[idx_plot]

        for exp_num, exp_name in enumerate(benchmark_df['experiment'].unique()):
            exp_df = benchmark_df[benchmark_df['experiment'] == exp_name]
            limit = exp_df['limit']
            metric_val = exp_df[metric_name]

            if not ttest:
                ax.plot(limit, metric_val, label=exp_name, marker=markers[exp_num], color=COLORS[exp_num], linestyle=linestyle)
            else:
                ttest_val = exp_df['significant']


                x_vals = limit
                y_vals = metric_val 

                for i, is_significant in enumerate(ttest_val):
                    if is_significant==True:
                        ax.plot(x_vals[i], y_vals[i], label=exp_name, marker='o',linestyle='-', color=COLORS[exp_num], markersize=5) # svd line
                    else:
                        ax.plot(x_vals[i], y_vals[i], label=exp_name, marker='o', linestyle='-', color=COLORS[exp_num], markerfacecolor='white', markersize=5) # full circle for significant values
                # draw a line connecting the points
                ax.plot(x_vals, y_vals, label=exp_name, linestyle=linestyle, color=COLORS[exp_num])  #line line
        
            if 'reranked' in exp_df.columns:
                reranked = exp_df['reranked']
                ax.plot(limit, reranked, label=exp_name+" reranked", marker=markers[exp_num], color=COLORS[exp_num], linestyle=reranked_linestyle)

        ax.set_xlabel('Number of crawled pages'+(r' $t$' if xmillions else ''))
        ax.set_title(benchmarks_dict[benchmark_name])
        
        if x_limit:
            x_lim_max = benchmark_df['limit'].max() + benchmark_df['limit'].min()
            ax.set_xlim(0, x_lim_max)
        if xmillions:
            ax.xaxis.set_major_formatter(plt.FuncFormatter(millions_nounit))

        if logy:
            ax.set_yscale('log')

        if grid:
            ax.grid(axis=grid, linestyle='--', alpha=0.7, zorder=2, color='gray')

        if idx_plot == 0:
            ax.set_ylabel(f"{metric_name}")

        if not no_legend:
            if idx_plot == (n_benchmarks - 1):
                ax.legend(loc='lower right')
    fig.tight_layout()
    return fig


def plot_ir_metric_at_many_limits(metric_name: str, tested_exps: dict, limits: list, tested_benchmarks: dict, ret_df: bool = True, reranked: bool | str = False, single_graph: bool = True, no_legend: bool = False, logy: bool = False, figsize: tuple = (7,4), grid: bool = False, sharey: bool = False, ttest: bool = False) -> list | tuple:
    all_dfs = {}
    all_figs = {}
    for benchmark_name in tested_benchmarks.keys():
        if reranked in [True, False]:
            df_res = _evaluate_at_many_limits(tested_exps, limits=limits, benchmark_name=benchmark_name, reranked=reranked)
            df_metric = _filter_metric_from_results(df_res, metric_name)
        else:
            df_res_unreranked = _evaluate_at_many_limits(tested_exps, limits=limits, benchmark_name=benchmark_name, reranked=False)
            df_res_unreranked = _filter_metric_from_results(df_res_unreranked, metric_name)
            df_res_reranked = _evaluate_at_many_limits(tested_exps, limits=limits, benchmark_name=benchmark_name, reranked=True)
            df_res_reranked = _filter_metric_from_results(df_res_reranked, metric_name)
            df_res_reranked['reranked'] = df_res_reranked[metric_name]
            df_res_reranked = df_res_reranked.drop(metric_name, axis=1)
            df_metric = pd.merge(df_res_unreranked, df_res_reranked, on=["experiment", "limit"])
        
        if not single_graph:
            fig = _plot_metric_limit(df_metric, metric_name, logy=logy)
            all_figs[benchmark_name] = fig
        
        all_dfs[benchmark_name] = df_metric

    if single_graph:
        fig = _multiple_plot_metric_limit(all_dfs, tested_benchmarks, metric_name, no_legend=no_legend, logy=logy, figsize=figsize, grid=grid, sharey=sharey, ttest=ttest)
        all_figs = fig

    if ret_df:
        return all_figs, all_dfs
    else:
        return all_figs