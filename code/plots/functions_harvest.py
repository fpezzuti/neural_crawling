import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from plots.plot_config import *


def plot_multiple_cmetric_across_limits(df_crawl_dict: dict, exp_dict: dict, metric: str = "harvest", printable_metric: str = "Relevant Harvest Rate", xmillions=True, grid=False, markers=MARKERS, lowerbound=None, x_limit=True, ubs_hr=None, ydecmillions=True, sharey: bool = True, leg_pos: str = "best", leg_ncol: int = 2, figsize: tuple = (7,4), markersize: int = 5, linewidth: float =1.2, ttest: bool = False):
    import matplotlib.pyplot as plt

    plt.rcParams["text.usetex"] = True  # enable LaTeX rendering

    n_benchmarks = len(df_crawl_dict)  
    x_figsize = figsize[0] * n_benchmarks
    y_figsize = figsize[1]
    fig, axes = plt.subplots(1, n_benchmarks, figsize=(x_figsize, y_figsize), sharey=sharey)  

    if n_benchmarks == 1:
        axes = [axes]

    legend_elements = []

    for idx_plot, (printable_bname, df_crawl) in enumerate(df_crawl_dict.items()):
        ax = axes[idx_plot]

        if lowerbound is not None:
            ax.axhline(y=lowerbound, color='black', linestyle='--', label='Lower bound')

        tested_limits = df_crawl['limit'].unique()
        tested_df = df_crawl.sort_values(by=['experiment', 'limit'], ascending=[False, True])

        for exp_num, exp_name in enumerate(exp_dict.keys()):
            metric_vals_exp = tested_df[tested_df['experiment'] == exp_name][metric].tolist()

            if not ttest:
                ax.plot(tested_limits, metric_vals_exp, label=exp_dict[exp_name], marker=markers[exp_num], color=COLORS[exp_num], linewidth=linewidth, markersize=markersize)
            else:
                ttest_val = tested_df[tested_df['experiment'] == exp_name]['significant'].tolist()
                x_vals = tested_limits
                y_vals = metric_vals_exp 

                for i, is_significant in enumerate(ttest_val):
                    if is_significant==True:
                        ax.plot(x_vals[i], y_vals[i], label=exp_name, marker=MARKERS[exp_num],linestyle='', color=COLORS[exp_num], linewidth=linewidth, markersize=markersize) 
                    else:
                        ax.plot(x_vals[i], y_vals[i], label=exp_name, marker=MARKERS[exp_num], linestyle='', color=COLORS[exp_num], markerfacecolor='white',  linewidth=linewidth, markersize=markersize) # full circle for significant values
                # draw a line connecting the points
                ax.plot(x_vals, y_vals, label=exp_name, linestyle="-", color=COLORS[exp_num])  #line

            if idx_plot == 0:
                printable_ename = exp_dict[exp_name]
                label = rf"\textsf {printable_ename}"
                legend_elements.append(Line2D([0], [0], marker=MARKERS[exp_num], color=COLORS[exp_num], markerfacecolor=COLORS[exp_num], markersize=markersize, label=label, linestyle="-"))
        if ubs_hr is not None:
            ax.plot(tested_limits, ubs_hr, label="UB", marker='.', color=COLOR_BLIND_PALETTE['black'])

        ax.set_xlabel('Number of crawled pages' + (" (t)" if xmillions else ''))

    
        if x_limit is not False:
            x_lim = tested_limits[-1] + tested_limits[0]
            ax.set_xlim(0, x_lim)
        if xmillions:
            ax.xaxis.set_major_formatter(plt.FuncFormatter(millions_nounit))
            ax.annotate(r'$\times10^{6}$', xy=(1.06, 0.0), xycoords='axes fraction',
                            ha='center', va='bottom', fontsize=12)

        if ydecmillions:
            if idx_plot == 0:
                ax.yaxis.set_major_formatter(plt.FuncFormatter(decmillions_nounit))
                ax.annotate(r'$\times10^{-5}$', xy=(0.02, 1.02), xycoords='axes fraction',
                            ha='center', va='bottom', fontsize=12)
            elif not sharey:
                ax.yaxis.set_major_formatter(plt.FuncFormatter(decmillions_nounit))
                ax.annotate(r'$\times10^{-5}$', xy=(0.02, 1.01), xycoords='axes fraction',
                            ha='center', va='bottom', fontsize=12)
        else:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x:.2f}'))
            
        if grid:        
            ax.grid(axis=grid, alpha=0.7, linestyle='--', color="gray")

        if idx_plot == 0:
            ax.set_ylabel(f"{printable_metric}")

        if idx_plot == (n_benchmarks - 1):
            if leg_ncol > 0:
                ax.legend(loc=leg_pos, ncol=leg_ncol, handles=legend_elements, handlelength=1.2, handleheight=0.8, columnspacing=0.5, labelspacing=0.1)
        
        ax.set_title(f'{printable_bname}')
    fig.tight_layout()
    return fig