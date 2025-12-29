import sys
import math

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg

sys.path.append('/Users/pav/Documents/SIA/Seat_Study2/Code')
from mystats import MyStatsFunctions

class MyPlots:
    def two_sample_plot(self, data1, label1, data2, label2, bin_width, start_width=None, end_width=None, xlabel=None, ylabel=None, title=None):
        if start_width == None or end_width == None:
            bins = np.arange(math.floor(min(min(data1), min(data2))), math.ceil(max(max(data1), max(data2)) + bin_width), bin_width)
        else:
            bins = np.arange(start_width, end_width, bin_width)
        
        sns.histplot(data1, kde=True, bins=bins, alpha=0.4, label=label1)
        sns.histplot(data2, kde=True, bins=bins, alpha=0.4, label=label2)

        if not xlabel is None:
            plt.xlabel(xlabel)

        if not ylabel is None:
            plt.ylabel(ylabel)

        if not title is None:
            plt.title(title)
            
        plt.legend()
        plt.show()
        
        print(f'Mean: {label1}: {np.mean(data1):.3f} {label2}: {np.mean(data2):.3f}')
        print(f'Median: {label1}: {np.median(data1):.3f} {label2}: {np.median(data2):.3f}')

        print("Hypothesis test")
        msf = MyStatsFunctions()
        msf.perform_analysis([data1, data2], [label1, label2])

        print(f"Effect size {pg.compute_effsize(data1, data2, eftype='hedges'):.3f}")

    def one_sample_plot(self, data, col, label, bin_width, start_width=None, end_width=None, hue=None, hue_labels=None, xlabel=None, ylabel=None, ylim_low=None, ylim_high=None, title=None):
        if start_width == None or end_width == None:
            bins = np.arange(math.floor(data[col].min()), math.ceil(data[col].max() + bin_width), bin_width)
        else:
            bins = np.arange(start_width, end_width, bin_width)

        if not hue is None:
            sns.histplot(data=data, x=col, kde=True, bins=bins, hue=hue, alpha=0.4)
            if not xlabel is None:
                plt.xlabel(xlabel)

            if not ylabel is None:
                plt.ylabel(ylabel)
    
            if not title is None:
                plt.title(title)

            if not hue_labels is None:
                plt.legend(labels=hue_labels)
            
            plt.show()

    
        sns.histplot(data=data, x=col, kde=True,alpha=0.4, bins=bins, label=label)
        if not xlabel is None:
            plt.xlabel(xlabel)

        if not ylabel is None:
            plt.ylabel(ylabel)

        if not title is None:
            plt.title(title)

        if not ylim_low is None and not ylim_high is None:
            plt.ylim(ylim_low, ylim_high)

        # ax.axvline(data[col].median(), color='red', linestyle='--', linewidth=2, label='Median')
        # ax.axvline(data[col].mean(), color='yellow', linestyle='--', linewidth=2, label='Median')
            
        plt.legend()
        plt.show()

        data = data[col]

        print("Hypothesis test")
        print("Normal" if MyStatsFunctions().normal_distribution(data) else "Not normal")
        t_stat, p_value_two = stats.ttest_1samp(data, alternative='two-sided', popmean=0)
        t_stat, p_value_greater = stats.ttest_1samp(data, alternative='greater', popmean=0)
        t_stat, p_value_less = stats.ttest_1samp(data, alternative='less', popmean=0)
        print(f"T: {t_stat:.3f} p-value (2 side): {p_value_two:.3f} p-value (greater): {p_value_greater:.4f} p-value (lesser): {p_value_less:.4f}")

        result_two = stats.wilcoxon(np.array(data) - 0, alternative='two-sided')
        result_greater = stats.wilcoxon(np.array(data) - 0, alternative='greater')
        result_less = stats.wilcoxon(np.array(data) - 0, alternative='less')
        print(f"(Wilcoxon) T: {result_two.statistic:.3f} p-value (2 side): {result_two.pvalue:.3f} p-value (greater): {result_greater.pvalue:.3f} p-value (lesser): {result_less.pvalue:.3f}")
    
        result = pg.ttest(x=data, y=0.0, paired=False)
        d = result['cohen-d'].values[0]
        n = len(data)
        J = 1 - (3 / (4 * n - 1))
        g = d * J
        
        print(f"Cohen's d: {d:.3f} Hedges' g: {g:.3f}")