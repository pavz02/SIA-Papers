import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import seaborn as sns
import math
import datetime
from scipy import stats
import pingouin as pg 
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from scikit_posthocs import posthoc_dunn
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
import pickle

class MyStatsFunctions:
    def z_score(self, x):
        return 0.5 * np.log((1 + x) / (1 - x))
    
    # Shapiro for normal distribution
    def normal_distribution(self, x):
        # return "normal" if stats.shapiro(z_score(x)).pvalue > 0.05 else "not normal"
        p_val = stats.shapiro(x).pvalue
        return p_val > 0.05
    
    # Levene for difference in variance
    def levene(self, xg):
        # return "equal" if stats.levene(*xg).pvalue > 0.05 else "not equal"
        p_val = stats.levene(*xg).pvalue
        return p_val > 0.05
    
    ###############################################################################################
    
    def anova(self, xg):
        p_value = stats.f_oneway(*xg).pvalue
        print(p_value)
        return p_value > 0.05
    
    def anova_posthoc(self, group):
        return pairwise_tukeyhsd(group['value'], group['group'], alpha=0.05)
    
    ###############################################################################################
    
    def welch_anova(self, group):
        p_value = pg.welch_anova(dv='value', between='group', data=group).values[0][4]
        print(p_value)
        return p_value > 0.5
    
    def welch_posthoc(self, group):
        return pg.pairwise_gameshowell(dv='value', between='group', data=group)[['A', 'B', 'pval']]
    
    ###############################################################################################
    
    # Kruskal-Wallis for difference in the groups without a normal distribution
    def kruskal(self, xg):
        p_value = stats.kruskal(*xg).pvalue
        print(p_value)
        
        #return "is no" if stats.kruskal(*xg).pvalue > 0.05 else "is a"
        return p_value > 0.05
    
    # Identify the groups with a difference
    def dunn(self, xg, features):
        data = pd.DataFrame({'z_score': np.concatenate(xg), 'group': np.repeat(features, [len(g) for g in xg])})
        dunn_results = posthoc_dunn(data, val_col='z_score', group_col='group', p_adjust='bonferroni')
    
        for i in range(0, len(features)):
            for j in range(i+1, len(features)):
                if dunn_results.loc[features[i], features[j]] < 0.05:
                    print(f'{features[i]} and {features[j]} {dunn_results.loc[features[i], features[j]]:.3f}')

    # Identify the groups with a difference
    def ks_pairwise(self, xg, features):
        for i in range(len(xg)):
            for j in range(i + 1, len(xg)):
                stat, p_value = stats.ks_2samp(xg[i], xg[j])
                print(f'{features[i]} {features[j]} {p_value:.3f}')
                if p_value < 0.05:
                    print(f'{features[i]} and {features[j]} are significantly different (p = {p_value:.3f})')
    
    ###############################################################################################
    
    def perform_analysis(self, z_groups, features, verbose=True):
        normally_distributed = True
        for i, corrf in enumerate(features):
            value = self.normal_distribution(z_groups[i])
            if not value:
                normally_distributed = False
    
        equal_variance = self.levene(z_groups)
    
        if normally_distributed and equal_variance:
            print('Normal distribution and equal variance... Using Anova and Tukey tests...')
            if not self.anova(z_groups):
                data = pd.DataFrame({'value': np.concatenate(z_groups), 'group': np.repeat(features, [len(g) for g in z_groups])})
                result = self.anova_posthoc(data)
                result_df = pd.DataFrame(result.summary().data[1:], columns=result.summary().data[0])
                pd.set_option('display.max_rows', 200)
                display(result_df[result_df['reject']])
                pd.reset_option('display.max_rows')
            else:
                print('No significant difference...')
        
        elif normally_distributed and not equal_variance:
            print('Normal distribution and unequal variance... Using Welch anova and Pairwise Games-Howell post-hoc tests...')
            data = pd.DataFrame({'value': np.concatenate(z_groups), 'group': np.repeat(features, [len(g) for g in z_groups])})
            if not self.welch_anova(data):
                pd.set_option('display.max_rows', 200)
                display(self.welch_posthoc(data))
                pd.reset_option('display.max_rows')
            else:
                print('No significant difference...')
                
        else:
            # print('Non-normal distribution... Using KS tests...')
            # self.ks_pairwise(z_groups, features)
            
            print('Non-normal distribution... Using Kruskal and Dunn posthoc tests...')
            if not self.kruskal(z_groups):
                print('The following are statistically different:')
                self.dunn(z_groups, features)
            else:
                print('No significant difference...')
