import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.stats import chi2_contingency
import statsmodels.formula.api as smf
from sklearn.mixture import GaussianMixture
from scipy import stats

def pseude_r_squared(cat_col, cont_col):
    """
    Determines the effect of a continuous column on a categorical column using logistic regression.

    Parameters:
    cat_col (pd.Series): Categorical column.
    cont_col (pd.Series): Continuous column.

    Returns:
    float: Pseudo R-squared value indicating the strength of influence.
    """
    if cat_col.nunique() == 2:
        # Binary logistic regression
        data = pd.DataFrame({'cat_col': cat_col, 'cont_col': cont_col})
        model = smf.logit('cat_col ~ cont_col', data=data).fit()
    else:
        # Multinomial logistic regression
        data = pd.DataFrame({'cat_col': cat_col, 'cont_col': cont_col})
        model = smf.mnlogit('cat_col ~ cont_col', data=data).fit()
    
    # Calculate Pseudo R-squared
    pseudo_r_squared = model.prsquared
    
    return pseudo_r_squared

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(k-1, r-1))))

def eta_squared(categorical, continuous):
    # Berechne den Gesamtmittelwert der kontinuierlichen Variable
    overall_mean = continuous.mean()
    
    # Gruppiere nach der kategorialen Variable und berechne die Varianzkomponenten
    categories = np.unique(categorical)
    
    # Erklärte Varianz (Summe der quadrierten Abweichungen der Gruppenmittelwerte vom Gesamtmittelwert)
    ss_between = sum(len(continuous[categorical == category]) * (continuous[categorical == category].mean() - overall_mean) ** 2
                     for category in categories)
    
    # Gesamtvarianz (Summe der quadrierten Abweichungen der einzelnen Werte vom Gesamtmittelwert)
    ss_total = sum((continuous - overall_mean) ** 2)
    
    # Eta-Quadrat berechnen
    eta_squared_value = ss_between / ss_total
    return np.sqrt(eta_squared_value)

def prsquared_correlations(df, types):
    cat_cols = [col for col in types.keys() if types[col]=="str"]
    cont_cols = [col for col in types.keys() if types[col]!="str"]
    
    data = {col: [] for col in types.keys()}
    for col1 in types.keys():
        for col2 in types.keys():
            if col1 in cont_cols and col2 in cat_cols:
                data[col1].append(pseude_r_squared(df[col1], df[col2]))
            else:
                data[col1].append(0)
                
    return pd.DataFrame(data, index=types.keys())

def eta_correlations(df, types):
    cat_cols = [col for col in types.keys() if types[col]=="str"]
    cont_cols = [col for col in types.keys() if types[col]!="str"]
    
    data = {col: [] for col in types.keys()}
    for col1 in types.keys():
        for col2 in types.keys():
            if col1 in cat_cols and col2 in cont_cols:
                data[col1].append(eta_squared(df[col1], df[col2]))
            else:
                data[col1].append(0)
                
    return pd.DataFrame(data, index=types.keys())

def cramers_correlations(df, types):
    cat_cols = [col for col in types.keys() if types[col]=="str"]
    
    data = {col: [] for col in types.keys()}
    for col1 in types.keys():
        for col2 in types.keys():
            if col1 in cat_cols and col2 in cat_cols:
                data[col1].append(cramers_v(df[col1], df[col2]))
            else:
                data[col1].append(0)
                
    return pd.DataFrame(data, index=types.keys())

def pearson_correlations(df, types):
    cont_cols = [col for col in types.keys() if types[col]!="str"]
    
    data = df.corr()
    for col1 in types.keys():
        for col2 in types.keys():
            if not (col1 in cont_cols and col2 in cont_cols):
                data.loc[col1, col2] = 0
                
    return data

def preprocess_df(df, descriptions, types, fast=False):
    metadata = {}
    
    cols = df.columns.tolist()
    for col in cols:
        vals = df[col].tolist()
        if types[col] == "str": # Object
            unique_values = [str(s) for s in df[col].unique().tolist()]
            df[col+'_int'] = [unique_values.index(str(item)) for item in df[col]]
            df[col] = df[col].astype(str)
            vals = [str(v) for v in vals]
            metadata[col] = {
                'unique': unique_values,
                'probs': [round(vals.count(val) / len(vals), 3) for val in unique_values],
                'description': descriptions[col],
                'type': 'str'
            }
            metadata[col+'_int'] = {
                'unique': list(range(len(unique_values))),
                'probs': [round(vals.count(val) / len(vals), 3) for val in unique_values],
                'description': descriptions[col],
                'type': 'int'
            }
        elif types[col] in ["int", "float"]: # Any int
            #print(col)
            metadata[col] = {
                'mean': float(np.mean(df[col])),
                'std': float(np.std(df[col])),
                'min': float(np.min(df[col])),
                'max': float(np.max(df[col])),
                '1%': float(np.percentile(df[col], 1)),
                '10%': float(np.percentile(df[col], 10)),
                '25%': float(np.percentile(df[col], 25)),
                '50%': float(np.percentile(df[col], 50)),  # or np.median(df[col])
                '75%': float(np.percentile(df[col], 75)),
                '90%': float(np.percentile(df[col], 90)),
                '99%': float(np.percentile(df[col], 99)),
                'skew': float(skew(df[col])),
                'kurtosis': float(kurtosis(df[col])),
                'description': descriptions[col],
                'type': types[col],
                #'hist': np.round(np.histogram(df[col][(df[col] >= np.percentile(df[col], 1)) & (df[col] <= np.percentile(df[col], 99))], bins=10)[0]/len(df[col]), 3).tolist(),
                #'hist_borders': np.histogram(df[col][(df[col] >= np.percentile(df[col], 1)) & (df[col] <= np.percentile(df[col], 99))], bins=10)[1].tolist()
                
            }
            
            X = df[col].values
            X = X[~np.isnan(X)]  # Remove NaNs
            X_sorted = np.sort(X)

            # List of candidate distributions
            candidate_distributions = {
                'norm': stats.norm,
                'expon': stats.expon,
                'uniform': stats.uniform,
                #'lognorm': stats.lognorm,
            }

            best_fit = None
            best_p = -1  # For K-S test, higher p-value means better fit
            best_stat = np.inf

            for name, dist in candidate_distributions.items():
                try:
                    # Fit distribution to data
                    params = dist.fit(X)
                    
                    # Perform Kolmogorov-Smirnov test
                    D, p = stats.kstest(X, name, args=params)

                    if p > best_p or (p == best_p and D < best_stat):
                        best_fit = name
                        best_p = p
                        best_stat = D
                except Exception as e:
                    print(f"Error fitting {name} to {col}: {e}")
                    continue  # Some distributions may not fit well and raise errors

            # Save to metadata
            metadata[col]["distribution_type"] = best_fit
            # if best_fit == "lognorm":
            #     metadata[col]["distribution_params"] = {
            #         'shape': params[0],
            #         'scale': params[1],
            #         'loc': params[2]
            #     }
            if best_fit == "norm":
                metadata[col]["distribution_params"] = {
                    'mean': params[0],
                    'std': params[1]
                }
            elif best_fit == "expon":
                metadata[col]["distribution_params"] = {
                    'scale': params[0]
                }
            elif best_fit == "uniform":
                metadata[col]["distribution_params"] = {
                    'loc': params[0],
                    'scale': params[1]
                }
                        # Add GMM fitting to find multiple means
            if not fast:
                try:
                    X = df[col].values.reshape(-1, 1)
                    n_components_range = range(1, 11)
                    best_gmm = None
                    lowest_bic = np.inf
                    relative_threshold = 0.10  # Require at least 10% BIC improvement

                    for n in n_components_range:
                        gmm = GaussianMixture(n_components=n, random_state=42)
                        gmm.fit(X)
                        bic = gmm.bic(X)

                        if lowest_bic == np.inf or (lowest_bic - bic) / abs(lowest_bic) >= relative_threshold:
                            best_gmm = gmm
                            lowest_bic = bic
                        else:
                            break  # Stop if BIC doesn't improve by 10% or more

                    # Use the best model found
                    gmm_means = sorted(best_gmm.means_.flatten().tolist())
                    metadata[col]['gmm_means'] = gmm_means
                    metadata[col]['gmm_weights'] = best_gmm.weights_.tolist()
                    metadata[col]['gmm_stds'] = [float(np.sqrt(np.diag(cov)[0])) for cov in best_gmm.covariances_]
                except Exception as e:
                    metadata[col]['gmm_error'] = str(e)
        else:
            print(df[col].dtype)

    cols = df.columns.tolist()
    cols = [col for col in cols if not col+'_int' in cols]
    df = df[cols]
    df.columns = [col.split('_int')[0] for col in df.columns.tolist()]
    
    # Get correlations after dropping non numeric columns
    if fast:
        corr_matrix = df.corr()
    else:
        corr_matrix = eta_correlations(df, types) + cramers_correlations(df, types) + pearson_correlations(df, types)
    for name_col in corr_matrix.columns:
        corr_matrix.loc[name_col, name_col] = 0
    corr_matrix = corr_matrix.applymap(lambda x: x if abs(x) > 0.2 else 0)
    
    for name_col in corr_matrix.columns:
        metadata[name_col]["correlations"] = {other_col: round(value, 2) for other_col, value in corr_matrix[name_col].items() if value != 0}
    
    if not fast:
        cat_cols = [col for col in types.keys() if metadata[col]["type"]=="str"]
        cont_cols = [col for col in types.keys() if metadata[col]["type"]!="str"]

        for key_col in types.keys():
            for other_col in metadata[key_col]["correlations"].keys():
                
                if key_col in cont_cols and other_col in cont_cols:
                    continue
                
                if "relationships" not in metadata[key_col]:
                    metadata[key_col]["relationships"] = {other_col: {}}
                else:
                    metadata[key_col]["relationships"][other_col] = {}
                
                # CAT CAT      
                if key_col in cat_cols and other_col in cat_cols:
                    for label in metadata[other_col+"_int"]['unique']:
                        group_df = df[df[other_col] == label]
                        vals = group_df[key_col].tolist()
                        global_vals = df[key_col].tolist()
                        unique_values = metadata[key_col+"_int"]['unique']
                        metadata[key_col]["relationships"][other_col][metadata[other_col]["unique"][label]] = {
                            'unique': [metadata[key_col]['unique'][i] for i in unique_values],
                            'probs': [round((vals.count(val) / max(1,len(vals))) - (global_vals.count(val) / max(1,len(global_vals))), 3) for val in unique_values],
                        }
                
                # CAT CONT
                if key_col in cat_cols and other_col in cont_cols:
                    big_df = df[df[other_col] >= df[other_col].quantile(0.5)]
                    small_df = df[df[other_col] < df[other_col].quantile(0.5)]
                    
                    big_vals = big_df[key_col].tolist()
                    small_vals = small_df[key_col].tolist()
                    global_vals = df[key_col].tolist()
                    unique_values = metadata[key_col+"_int"]['unique']
                    metadata[key_col]["relationships"][other_col]["big"] = {
                        'unique': [metadata[key_col]['unique'][i] for i in unique_values],
                        'probs': [round((big_vals.count(val) / max(1,len(big_vals))) - (global_vals.count(val) / max(1,len(global_vals))), 3) for val in unique_values],
                    }
                    metadata[key_col]["relationships"][other_col]["small"] = {
                        'unique': [metadata[key_col]['unique'][i] for i in unique_values],
                        'probs': [round((small_vals.count(val) / max(1,len(small_vals))) - (global_vals.count(val) / max(1,len(global_vals))), 3) for val in unique_values],
                    }
                    
                # CONT CAT
                if key_col in cont_cols and other_col in cat_cols:
                    grouped_stats = df.groupby(other_col)[key_col].agg(['mean', 'std'])
                    grouped_stats["mean"] = (grouped_stats["mean"] - df[key_col].mean()).round(2)
                    grouped_stats["std"] = (grouped_stats["std"] - df[key_col].std()).round(2)
                    for label in grouped_stats.index:
                        metadata[key_col]["relationships"][other_col][metadata[other_col]["unique"][label]] = {
                            'mean': grouped_stats.loc[label]['mean'],
                            'std': grouped_stats.loc[label]['std'],
                        }
                             
    return df, metadata
