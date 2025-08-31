import pandas as pd
import numpy as np
from scipy.spatial import distance
from scipy.stats import wasserstein_distance
from scipy.stats import chi2_contingency
import statsmodels.formula.api as smf
import concurrent.futures

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


def pseudo_r_squared(cat_col, cont_col):
    """
    Determines the effect of a continuous column on a categorical column using logistic regression.

    Parameters:
    cat_col (pd.Series): Categorical column.
    cont_col (pd.Series): Continuous column.

    Returns:
    float: Pseudo R-squared value indicating the strength of influence.
    """
    def compute_pseudo_r_squared():
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
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(compute_pseudo_r_squared)
        try:
            result = future.result(timeout=10)
        except concurrent.futures.TimeoutError:
            print("TimeoutError: Pseudo R-squared calculation took too long.")
            result = 0
    
    return result

def prsquared_correlations(df, types):
    cat_cols = [col for col in types.keys() if types[col]=="str"]
    cont_cols = [col for col in types.keys() if types[col]!="str"]
    
    data = {col: [] for col in types.keys()}
    for col1 in types.keys():
        for col2 in types.keys():
            if col1 in cont_cols and col2 in cat_cols:
                
                data[col1].append(pseudo_r_squared(df[col1], df[col2]))
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
            elif col1 in cont_cols and col2 in cat_cols:
                data[col1].append(eta_squared(df[col2], df[col1]))
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

class TargetCorrelationPerformanceEvaluator:
    def __init__(self, target_column):
        self.target_column = target_column

    def evaluate_default(self, gt, synthetic):
        _, gt = gt
        
        # Calculate correlations of the target column with all other columns
        gt_corr = gt.corr()[self.target_column].drop(self.target_column).to_numpy()
        synthetic_corr = synthetic.corr()[self.target_column].drop(self.target_column).to_numpy()

        # Calculate and return the mean absolute difference between correlations
        return abs(gt_corr - synthetic_corr).mean()

class CorrelationPerformanceEvaluator:
    def __init__(self):
        pass

    def evaluate_default(self, gt, synthetic):
        _, gt = gt
        
        self.gt_corr = gt.corr().to_numpy()
        self.synthetic_corr = synthetic.corr().to_numpy()

        return abs(self.gt_corr - self.synthetic_corr).mean()
    
class MixedCorrelationPerformanceEvaluator:
    def __init__(self, types):
        self.types = types

    def evaluate_default(self, gt, synthetic):
        _, gt = gt
        
        self.gt_corr = eta_correlations(gt, self.types) + cramers_correlations(gt, self.types) + pearson_correlations(gt, self.types)# + prsquared_correlations(gt, self.types)
        self.synthetic_corr = eta_correlations(synthetic, self.types) + cramers_correlations(synthetic, self.types) + pearson_correlations(synthetic, self.types)# + prsquared_correlations(gt, self.types)

        return abs(self.gt_corr.fillna(0).to_numpy() - self.synthetic_corr.fillna(0).to_numpy()).mean()
    
class JSDPerformanceEvaluator:
    def __init__(self, columns):
        self.columns = columns

    def evaluate_default(self, gt, synthetic):
        _, gt = gt
        
        values = []
        for col in self.columns:
            length = min(len(gt), len(synthetic))
            values.append(distance.jensenshannon(gt[col].to_numpy().astype(np.int64)[:length], synthetic[col].to_numpy().astype(np.int64)[:length]))

        return np.mean(values)
    
class WDPerformanceEvaluator:
    def __init__(self, columns, scaler=None):
        self.columns = columns
        self.scaler = scaler

    def evaluate_default(self, gt, synthetic):
        _, gt = gt
        
        if self.scaler is not None:
            gt = pd.DataFrame(self.scaler.transform(gt), columns=gt.columns)
            synthetic = pd.DataFrame(self.scaler.transform(synthetic), columns=synthetic.columns)
        
        values = []
        for col in self.columns:
            length = min(len(gt), len(synthetic))
            values.append(wasserstein_distance(gt[col].to_numpy()[:length], synthetic[col].to_numpy()[:length]))

        return np.mean(values)
    