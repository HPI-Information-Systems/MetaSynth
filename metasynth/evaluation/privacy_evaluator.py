import pandas as pd
import numpy as np 
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

class PrivacyEvaluator:
    def __init__(self):
        pass

    def evaluate_default(self, gt, synthetic):
        """
        Returns privacy metrics

        Inputs:
        1) real_path -> path to real data
        2) fake_path -> path to corresponding synthetic data
        3) data_percent -> percentage of data to be sampled from real and synthetic datasets for computing privacy metrics

        Outputs:
        1) List containing the 5th percentile distance to closest record (DCR) between real and synthetic as well as within real and synthetic datasets
        along with 5th percentile of nearest neighbour distance ratio (NNDR) between real and synthetic as well as within real and synthetic datasets

        """

        # Loading real and synthetic datasets and removing duplicates if any
        real = pd.concat(gt).reset_index(drop=True).drop_duplicates(keep=False)
        fake = synthetic.drop_duplicates(keep=False)

        # Scaling real and synthetic data samples
        scalerR = StandardScaler()
        scalerR.fit(real)
        scalerF = StandardScaler()
        scalerF.fit(fake)
        df_real_scaled = scalerR.transform(real)
        df_fake_scaled = scalerF.transform(fake)

        # Computing pair-wise distances between real and synthetic 
        dist_rf = metrics.pairwise_distances(df_real_scaled, Y=df_fake_scaled, metric='minkowski', n_jobs=-1)

        # Computing first and second smallest nearest neighbour distances between real and synthetic
        smallest_two_indexes_rf = [dist_rf[i].argsort()[:2] for i in range(len(dist_rf))]
        smallest_two_rf = [dist_rf[i][smallest_two_indexes_rf[i]] for i in range(len(dist_rf))]       

        # Computing 5th percentiles for DCR and NNDR between and within real and synthetic datasets
        min_dist_rf = np.array([i[0] for i in smallest_two_rf])
        fifth_perc_rf = np.percentile(min_dist_rf,5)


        return fifth_perc_rf
    
class DCREvaluator:
    def __init__(self):
        pass

    def evaluate_default(self, gt, synthetic):
        """
        Returns privacy metrics

        Inputs:
        1) real_path -> path to real data
        2) fake_path -> path to corresponding synthetic data
        3) data_percent -> percentage of data to be sampled from real and synthetic datasets for computing privacy metrics

        Outputs:
        1) List containing the 5th percentile distance to closest record (DCR) between real and synthetic as well as within real and synthetic datasets
        along with 5th percentile of nearest neighbour distance ratio (NNDR) between real and synthetic as well as within real and synthetic datasets

        """

        real = pd.concat(gt).reset_index(drop=True)
        fake = synthetic

        # Scaling real and synthetic data samples
        scalerR = StandardScaler()
        scalerR.fit(real)
        scalerF = StandardScaler()
        scalerF.fit(fake)
        df_real_scaled = scalerR.transform(real)
        df_fake_scaled = scalerF.transform(fake)

        # Computing pair-wise distances between real and synthetic 
        dist_rf = metrics.pairwise_distances(df_real_scaled, Y=df_fake_scaled, metric='minkowski', n_jobs=-1)

        return dist_rf.min(axis=0).mean()
    
class DublicateEvaluator:
    def __init__(self):
        pass

    def evaluate_default(self, gt, synthetic):
        """
        Returns privacy metrics

        Inputs:
        1) real_path -> path to real data
        2) fake_path -> path to corresponding synthetic data
        3) data_percent -> percentage of data to be sampled from real and synthetic datasets for computing privacy metrics

        Outputs:
        1) List containing the 5th percentile distance to closest record (DCR) between real and synthetic as well as within real and synthetic datasets
        along with 5th percentile of nearest neighbour distance ratio (NNDR) between real and synthetic as well as within real and synthetic datasets

        """

        fake = synthetic
        # Counting the number of duplicates in the fake dataframe
        num_duplicates = (fake.duplicated().sum() * 1000) / len(fake)
        return int(num_duplicates)