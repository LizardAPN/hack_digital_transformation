from typing import List, Dict
import numpy as np
import pandas as pd

from data.datasets import haversine_m


def geo_metrics_from_indices(df_test: pd.DataFrame, df_train: pd.DataFrame,
                             indices: np.ndarray,
                             ks: List[int] = [1, 5, 10],
                             radii_m: List[float] = [25.0, 50.0, 100.0]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    valid_mask = (~df_test['lat'].isna()) & (~df_test['lon'].isna())
    coverage = 100.0 * float(valid_mask.sum()) / float(len(df_test)) if len(df_test) > 0 else 0.0
    metrics['Coverage_with_coords_%'] = round(coverage, 2)

    top1 = indices[:, 0]
    dists = []
    for i in range(len(df_test)):
        tlat = float(df_test.iloc[i]['lat'])
        tlon = float(df_test.iloc[i]['lon'])
        gidx = int(top1[i])
        glat = float(df_train.iloc[gidx]['lat'])
        glon = float(df_train.iloc[gidx]['lon'])
        dists.append(haversine_m(tlat, tlon, glat, glon))
    metrics['Top1_Distance_mean_m'] = float(np.mean(dists)) if len(dists) > 0 else float('nan')
    metrics['Top1_Distance_median_m'] = float(np.median(dists)) if len(dists) > 0 else float('nan')
    metrics['Top1_Distance_std_m'] = float(np.std(dists)) if len(dists) > 0 else float('nan')

    for rad in radii_m:
        for K in ks:
            hits = []
            rr_list = []
            for i in range(len(df_test)):
                tlat = float(df_test.iloc[i]['lat'])
                tlon = float(df_test.iloc[i]['lon'])
                found_rank = None
                max_rank = min(K, indices.shape[1])
                for rank_i in range(max_rank):
                    gidx = int(indices[i, rank_i])
                    glat = float(df_train.iloc[gidx]['lat'])
                    glon = float(df_train.iloc[gidx]['lon'])
                    d = haversine_m(tlat, tlon, glat, glon)
                    if d <= rad:
                        found_rank = rank_i + 1
                        break
                if found_rank is None:
                    hits.append(0.0)
                    rr_list.append(0.0)
                else:
                    hits.append(1.0)
                    rr_list.append(1.0 / found_rank)
            metrics[f'Geo_Recall@{K}_≤{int(rad)}m'] = float(np.mean(hits))
            metrics[f'Geo_MRR@{K}_≤{int(rad)}m'] = float(np.mean(rr_list))
    return metrics