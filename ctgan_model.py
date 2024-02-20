import numpy as np
import os, sys
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from collections import Counter
from tqdm.auto import tqdm, trange
from sklearn.preprocessing import MinMaxScaler
import re
import concurrent.futures
from sklearn.metrics import silhouette_score
from scipy.stats import pearsonr
import copy
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality
from sdv.single_table import CTGANSynthesizer
from sdv.sampling import Condition
from sdv.evaluation.single_table import get_column_plot

import dask.dataframe as dpd
import dask_geopandas as dgpd
from dask.diagnostics import ProgressBar
from dask.distributed import Client

import warnings
warnings.filterwarnings('ignore')

from utils.logger import Logger

import gc
gc.collect()

if torch.cuda.is_available():
    print("CUDA is available. Detected CUDA Devices:")
    for i in range(torch.cuda.device_count()):
        print(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available.")

coord_gdf = gpd.read_file('src/coord/coord_gdf.shp')
coord_gdf = coord_gdf.drop(columns=['cell_rmse1', 'cell_r21', 'cell_rmse2', 'cell_r22', 'depth'])
coord_union_gdf = gpd.GeoDataFrame(geometry=[coord_gdf.unary_union], crs=coord_gdf.crs)

np.random.seed(3)
sample_event_num = 50
event_indices = np.random.choice(range(593), sample_event_num, replace=False)

scaler = MinMaxScaler()
xy_scaled = scaler.fit_transform(coord_gdf[['x', 'y']])

def load_and_scale(file_path, scale=False):
    df = pd.read_parquet(file_path)[['x', 'y', 'channel', 'ter', 'cumu_rain', 'peak_int', 'duration', 'depth']]
    if scale:
        df[['x', 'y']] = xy_scaled
    return df
file_paths = [f'src/tables/data{i}.parquet' for i in event_indices]
selected_events = [load_and_scale(file, scale=True) for file in file_paths]
with ProgressBar():
    result = dpd.concat(selected_events, axis=0)
selected_events_df = result.compute()
selected_events_df = selected_events_df.drop(columns=['channel', 'ter', 'depth'])
selected_events_df['duration'] = selected_events_df['duration'].astype(float)

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(selected_events_df)

lr_sets = [
    # G, D
    [1e-5, 1e-5],
    [2e-5, 1e-5],
    [1e-4, 1e-4],
    [2e-4, 1e-4],
    [1e-3, 1e-3],
    [2e-3, 1e-3]
]

x_bounds_constraints = {
    'constraint_class': 'ScalarRange',
    'constraint_parameters': {
        'column_name': 'x',
        'low_value': 0.0,
        'high_value': 1.0,
        'strict_boundaries': False
    }
}

y_bounds_constraints = {
    'constraint_class': 'ScalarRange',
    'constraint_parameters': {
        'column_name': 'y',
        'low_value': 0.0,
        'high_value': 1.0,
        'strict_boundaries': False
    }
}

peak_int_constraints = {
    'constraint_class': 'PeakIntConstraintClass',
    'constraint_parameters': {
        'column_names': ['cumu_rain', 'peak_int', 'duration']
    }
}

positive_constraints = {
    'constraint_class': 'Positive',
    'constraint_parameters': {
        'column_name': 'peak_int',
        'strict_boundaries': False
    }
}

inequalty_constraints = {
    'constraint_class': 'Inequality',
    'constraint_parameters': {
        'low_column_name': 'peak_int',
        'high_column_name': 'cumu_rain'
    }
}


def filter_rows_by_condition(df):
    return df[(df['peak_int'] >= df['cumu_rain'] / df['duration']) & (df['cumu_rain'] >= df['peak_int'])]

ctgan_logger = Logger('logs/cCTGAN_model.log')

for lr_id, lrs in tqdm(enumerate(lr_sets), total=len(lr_sets)):
    for epoch in range(50, 301, 50):
        try:
            ctgan_logger.log_info(f'[{lr_id+1}/{len(lr_sets)}] [Epoch: {epoch}]: {lrs[0]}_{lrs[1]}_{epoch+1}')
            ctgan_synthesizer = CTGANSynthesizer(metadata, epochs=epoch, 
                                                cuda=True, verbose=True, enforce_rounding=False, 
                                                batch_size=5000, generator_lr=lrs[0], discriminator_lr=lrs[1])
            ctgan_synthesizer.load_custom_constraint_classes(
                filepath = 'models/cCTGAN.py',
                class_names = ['PeakIntConstraintClass']
            )
            ctgan_synthesizer.add_constraints(
                constraints = 
                [peak_int_constraints]
                + [x_bounds_constraints, y_bounds_constraints, inequalty_constraints, positive_constraints]
                
            )
            # train ctgan
            ctgan_logger.log_info('Start training...')
            ctgan_synthesizer.fit(selected_events_df)
            ctgan_synthesizer.save(f'checkpoints/cCTGAN/{lrs[0]}_{lrs[1]}_{epoch+1}.pkl')
            ctgan_logger.log_info('Generating events...')
            filtered_df_path = f'outputs/augmented_data/{lrs[0]}_{lrs[1]}_{epoch}.parquet'
            ctgan_synthetic_df = ctgan_synthesizer.sample(num_rows=5000000, batch_size=5000)
            ctgan_synthetic_df[['x', 'y']] = scaler.inverse_transform(ctgan_synthetic_df[['x', 'y']])
            ctgan_synthetic_gdf = gpd.GeoDataFrame(
                ctgan_synthetic_df, 
                geometry=gpd.points_from_xy(ctgan_synthetic_df.x, ctgan_synthetic_df.y),
                crs=coord_gdf.crs
            )
            ctgan_logger.log_info('Filtering...')
            ctgan_synthetic_inbound_df = ctgan_synthetic_gdf.sjoin(coord_union_gdf, predicate='within').drop(columns=['geometry', 'index_right'])
            ctgan_synthetic_inbound_df = filter_rows_by_condition(ctgan_synthetic_inbound_df).reset_index(drop=True)
            ctgan_synthetic_inbound_df.to_parquet(filtered_df_path)
        except Exception as e:
            ctgan_logger.log_error(e)