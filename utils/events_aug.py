import numpy as np
import os, sys
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import dask.dataframe as dpd
from dask.diagnostics import ProgressBar
from sdv.single_table import CTGANSynthesizer
from dask.distributed import Client
from tqdm.auto import tqdm, trange
import yaml
from .logger import Logger

np.random.seed(0)

def load_and_scale(file_path, xy_scaled, scale=False):
    df = pd.read_parquet(file_path)[['x', 'y', 'channel', 'ter', 'cumu_rain', 'peak_int', 'duration', 'depth']]
    if scale:
        df[['x', 'y']] = xy_scaled
    return df

def events_sample_concat(coord_gdf, sample_event_num=50, n_workers=100):
    # Create a Dask client
    client = Client(n_workers=n_workers)

    try:
        event_indices = np.random.choice(range(593), sample_event_num, replace=False)
        scaler = MinMaxScaler()
        xy_scaled = scaler.fit_transform(coord_gdf[['x', 'y']])
        file_paths = [f'src/tables/data{i}.parquet' for i in event_indices]
        selected_events = [load_and_scale(file, xy_scaled, scale=True) for file in file_paths]
        
        with ProgressBar():
            result = dpd.concat(selected_events, axis=0)
        selected_events_df = result.compute()
        selected_events_df = selected_events_df.drop(columns=['channel', 'ter', 'depth'])

    finally:
        # Close the Dask client
        client.close()

    return selected_events_df

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

positive_constraints = [{
    'constraint_class': 'Positive',
    'constraint_parameters': {
        'column_name': col,
        'strict_boundaries': False
    }
} for col in ['cumu_rain', 'peak_int']]

def cCTGAN_modeling(real_df, metadata, config, constraint=False, train=False):
    logger = Logger('logs/cCTGAN_model.log')
    lr_sets = config['gan_lr_sets']
    for lrs in tqdm(lr_sets, total=len(lr_sets)):
        for epoch in range(50, config['gan_epochs']+1, 50):
            try:
                checkpoint_path = f'checkpoints/cCGTAN/{lrs[0]}_{lrs[1]}_{epoch+1}.pkl'
                ctgan_synthesizer = CTGANSynthesizer(metadata, epochs=epoch, 
                                                    cuda=True, verbose=True, enforce_rounding=False, 
                                                    batch_size=512, generator_lr=lrs[0], discriminator_lr=lrs[1])
                if constraint:
                    ctgan_synthesizer.load_custom_constraint_classes(
                        filepath = 'models/cCTGAN.py',
                        class_names = ['PeakIntConstraintClass']
                    )
                    ctgan_synthesizer.add_constraints(
                        constraints = [x_bounds_constraints, y_bounds_constraints] + positive_constraints
                        + [peak_int_constraints]
                    )
                if train:
                    # train ctgan
                    logger.log_info(f'Start traing...')
                    ctgan_synthesizer.fit(real_df)
                    ctgan_synthesizer.save(checkpoint_path)
            except Exception as e:
                logger.log_error(f"Error occurred during CTGAN modeling: {e}", exc_info=True)