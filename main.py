import os, sys
import geopandas as gpd
from argparse import ArgumentParser
import yaml
from sdv.metadata import SingleTableMetadata
import torch
from dask.distributed import Client
from utils.events_aug import cCTGAN_modeling, events_sample_concat

import warnings
warnings.filterwarnings('ignore')

if torch.cuda.is_available():
    print("CUDA is available. Detected CUDA Devices:")
    for i in range(torch.cuda.device_count()):
        print(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")
    torch.cuda.set_device(0)
else:
    print("CUDA is not available.")


if __name__ == '__main__':
    client = Client(n_workers=100)
    config = yaml.safe_load(open('config.yaml'))

    parser = ArgumentParser()
    parser.add_argument('--coord-path', default='src/coord/coord_gdf.shp')
    parser.add_argument('--gan-train-event-num', default=config['gan_train_event_num'])
    opt = parser.parse_args()

    coord_gdf = gpd.read_file(opt.coord_path)
    coord_gdf = coord_gdf.drop(columns=['cell_rmse1', 'cell_r21', 'cell_rmse2', 'cell_r22', 'depth'])

    selected_events_df = events_sample_concat(coord_gdf, opt.gan_train_event_num)
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(selected_events_df)
    cCTGAN_modeling(selected_events_df, metadata, config)