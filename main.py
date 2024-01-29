import os, sys
import geopandas as gpd
from argparse import ArgumentParser
import yaml
from sdv.metadata import SingleTableMetadata
import torch
from utils.events_aug import cCTGAN_modeling, events_sample_concat

import warnings
warnings.filterwarnings('ignore')


torch.cuda.set_device(0)


if __name__ == '__main__':
    config = yaml.safe_load(open('config.yaml'))

    parser = ArgumentParser()
    parser.add_argument('--coord-path', default='src/coord/coord_gdf.shp')
    parser.add_argument('--gan-train-event-num', default=config['gan_train_event_num'])
    parser.add_argument('--n-workers', default=config['n_workers'])
    parser.add_argument('--constraints', type=bool, default=False)
    opt = parser.parse_args()

    coord_gdf = gpd.read_file(opt.coord_path)
    coord_gdf = coord_gdf.drop(columns=['cell_rmse1', 'cell_r21', 'cell_rmse2', 'cell_r22', 'depth'])

    selected_events_df = events_sample_concat(coord_gdf, opt.gan_train_event_num, opt.n_workers)
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(selected_events_df)
    print('Start cCTGAN modeling...')
    cCTGAN_modeling(selected_events_df, metadata, config, constraint=opt.constraints, train=True)