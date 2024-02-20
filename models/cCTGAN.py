from sdv.constraints import create_custom_constraint_class
import numpy as np
import pandas as pd

def is_valid_ieq(column_names, data):
    try:
        # Assuming column_names[0] = 'cumu_rain', column_names[1] = 'peak_int', column_names[2] = 'duration'
        # Check if peak_int >= cumu_rain / duration for each row
        validity = data[column_names[1]] >= (data[column_names[0]] / (data[column_names[2]] + 1e-6))
        return pd.Series(validity)
    except Exception as e:
        print(f"Error in is_valid_ieq: {e}")
        raise

def transform_fn_ieq(column_names, data):
    try:
        # # Modify peak_int to ensure it's >= cumu_rain / duration
        transformed_data = data.copy()
        transformed_data['ieq'] = data[column_names[0]]
        transformed_data['ieq'] = data[column_names[0]] / (data[column_names[-1]] + 1e-6)
        transformed_data[column_names[1]] = transformed_data[[column_names[1], 'ieq']].max(axis=1)
        data[column_names[1]] = transformed_data[column_names[1]]
        return data
    except Exception as e:
        print(f"Error in transform_fn_ieq: {e}")
        raise


def reverse_transform_ieq(column_names, transformed_data):
    return transformed_data.copy()

PeakIntConstraintClass = create_custom_constraint_class(
    is_valid_fn = is_valid_ieq,
    transform_fn = transform_fn_ieq,
    reverse_transform_fn = reverse_transform_ieq
)

