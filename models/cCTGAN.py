from sdv.constraints import create_custom_constraint_class
import pandas as pd

def is_valid_ieq(column_names, data):
    # Assuming column_names[0] = 'cumu_rain', column_names[1] = 'peak_int', column_names[2] = 'duration'
    # Check if peak_int >= cumu_rain / duration for each row
    validity = data[column_names[1]] >= data[column_names[0]] / data[column_names[2]]
    return pd.Series(validity)

def transform_fn_ieq(column_names, data):
    # Assuming column_names[0] = 'cumu_rain', column_names[1] = 'peak_int', column_names[2] = 'duration'
    # Modify peak_int to ensure it's >= cumu_rain / duration
    data[column_names[1]] = data.apply(
        lambda row: max(row[column_names[1]], row[column_names[0]] / row[column_names[2]]),
        axis=1
    )
    return data

def reverse_transform_ieq(column_names, transformed_data):
    return transformed_data.copy()

PeakIntConstraintClass = create_custom_constraint_class(
    is_valid_fn = is_valid_ieq,
    transform_fn = transform_fn_ieq,
    reverse_transform_fn = reverse_transform_ieq
)

