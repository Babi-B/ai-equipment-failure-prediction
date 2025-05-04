import os
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy import stats
from scipy.fft import fft
from io import StringIO


def load_nasa_data(file_paths, is_test=False):
    """Load multiple NASA files with metadata - handles both paths and file objects"""
    dfs = []

    for file_path in file_paths:
        if hasattr(file_path, 'read'):  # File-like object
            content = file_path.read().decode('utf-8')
            file_obj = StringIO(content)
            filename = file_path.name
        else:  # File path string
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Data file not found: {file_path}")
            file_obj = file_path
            filename = os.path.basename(file_path)

        cols = ["engine_id", "cycle"] + [f"op_{i}" for i in range(1, 4)] + [f"sensor_{i}" for i in range(1, 22)]

        try:
            df = pd.read_csv(
                file_obj,
                sep=r"\s+",
                header=None,
                names=cols,
                engine="python",
                dtype={
                    'engine_id': 'str',
                    'cycle': 'int32',
                    **{f'op_{i}': 'float32' for i in range(1, 4)},
                    **{f'sensor_{i}': 'float32' for i in range(1, 22)}
                }
            )
        except Exception as e:
            raise ValueError(f"Error reading file {filename}: {str(e)}")

        df = df.dropna(axis=1, how='all')
        df["engine_id"] = df["engine_id"].astype(str).str.replace('nan', '').str.strip()

        dataset_id = filename.split('_')[-1].split('.')[0]
        df["engine_id"] = dataset_id + "_" + df["engine_id"]
        df["dataset_id"] = dataset_id

        df["operating_condition"] = (
            df[["op_1", "op_2", "op_3"]]
            .apply(tuple, axis=1)
            .astype('category')
            .cat.codes
        )

        df["fault_mode"] = int(dataset_id[-1])

        if not is_test:
            df = add_rul(df)

        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    if combined["engine_id"].isna().any():
        raise ValueError("NaN values detected in engine_id after processing")

    numeric_cols = combined.select_dtypes(include=[np.number]).columns
    combined[numeric_cols] = combined[numeric_cols].apply(pd.to_numeric, errors='coerce')

    return combined


def handle_missing_data(df):
    """Handle missing data with type preservation"""
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if 'engine_id' not in non_numeric_cols and 'engine_id' in df.columns:
        non_numeric_cols.append('engine_id')

    numeric_df = df[numeric_cols].copy()
    numeric_df = numeric_df.astype('float32')
    numeric_df = numeric_df.interpolate(method='linear', limit_direction='forward')

    non_numeric_df = df[non_numeric_cols].copy()
    result = pd.concat([non_numeric_df.reset_index(drop=True),
                        numeric_df.reset_index(drop=True)], axis=1)

    return result[df.columns]


def add_rul(df):
    """Calculate Remaining Useful Life with exponential weighting"""
    df["max_cycle"] = df.groupby("engine_id")["cycle"].transform("max")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df["RUL"] = np.where(
        df["RUL"] > 30,
        df["RUL"],
        30 * np.exp(0.1 * (df["RUL"] - 30)))
    return df.drop(columns=["max_cycle"])


def create_sequences(data, window_size=30, is_test=False):
    """Create time-series sequences with proper window alignment"""
    sequences = []
    targets = [] if not is_test else None

    # Ensure we only use numeric columns
    feature_columns = data.select_dtypes(include=np.number).columns.tolist()
    feature_columns = [col for col in feature_columns if col not in ['engine_id', 'RUL', 'dataset_id']]

    valid_engines = data.groupby('engine_id').filter(lambda x: len(x) >= window_size)['engine_id'].unique()

    for engine_id in valid_engines:
        engine_data = data[data["engine_id"] == engine_id]
        num_windows = len(engine_data) - window_size

        for i in range(num_windows):
            seq = engine_data.iloc[i:i + window_size]
            sequences.append(seq[feature_columns].values.astype('float32'))

            if not is_test:
                targets.append(seq["RUL"].values[-1])

    return (np.array(sequences), np.array(targets)) if not is_test else np.array(sequences)


def create_tabular_features(data, window_size=30):
    """Create tabular features with robust error handling"""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame")

    required_columns = {'engine_id', 'sensor_7', 'sensor_12', 'op_3', 'fault_mode'}
    if not required_columns.issubset(data.columns):
        missing = required_columns - set(data.columns)
        raise ValueError(f"Missing required columns: {missing}")

    tab_features = []
    engine_groups = data.groupby("engine_id")
    valid_engines = [eng for eng, grp in engine_groups if len(grp) >= window_size]

    for engine_id in valid_engines:
        engine_data = engine_groups.get_group(engine_id)
        num_windows = len(engine_data) - window_size + 1

        for i in range(num_windows):
            window = engine_data.iloc[i:i + window_size]

            # Ensure all values are numeric
            features = {
                's7_mean': float(window['sensor_7'].mean()),
                's12_var': float(window['sensor_12'].var()),
                'op3_max': float(window['op_3'].max()),
                'fault_mode': int(window['fault_mode'].iloc[0]),
                'window_num': int(i)
            }
            tab_features.append(features)

    result_df = pd.DataFrame(tab_features)

    # Final type conversion
    for col in result_df.columns:
        result_df[col] = pd.to_numeric(result_df[col], errors='coerce')

    return result_df