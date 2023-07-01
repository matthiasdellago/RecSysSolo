import os
import pandas as pd
import numpy as np

def load_data(data_path, verbose=False):
    '''
    This function loads all the tab-separated data from csvs into one pandas dataframe.
    
    Parameters:
    - data_path: string that represents the directory where the csv files are located
    - verbose: Boolean indicating whether to print information about the data loading process
    
    Returns:
    - data: a pandas DataFrame that contains all the data from the csv files
    '''
    assert os.path.exists(data_path), f"Data path not found at {data_path}"
    
    cvs_files = os.listdir(data_path)
    dataframes = []
    
    for file in cvs_files:
        csv_file = os.path.join(data_path, file)
        if verbose:
            print(f"Loading data from file: {file}")
        df = pd.read_csv(csv_file, delimiter='\t')
        dataframes.append(df)
    
    data = pd.concat(dataframes, ignore_index=True)
    
    if verbose:
        print(f"Successfully loaded data from {len(cvs_files)} files.")
    
    return data

def downcast(df, verbose=False):
    '''
    This function downcasts the data types of the columns in a DataFrame to reduce memory usage.
    
    Parameters:
    - df: a pandas DataFrame to be downcast
    - verbose: Boolean indicating whether to print information about the downcasting process
    
    Returns:
    - The function doesn't return anything but modifies the input DataFrame in place.
    '''
    
    cols = df.dtypes.index.tolist()
    types = df.dtypes.values.tolist()

    # Mapping of types and their corresponding downcasted types
    map_int = {1: np.int8, 2: np.int16, 4: np.int32, 8: np.int64}
    map_float = {2: np.float16, 4: np.float32, 8: np.float64}
    
    initial_memory = df.memory_usage().sum() / 1024**2  # Memory usage in MB

    for i,t in enumerate(types):
        if pd.api.types.is_numeric_dtype(df[cols[i]]):
            try:
                c_min = df[cols[i]].min()
                c_max = df[cols[i]].max()
                if pd.api.types.is_integer_dtype(df[cols[i]]) and not pd.api.types.is_float_dtype(df[cols[i]]):
                    df[cols[i]] = df[cols[i]].astype(get_downcast_type(map_int, c_min, c_max))
                elif pd.api.types.is_float_dtype(df[cols[i]]):
                    df[cols[i]] = df[cols[i]].astype(get_downcast_type(map_float, c_min, c_max))
                if verbose:
                    print(f"Downcasting {cols[i]} to {df[cols[i]].dtype}")
            except Exception as e:
                if verbose:
                    print(f"An error occurred when attempting to downcast {cols[i]}: {e}")
                    print(f"Data type before downcasting: {t}")
                    print(f"Min and Max values in the column: {c_min}, {c_max}")
        else:
            if verbose:
                print(f"Skipping non-numeric column {cols[i]} of type {df[cols[i]].dtype}")
    
    final_memory = df.memory_usage().sum() / 1024**2  # Memory usage in MB

    if verbose:
        print(f"Initial memory usage: {initial_memory} MB")
        print(f"Final memory usage: {final_memory} MB")
        print(f"Reduced memory usage by {100 * (initial_memory - final_memory) / initial_memory} % by downcasting.")


def get_downcast_type(type_map, c_min, c_max):
    '''
    This function identifies the smallest possible data type that can be used to represent
    the values in a range.
    
    Parameters:
    - type_map: a dictionary mapping of sizes to their corresponding data types
    - c_min: the minimum value in the range
    - c_max: the maximum value in the range
    
    Returns:
    - The smallest data type that can represent all values in the range
    
    Raises:
    - ValueError: If no suitable data type is found.
    '''
    for size, dtype in sorted(type_map.items()):
        dtype_info = np.iinfo(dtype) if np.issubdtype(dtype, np.integer) else np.finfo(dtype)
        if dtype_info.min <= c_min and c_max <= dtype_info.max:
            return dtype

    # If the function hasn't returned by this point, no suitable dtype was found.
    raise ValueError(f"No suitable dtype found for range: {c_min} to {c_max}")
