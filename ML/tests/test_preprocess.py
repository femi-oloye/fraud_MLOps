import os
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.preprocess import load_and_preprocess_data

def test_load_and_preprocess_data(tmp_path):
    # Create a dummy CSV file that mimics a simplified credit card dataset
    data = {
        "Time": [0, 1],
        "Amount": [100, 150],
        "Class": [0, 1]
    }
    df = pd.DataFrame(data)
    # Save the dummy data to a temporary file
    dummy_file = tmp_path / "dummy_data.csv"
    df.to_csv(dummy_file, index=False)
    
    # Load and preprocess the dummy data
    processed_df = load_and_preprocess_data(dummy_file)
    
    # Check that the processed dataframe is not empty and has the expected columns
    assert not processed_df.empty
    # Expect new columns 'scaled_amount' and 'scaled_time' to be present, and 'Amount', 'Time' dropped
    expected_columns = set(["Class", "scaled_amount", "scaled_time"]) 
    assert expected_columns.issubset(set(processed_df.columns))
