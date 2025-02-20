import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Print the current working directory
print(f"Current Working Directory: {os.getcwd()}")

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

def load_and_preprocess_data(filepath):
    """
    Load the credit card dataset and perform preprocessing:
    - Drop missing values (if any)
    - Standardize 'Amount' and 'Time' columns
    - Drop the original 'Amount' and 'Time' columns after scaling
    """
    # Load the dataset
    df = pd.read_csv(filepath)
    
    # Check for and drop any missing values (the dataset typically has none)
    df.dropna(inplace=True)
    
    # Initialize the StandardScaler
    scaler = StandardScaler()
    
    # Scale the 'Amount' column and add it as 'scaled_amount'
    df['scaled_amount'] = scaler.fit_transform(df[['Amount']])
    
    # Scale the 'Time' column and add it as 'scaled_time'
    df['scaled_time'] = scaler.fit_transform(df[['Time']])
    
    # Drop the original 'Amount' and 'Time' columns
    df = df.drop(['Amount', 'Time'], axis=1)
    
    return df

if __name__ == "__main__":
    # File path to the raw dataset
    input_filepath = os.path.join(script_dir, '../data/cc.csv')

    # Ensure the data directory exists
    data_dir = os.path.join(script_dir, '../data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Load and preprocess the data
    df = load_and_preprocess_data(input_filepath)

    # Split the dataset into training and testing sets
    train_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42, 
        stratify=df['Class']  # ensures the class distribution is similar in both splits
    )

    # Save the preprocessed data into CSV files
    train_df.to_csv(os.path.join(data_dir, 'train_preprocessed.csv'), index=False)
    test_df.to_csv(os.path.join(data_dir, 'test_preprocessed.csv'), index=False)

    print("Data preprocessing complete. Preprocessed training and testing files saved in the data folder.")
