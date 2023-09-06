import os
import pickle

def save_model_with_pickle(model, filename):
    # Create 'models' directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Save the model
    with open(f'models/{filename}.pkl', 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved as {filename}.pkl in the 'models' directory.")

def load_model_with_pickle(filename):
    try:
        with open(f'models/{filename}.pkl', 'rb') as f:
            model = pickle.load(f)
        print(f"Model {filename}.pkl loaded from the 'models' directory.")
        return model
    except FileNotFoundError:
        print("File not found. Make sure the 'models' directory exists and contains the model file.")
        return None