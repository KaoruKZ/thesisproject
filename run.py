import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from datetime import datetime
from train import train_model
from data_preprocess import prepare_data
from model import BReGNeXt

def get_run_directory(base_dir='./outputs'):
    """
    Generates a unique directory for each run, based on the current date and incrementing run number.
    """
    # Get today's date in YYYY-MM-DD format
    today_date = datetime.now().strftime('%Y-%m-%d')
    
    # Generate a base directory path for today's runs
    date_run_dir = os.path.join(base_dir, today_date)
    
    # Ensure the directory exists
    if not os.path.exists(date_run_dir):
        os.makedirs(date_run_dir)
    
    # Get the next available run number for today
    run_number = 1
    while os.path.exists(os.path.join(date_run_dir, f'run_{today_date}_{run_number}')):
        run_number += 1
    
    # Construct the final run directory path
    run_dir = os.path.join(date_run_dir, f'run_{today_date}_{run_number}')
    
    # Create the directory
    os.makedirs(run_dir)
    
    return run_dir


def run_training(image_folder, epochs=1, batch_size=8, lr=0.001, val_split=0.2):
    # Set the unique run directory for this training session
    run_name = get_run_directory()  # This will create a unique folder for this run
    print(f"Training run will be saved in: {run_name}")
    
    # Prepare the data using the preprocessing function
    train_loader, val_loader, emotion_map = prepare_data(image_folder, val_split=val_split, batch_size=batch_size)

    # Initialize the model
    model = BReGNeXt(n_classes=7)  # Ensure the model's output layer matches the number of classes

    # Start training the model
    model = train_model(train_loader, val_loader, model, epochs=epochs, lr=lr, run_name=run_name)


if __name__ == "__main__":
    # Define dataset folder (the path to the FER-2013 dataset)
    image_folder = './fer2013'  # Path to your dataset folder

    # Start the training process
    run_training(image_folder, epochs=2000, batch_size=32, lr=0.001, val_split=0.2)
