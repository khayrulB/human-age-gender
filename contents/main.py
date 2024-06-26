import os 
import pandas as pd
from sklearn.model_selection import train_test_split
from glob import glob
import cv2
import argparse 
from evaluate import evaluate_model
from predict import predict_age_gender
from data_loader import load_data, create_data_generators 
from model import create_model
from train import compile_and_train_model
import time 


def create_trainValid_datasets_UTKFace():

    image_dir = 'data/UTKFace'
    out_dir_train = 'data/train/images'
    out_dir_val = 'data/validation/images'

    # [age]_[gender]_[race]_[date&time].jpg 
    imgpaths = sorted(glob(os.path.join(image_dir, '*.jpg'), recursive=True))
    #images = [cv2.imread(path2img) for path2img in imgpaths]
    print(f'The size of dataset:{len(imgpaths)}')

    db_info=[]
    for i, imgpath in enumerate(imgpaths):
        #print(imgpath)
        full_filename = imgpath.split('/')[-1]
        parts = full_filename.split('_')
        db_info.append({'filename':full_filename,'age':parts[0], 'gender':parts[1]})

    df = pd.DataFrame(db_info)
    print(df.shape)
    print(df.head())

    #---------------------------------------------------------------------
    # Split the DataFrame into two sets (e.g., 70% train, 30% test)
    #----------------------------------------------------------------------
    train_df, val_df = train_test_split(df, test_size=0.3, random_state=42)
    print("Train DataFrame:")
    print(train_df)
    print("\nValidation DataFrame:")
    print(val_df)
     # Save the DataFrames to .csv files
    train_df.to_csv('data/train/train_data.csv', index=False)
    val_df.to_csv('data/validation/val_data.csv', index=False)
    train_filelist = train_df['filename'].values.tolist()
    val_filelist = val_df['filename'].values.tolist()

    print(len(train_filelist))
    print(len(val_filelist))
    print(f'The size of dataset:{len(imgpaths)}')

    print(train_filelist[:10])
    print(val_filelist[:10])
    
    # Iterate over training and validation file lists
    for filelist, out_dir in [(train_filelist, out_dir_train), (val_filelist, out_dir_val)]:
    #for filelist, out_dir in [(val_filelist, out_dir_val)]:
        for filename in filelist:
            filepath = os.path.join(image_dir, filename)
            img = cv2.imread(filepath)
            if img is not None:
                dest_filepath = os.path.join(out_dir, filename)
                cv2.imwrite(dest_filepath, img)
            else:
                print(f"Failed to read image: {filepath}")
#=============================================================================
def create_trainValid_datasets_HERBIS():
    # Read data to DF
    # Assuming your .txt file is named 'data.txt' and contains tab-separated values
    file_path = 'data/3_herbis_202311/3_herbis_202311_train.txt'  
    # Read the .txt file into a DataFrame
    df = pd.read_csv(file_path, sep=' ', header=None)
    print(df.shape)
    print(df.iloc[5])
    # Display the DataFrame
    print(df.head())

    # Split the DataFrame into two sets (e.g., 80% train, 20% test)
    #----------------------------------------------------------------------
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.columns = ['filepath', 'age', 'gender']
    val_df.columns = ['filepath', 'age', 'gender']
    # Save the DataFrames to .csv files
    train_df.to_csv('data/train/train_data.csv', index=False)
    val_df.to_csv('data/validation/val_data.csv', index=False)
    # Display the DataFrames
    print("Train DataFrame:")
    print(train_df)
    print("\nTest DataFrame:")
    print(val_df)
    #Put data to train and validation folder
#==============================================================================
def normalize_age(df):
    df['age'] = df['age'] / 105.0  # Assuming max age is around 100
    return df

#===============================================================================
#===============================================================================
def main():
    parser = argparse.ArgumentParser(description='Age and Gender Classification')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--predict', type=str, help='Predict age and gender for an image')

    args = parser.parse_args()
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    print(f'I am developing age-gender model')

    #log_file = 'train_log.txt' # Save to current_dir

    train_csv_path = 'data/train/train_data.csv'
    val_csv_path = 'data/validation/val_data.csv'
    #model_path = 'age_gender_model.h5'

    # Load data
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)

    # Specify directories for image data
    train_dir = 'data/train/images/'
    val_dir = 'data/validation/images/'

    # Specify image size and batch size
    target_img_size = 224  # expected size for MobileNet2 model
    batch_train = 128
    batch_val = 64
    epochs = 50

    
    # Load data
    train_df, val_df = load_data(train_csv_path, val_csv_path)

    train_df = normalize_age(train_df)
    val_df = normalize_age(val_df)
    train_generator, validation_generator = create_data_generators(train_df, val_df, train_dir, val_dir, target_img_size, batch_train, batch_val)

    # Build model
    #model = create_model(target_img_size)

    # Train the model
    history = compile_and_train_model(train_generator, validation_generator, target_img_size, epochs)

    

    '''

    if args.train:
        train_model(train_csv_path, val_csv_path)

    if args.evaluate:
        evaluate_model(model_path, val_csv_path)

    if args.predict:
        image_path = args.predict
        age, gender = predict_age_gender(model_path, image_path)
        print(f"Predicted Age: {age}")
        print(f"Predicted Gender: {gender}")
    
    '''

if __name__ == "__main__":
    
    # Start time
    start_time = time.time()
    #-----------------------
    main()
    #-----------------------
    # End time
    end_time = time.time()

    # Compute elapsed time
    elapsed_time = end_time - start_time

    # Convert elapsed time to hours, minutes, seconds
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    # Print elapsed time
    print(f"\nElapsed time: {hours} hours, {minutes} minutes, {seconds} seconds")
