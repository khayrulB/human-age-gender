import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator



def load_data(train_csv_path, val_csv_path):
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)

    return train_df, val_df

def create_data_generators(train_df, val_df, train_dir, val_dir, img_size, batch_train, batch_val):
#def create_data_generators(train_df, val_df, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    
    # If you have SINGLE DATASET
    # train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.3) # horizontal_flip=True
    
    # If you have SEPARATE TRAIN and VAL DATASETS, split para is NOT necessary
    # Initialize ImageDataGenerator for training data with rescaling
    #train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)

    # Initialize ImageDataGenerator for data augmentation and normalization
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.1,
        horizontal_flip=True
    )


    # Initialize ImageDataGenerator for validation data with rescaling
    val_datagen = ImageDataGenerator(rescale=1./255)

    # Specify the directories for training and validation data
    #train_dir = 'data/train/images'
    #val_dir = 'data/validation/images'


    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=train_dir,  # Adjusted path
        x_col='filename',
        y_col=['age', 'gender'],
        target_size=(img_size, img_size),
        batch_size=batch_train,
        class_mode='raw'  # Use 'raw' to handle multiple outputs manually (like two heads each has different classes)
        #subset='training' # Used when single dataset is used for train and validation
    )

    validation_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory= val_dir,  # Adjusted path
        x_col='filename',
        y_col=['age', 'gender'],
        target_size=(img_size, img_size),
        batch_size=batch_val,
        class_mode='raw'  # Use 'raw' to handle multiple outputs manually
        #subset='validation' #Used when single dataset is used for train and validation
    )

    return train_generator, validation_generator
