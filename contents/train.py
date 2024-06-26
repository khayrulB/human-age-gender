
# train.py
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, Callback
from model import create_model, compile_model
import sys, time


class SaveBestModel(Callback):
    def __init__(self, filepath, monitor='val_loss', verbose=0):
        super(SaveBestModel, self).__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.verbose = verbose
        self.best = float('inf')  # Initialize best to a large number for comparison (for loss minimization)


    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get(self.monitor)
        if current_val_loss is None:
            print(f"Warning: {self.monitor} is not available.")
            return
        
        if current_val_loss < self.best:
            self.best = current_val_loss
            self.best_epoch = epoch + 1  # Epochs are zero-indexed, so add 1 for human readability
            if self.verbose > 0:
                print(f"\nEpoch {epoch + 1}: {self.monitor} improved to {current_val_loss:.4f}, saving model to {self.filepath}")
            self.model.save_weights(self.filepath, overwrite=True)
            self.best_epoch += 1

    def on_train_end(self, logs=None):
        print(f"\nBest epoch: {self.best_epoch}")

#================================================================================================
def compile_and_train_model(train_generator, validation_generator, img_size, epochs=10):
    
    print('\n I am inside train KKKK')

    # Check if GPU is available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Set memory growth for GPU
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU available, using CPU.")


    # Create model
    #img_size = 224  # Define img_size as needed
    model = create_model(img_size)
    # Compile model
    model = compile_model(model)

    
    # Define callbacks
    checkpoint_callback = SaveBestModel(
        'best_model.h5',
        monitor='val_loss',  # Monitor validation loss
        #save_best_only=True,  # Save only the best model (based on the monitored metric)
        #mode='min',  # Save the model with the minimum validation loss
        verbose=1  # Print messages about model saving
    )

    # Reduce learning rate when validation loss plateaus
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, mode = 'min', verbose = 1)
    # Early Stopping 
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode = 'min', restore_best_weights=True)


    # Train the model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[checkpoint_callback, reduce_lr, early_stopping]
    )
    
    
    return history


#===========================================================================================
'''
if __name__ == "__main__":

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
