import tensorflow as tf
from data_loader import load_data, create_data_generators

def evaluate_model(model_path, val_csv_path):
    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Load data
    _, val_df = load_data(None, val_csv_path)

    # Create validation data generator
    _, validation_generator = create_data_generators(None, val_df)

    # Evaluate the model
    loss, age_loss, gender_loss, age_mae, gender_accuracy = model.evaluate(validation_generator)

    print(f"Age MAE: {age_mae}")
    print(f"Gender Accuracy: {gender_accuracy}")

if __name__ == "__main__":
    model_path = 'age_gender_model.h5'
    val_csv_path = 'data/validation/val.csv'
    evaluate_model(model_path, val_csv_path)
