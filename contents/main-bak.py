from train import train_model
from evaluate import evaluate_model
from predict import predict_age_gender

def main():
    train_csv_path = 'path_to_train_csv'
    val_csv_path = 'path_to_val_csv'
    image_path = 'path_to_new_image.jpg'
    model_path = 'age_gender_model.h5'

    # Train the model
    train_model(train_csv_path, val_csv_path)

    # Evaluate the model
    evaluate_model(model_path, val_csv_path)

    # Predict age and gender for a new image
    age, gender = predict_age_gender(model_path, image_path)
    print(f"Predicted Age: {age}")
    print(f"Predicted Gender: {gender}")

if __name__ == "__main__":
    main()
