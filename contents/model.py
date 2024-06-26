
# model.py

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Flatten, Dropout, Input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras import regularizers

def create_model(img_size):
    # Define the input layer
    input_layer = Input(shape=(img_size, img_size, 3))

    # Load MobileNetV2 as the base model, excluding the top layer
    base_model = MobileNetV2(input_tensor=input_layer, include_top=False, weights='imagenet')

    # Add global average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    #x = GlobalMaxPooling2D()(x)
    
    # Additional dense layers
    
    # Additional dense layers with L2 regularization
    #x = Flatten()(x)
    x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = Dropout(0.1)(x)
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x) 
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x) 
    x = Dropout(0.3)(x)

    '''
    # Additional dense layers with Elastic Net regularization
    # Elastic Net regularization with L1 and L2 regularization terms
    l1 = 0.01
    l2 = 0.01
    x = Flatten()(x)
    x = Dense(512, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2))(x)
    x = Dropout(0.1)(x)
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2))(x) 
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2))(x) 
    x = Dropout(0.3)(x)

    
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    '''

    # Age prediction head
    #age_output = Dense(1, name='age_output')(x)
    age_output = Dense(1, activation='linear', name='age_output')(x)

    # Gender prediction head
    gender_output = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01), name='gender_output')(x)
    # gender_output = Dense(1, activation='sigmoid', name='gender_output')(x)

    # Create the model
    model = Model(inputs=input_layer, outputs=[age_output, gender_output])

    return model

def compile_model(model):
    # Compile the model with Adam optimizer and default learning rate
    #optimizer = Adam(learning_rate=0.001)
    optimizer = Adam()
    model.compile(optimizer=optimizer,
                  loss={'age_output': 'mean_squared_error', 'gender_output': 'binary_crossentropy'},
                  metrics={'age_output': 'mae', 'gender_output': 'accuracy'})

    return model


'''

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam

def create_model(img_size):
    # Define the input layer
    input_layer = Input(shape=(img_size, img_size, 3))

    # Load MobileNetV2 as the base model, excluding the top layer
    base_model = MobileNetV2(input_tensor=input_layer, include_top=False, weights='imagenet')

    # Add global average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # Age prediction head
    age_output = Dense(1, activation='linear', name='age_output')(x)

    # Gender prediction head
    gender_output = Dense(1, activation='sigmoid', name='gender_output')(x)

    # Create the model (VVI)
    model = Model(inputs=input_layer, outputs=[age_output, gender_output])

    # Compile the model (VVI)
    model.compile(optimizer=Adam(),
                  loss={'age_output': 'mean_squared_error', 'gender_output': 'binary_crossentropy'},
                  metrics={'age_output': 'mae', 'gender_output': 'accuracy'})

    return model


#definition Model 
def build_model(base_modelx):
    model = Sequential(base_modelx) 
    model.add(GlobalMaxPooling2D())
    model.add(Flatten())
    
    model.add(Dense(512,activation ='relu',kernel_regularizer=regularizers.l2(0.01))) 
    model.add(Dropout(0.1))
    
    model.add(Dense(256,activation ='relu',kernel_regularizer=regularizers.l2(0.01))) 
    model.add(Dropout(0.2))
    
    model.add(Dense(128,activation ='relu',kernel_regularizer=regularizers.l2(0.01))) 
    model.add(Dropout(0.3))
    
    model.add(Dense(1))
    print(model.summary())
    model.compile(optimizer='adam', loss='mse') 
    return model

'''
