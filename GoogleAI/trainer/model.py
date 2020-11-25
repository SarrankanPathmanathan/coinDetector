from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Dense, Input, Dropout
from keras.applications import MobileNetV2


def create_keras_model():
    input_tensor = Input(shape=(224, 224, 3))
    base_model = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_tensor=input_tensor,
        input_shape=(224, 224, 3),
        pooling='avg')

    for layer in base_model.layers:
        layer.trainable = True  # trainable has to be false in order to freeze the layers

    x = Dense(512, activation='relu')(base_model.output)
    x = Dropout(.8)(x)

    predictions = Dense(211, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model
