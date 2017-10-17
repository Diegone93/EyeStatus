from keras.models import *
from keras.layers import *
from keras.optimizers import Adam

def EyesStatusNet(input_shape=None, weights_path=None):

    if input_shape == None:
        input_shape = (1, None, None)

    input_image = Input(shape=input_shape, name='input')

    x = Conv2D(64, (3, 3), activation='relu', name='conv1_1', padding='same')(input_image)
    x = Conv2D(64, (3, 3), activation='relu', name='conv1_2', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1_stage1')(x)
    x = Conv2D(32, (3, 3), activation='relu', name='conv2_1', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', name='conv2_2', padding='same')(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(2, activation='softmax')(x)
    model = Model(inputs=input_image, outputs=[output])

    if weights_path != None:
        print('loading weights')
        model.load_weights(filepath=weights_path)
        print('done')
    opt = Adam(lr=5e-5)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
    model.summary()
    return model


