def initial_Cnn():
    # Definire un modello sequenziale di Keras
    model = Sequential()

    # Aggiungere un layer convoluzionale con 32 filtri, dimensione del kernel di 3x3, funzione di attivazione ReLU e padding "same"
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(image_size, image_size, 3)))

    # Aggiungere un layer di max pooling con dimensione 2x2
    model.add(MaxPooling2D((2, 2)))

    # Aggiungere un altro layer convoluzionale con 64 filtri, dimensione del kernel di 3x3, funzione di attivazione ReLU e padding "same"
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))

    # Aggiungere un altro layer di max pooling con dimensione 2x2
    model.add(MaxPooling2D((2, 2)))

    # Aggiungere un altro layer convoluzionale con 128 filtri, dimensione del kernel di 3x3, funzione di attivazione ReLU e padding "same"
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))

    # Aggiungere un altro layer di max pooling con dimensione 2x2
    model.add(MaxPooling2D((2, 2)))

    # Aggiungere un layer di appiattimento per convertire l'output del layer precedente in un vettore unidimensionale
    model.add(Flatten())

    # Aggiungere un layer completamente connesso con 512 unità e funzione di attivazione ReLU
    model.add(Dense(512, activation='relu'))

    # Aggiungere un layer di dropout con una percentuale di dropout di 0,5
    model.add(Dropout(0.5))

    # Aggiungere un altro layer completamente connesso con 4 unità (una per ogni classe) e funzione di attivazione softmax
    model.add(Dense(4, activation='softmax'))

    # Stampa il riepilogo del modello
    model.summary()

def Cnn1():
    """
        First CNN model with one convolutional layer and one max pooling layer at the biginning.
        Than one flatten layer and one dense layer with 4 units (one for each class) and softmax activation function.

        Epoch 1/10
        135/135 [==============================] - 51s 378ms/step - loss: 0.4653 - accuracy: 0.8400 - val_loss: 0.5552 - val_accuracy: 0.7895 - lr: 0.0010
        Epoch 2/10
        135/135 [==============================] - 35s 260ms/step - loss: 0.1964 - accuracy: 0.9467 - val_loss: 0.5493 - val_accuracy: 0.8182 - lr: 0.0010
        Epoch 3/10
        135/135 [==============================] - 37s 275ms/step - loss: 0.0749 - accuracy: 0.9865 - val_loss: 0.5567 - val_accuracy: 0.8280 - lr: 0.0010
        Epoch 4/10
        135/135 [==============================] - 35s 262ms/step - loss: 0.0317 - accuracy: 0.9972 - val_loss: 0.5820 - val_accuracy: 0.8280 - lr: 0.0010
        Epoch 5/10
        135/135 [==============================] - 37s 275ms/step - loss: 0.0163 - accuracy: 0.9986 - val_loss: 0.6349 - val_accuracy: 0.8168 - lr: 0.0010
        Epoch 6/10
        135/135 [==============================] - 35s 257ms/step - loss: 0.0137 - accuracy: 0.9993 - val_loss: 0.6135 - val_accuracy: 0.8294 - lr: 0.0010
        Epoch 7/10
        135/135 [==============================] - 37s 273ms/step - loss: 0.0068 - accuracy: 0.9995 - val_loss: 0.6632 - val_accuracy: 0.8287 - lr: 0.0010
        Epoch 8/10
        135/135 [==============================] - 35s 257ms/step - loss: 0.0036 - accuracy: 0.9998 - val_loss: 0.6580 - val_accuracy: 0.8280 - lr: 1.0000e-04
        Epoch 9/10
        135/135 [==============================] - 35s 262ms/step - loss: 0.0033 - accuracy: 1.0000 - val_loss: 0.6617 - val_accuracy: 0.8287 - lr: 1.0000e-04
        Epoch 10/10
        135/135 [==============================] - 36s 269ms/step - loss: 0.0031 - accuracy: 1.0000 - val_loss: 0.6648 - val_accuracy: 0.8280 - lr: 1.0000e-04
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(image_size, image_size, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(4, activation='softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5)
    checkpoint = ModelCheckpoint(filepath=dataset_path + '/modello_cnn_exp1.h5', monitor='val_loss', save_best_only=True)

    callbacks_list = [reduce_lr, checkpoint]
    history = model.fit(train_generator, epochs=10, validation_data=val_generator, callbacks=callbacks_list)

    return model

def Cnn2():
    """
        Second CNN model with three convolutional layers and three max pooling layers at the biginning.
        Than one flatten layer and one dense layer with 256 units and softmax activation function.
        We add a dropout layer with 0.5 probability to avoid overfitting.
        At the end we add a dense layer with 4 units (one for each class) and softmax activation function.

        Epoch 1/10
        135/135 [==============================] - 41s 288ms/step - loss: 1.1161 - accuracy: 0.5384 - val_loss: 0.9233 - val_accuracy: 0.6196 - lr: 0.0010
        Epoch 2/10
        135/135 [==============================] - 41s 302ms/step - loss: 0.7222 - accuracy: 0.7130 - val_loss: 0.6251 - val_accuracy: 0.7678 - lr: 0.0010
        Epoch 3/10
        135/135 [==============================] - 42s 307ms/step - loss: 0.5197 - accuracy: 0.8102 - val_loss: 0.5538 - val_accuracy: 0.7979 - lr: 0.0010
        Epoch 4/10
        135/135 [==============================] - 42s 312ms/step - loss: 0.4005 - accuracy: 0.8477 - val_loss: 0.4557 - val_accuracy: 0.8378 - lr: 0.0010
        Epoch 5/10
        135/135 [==============================] - 40s 295ms/step - loss: 0.2675 - accuracy: 0.9026 - val_loss: 0.4224 - val_accuracy: 0.8580 - lr: 0.0010
        Epoch 6/10
        135/135 [==============================] - 37s 274ms/step - loss: 0.1877 - accuracy: 0.9265 - val_loss: 0.4317 - val_accuracy: 0.8545 - lr: 0.0010
        Epoch 7/10
        135/135 [==============================] - 36s 267ms/step - loss: 0.1461 - accuracy: 0.9477 - val_loss: 0.4655 - val_accuracy: 0.8650 - lr: 0.0010
        Epoch 8/10
        135/135 [==============================] - 38s 281ms/step - loss: 0.1033 - accuracy: 0.9653 - val_loss: 0.4769 - val_accuracy: 0.8671 - lr: 0.0010
        Epoch 9/10
        135/135 [==============================] - 35s 262ms/step - loss: 0.0940 - accuracy: 0.9693 - val_loss: 0.4852 - val_accuracy: 0.8573 - lr: 0.0010
        Epoch 10/10
        135/135 [==============================] - 36s 264ms/step - loss: 0.0701 - accuracy: 0.9758 - val_loss: 0.5696 - val_accuracy: 0.8755 - lr: 0.0010
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(image_size, image_size, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5)
    checkpoint = ModelCheckpoint(filepath=dataset_path + '/modello_cnn_exp2.h5', monitor='val_loss', save_best_only=True)

    callbacks_list = [reduce_lr, checkpoint]
    history = model.fit(train_generator, epochs=10, validation_data=val_generator, callbacks=callbacks_list)

    return model

def Cnn3():
    """
        Third CNN model with four convolutional layers and four max pooling layers at the biginning.
        Each of the max pooling layer have a different pool size.
        Than one flatten layer and one dense layer with 256 units and softmax activation function.
        We add a dropout layer with 0.5 probability to avoid overfitting.
        At the end we add a dense layer with 4 units (one for each class) and softmax activation function.    

        Epoch 1/10
        135/135 [==============================] - 39s 269ms/step - loss: 1.2229 - accuracy: 0.4314 - val_loss: 1.0087 - val_accuracy: 0.5881 - lr: 0.0010
        Epoch 2/10
        135/135 [==============================] - 36s 268ms/step - loss: 0.8675 - accuracy: 0.6484 - val_loss: 0.7306 - val_accuracy: 0.7070 - lr: 0.0010
        Epoch 3/10
        135/135 [==============================] - 38s 283ms/step - loss: 0.7076 - accuracy: 0.7123 - val_loss: 0.7082 - val_accuracy: 0.7210 - lr: 0.0010
        Epoch 4/10
        135/135 [==============================] - 35s 262ms/step - loss: 0.6195 - accuracy: 0.7533 - val_loss: 0.6254 - val_accuracy: 0.7427 - lr: 0.0010
        Epoch 5/10
        135/135 [==============================] - 36s 267ms/step - loss: 0.5406 - accuracy: 0.7974 - val_loss: 0.5477 - val_accuracy: 0.7923 - lr: 0.0010
        Epoch 6/10
        135/135 [==============================] - 39s 286ms/step - loss: 0.4631 - accuracy: 0.8272 - val_loss: 0.5177 - val_accuracy: 0.7972 - lr: 0.0010
        Epoch 7/10
        135/135 [==============================] - 35s 257ms/step - loss: 0.3972 - accuracy: 0.8479 - val_loss: 0.5326 - val_accuracy: 0.7958 - lr: 0.0010
        Epoch 8/10
        135/135 [==============================] - 36s 269ms/step - loss: 0.3445 - accuracy: 0.8672 - val_loss: 0.4307 - val_accuracy: 0.8322 - lr: 0.0010
        Epoch 9/10
        135/135 [==============================] - 37s 276ms/step - loss: 0.2780 - accuracy: 0.8956 - val_loss: 0.3484 - val_accuracy: 0.8706 - lr: 0.0010
        Epoch 10/10
        135/135 [==============================] - 37s 272ms/step - loss: 0.2110 - accuracy: 0.9233 - val_loss: 0.3744 - val_accuracy: 0.8811 - lr: 0.0010
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(image_size, image_size, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((3, 3)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((4, 4)))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((5, 5)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5)
    checkpoint = ModelCheckpoint(filepath=dataset_path + '/modello_cnn_exp3.h5', monitor='val_loss', save_best_only=True)

    callbacks_list = [reduce_lr, checkpoint]
    history = model.fit(train_generator, epochs=10, validation_data=val_generator, callbacks=callbacks_list)

    return model

def Cnn4():
    """
        Fourth CNN model with four convolutional layers and four max pooling layers at the biginning.
        Now each of the max pooling layer have the same pool size.
        Than one flatten layer and one dense layer with 256 units and softmax activation function.
        We add a dropout layer with 0.5 probability to avoid overfitting.
        At the end we add a dense layer with 4 units (one for each class) and softmax activation function.    

        Epoch 1/10
        135/135 [==============================] - 42s 292ms/step - loss: 1.1654 - accuracy: 0.4826 - val_loss: 0.8825 - val_accuracy: 0.6531 - lr: 0.0010
        Epoch 2/10
        135/135 [==============================] - 38s 279ms/step - loss: 0.7967 - accuracy: 0.6784 - val_loss: 0.6830 - val_accuracy: 0.7217 - lr: 0.0010
        Epoch 3/10
        135/135 [==============================] - 40s 293ms/step - loss: 0.6138 - accuracy: 0.7649 - val_loss: 0.6170 - val_accuracy: 0.7531 - lr: 0.0010
        Epoch 4/10
        135/135 [==============================] - 38s 282ms/step - loss: 0.5000 - accuracy: 0.8119 - val_loss: 0.4859 - val_accuracy: 0.8119 - lr: 0.0010
        Epoch 5/10
        135/135 [==============================] - 41s 303ms/step - loss: 0.3887 - accuracy: 0.8556 - val_loss: 0.4542 - val_accuracy: 0.8350 - lr: 0.0010
        Epoch 6/10
        135/135 [==============================] - 39s 289ms/step - loss: 0.3126 - accuracy: 0.8842 - val_loss: 0.4323 - val_accuracy: 0.8406 - lr: 0.0010
        Epoch 7/10
        135/135 [==============================] - 39s 289ms/step - loss: 0.2445 - accuracy: 0.9177 - val_loss: 0.4425 - val_accuracy: 0.8580 - lr: 0.0010
        Epoch 8/10
        135/135 [==============================] - 38s 282ms/step - loss: 0.1982 - accuracy: 0.9260 - val_loss: 0.4081 - val_accuracy: 0.8783 - lr: 0.0010
        Epoch 9/10
        135/135 [==============================] - 38s 278ms/step - loss: 0.1590 - accuracy: 0.9414 - val_loss: 0.4584 - val_accuracy: 0.8713 - lr: 0.0010
        Epoch 10/10
        135/135 [==============================] - 37s 272ms/step - loss: 0.1188 - accuracy: 0.9581 - val_loss: 0.4282 - val_accuracy: 0.8685 - lr: 0.0010
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(image_size, image_size, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5)
    checkpoint = ModelCheckpoint(filepath=dataset_path + '/modello_cnn_exp4.h5', monitor='val_loss', save_best_only=True)

    callbacks_list = [reduce_lr, checkpoint]
    history = model.fit(train_generator, epochs=10, validation_data=val_generator, callbacks=callbacks_list)

    return model

def Cnn5():
    """
        Fifth CNN model with four convolutional layers and four max pooling layers at the biginning.
        Now each of the max pooling layer have a different pool size.
        Than one flatten layer and one dense layer with 256 units and softmax activation function.
        We add a dropout layer with 0.5 probability to avoid overfitting.
        Than one other dense layer with 256 units and softmax activation function.
        We add a dropout layer with 0.5 probability to avoid overfitting.
        At the end we add a dense layer with 4 units (one for each class) and softmax activation function.

        Epoch 1/10
        135/135 [==============================] - 39s 263ms/step - loss: 1.3784 - accuracy: 0.2788 - val_loss: 1.3458 - val_accuracy: 0.4189 - lr: 0.0010
        Epoch 2/10
        135/135 [==============================] - 36s 268ms/step - loss: 1.3468 - accuracy: 0.3563 - val_loss: 1.3199 - val_accuracy: 0.4070 - lr: 0.0010
        Epoch 3/10
        135/135 [==============================] - 36s 267ms/step - loss: 1.3299 - accuracy: 0.3709 - val_loss: 1.3009 - val_accuracy: 0.4070 - lr: 0.0010
        Epoch 4/10
        135/135 [==============================] - 38s 282ms/step - loss: 1.3222 - accuracy: 0.3609 - val_loss: 1.2886 - val_accuracy: 0.4217 - lr: 0.0010
        Epoch 5/10
        135/135 [==============================] - 35s 263ms/step - loss: 1.3096 - accuracy: 0.3588 - val_loss: 1.2778 - val_accuracy: 0.4238 - lr: 0.0010
        Epoch 6/10
        135/135 [==============================] - 37s 271ms/step - loss: 1.2941 - accuracy: 0.3914 - val_loss: 1.2501 - val_accuracy: 0.4469 - lr: 0.0010
        Epoch 7/10
        135/135 [==============================] - 39s 285ms/step - loss: 1.2732 - accuracy: 0.4070 - val_loss: 1.2262 - val_accuracy: 0.4350 - lr: 0.0010
        Epoch 8/10
        135/135 [==============================] - 37s 275ms/step - loss: 1.2519 - accuracy: 0.4221 - val_loss: 1.1986 - val_accuracy: 0.5385 - lr: 0.0010
        Epoch 9/10
        135/135 [==============================] - 35s 258ms/step - loss: 1.2112 - accuracy: 0.4716 - val_loss: 1.1392 - val_accuracy: 0.5636 - lr: 0.0010
        Epoch 10/10
        135/135 [==============================] - 36s 265ms/step - loss: 1.1514 - accuracy: 0.5188 - val_loss: 1.0671 - val_accuracy: 0.6175 - lr: 0.0010
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(image_size, image_size, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((3, 3)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((4, 4)))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((5, 5)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='softmax'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5)
    checkpoint = ModelCheckpoint(filepath=dataset_path + '/modello_cnn_exp5.h5', monitor='val_loss', save_best_only=True)

    callbacks_list = [reduce_lr, checkpoint]
    history = model.fit(train_generator, epochs=10, validation_data=val_generator, callbacks=callbacks_list)

    return model

def Cnn6():
    """
        Sixth CNN model with four convolutional layers and four max pooling layers at the biginning.
        Now each of the max pooling layer have the same pool size.
        Than one flatten layer and one dense layer with 128 units and softmax activation function.
        We add a dropout layer with 0.5 probability to avoid overfitting.
        Than one other dense layer with 256 units and softmax activation function.  
        We add a dropout layer with 0.5 probability to avoid overfitting.
        At the end we add a dense layer with 4 units (one for each class) and softmax activation function. 

        MERDOSO
    """
    model.Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(image_size, image_size, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='softmax'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5)
    checkpoint = ModelCheckpoint(filepath=dataset_path + '/modello_cnn_exp6.h5', monitor='val_loss', save_best_only=True)

    callbacks_list = [reduce_lr, checkpoint]
    history = model.fit(train_generator, epochs=10, validation_data=val_generator, callbacks=callbacks_list)

    return model

def Cnn7():
    """
        Seventh CNN model with four convolutional layers and four max pooling layers at the biginning.
        Now each of the max pooling layer have a different pool size.
        Than one flatten layer and one dense layer with 256 units and softmax activation function.
        We add a dropout layer with 0.5 probability to avoid overfitting.
        Than one other dense layer with 256 units and softmax activation function.
        We add a dropout layer with 0.5 probability to avoid overfitting.
        At the end we add a dense layer with 4 units (one for each class) and softmax activation function.

        Epoch 1/10
        135/135 [==============================] - 1216s 9s/step - loss: 1.3909 - accuracy: 0.2586 - val_loss: 1.3844 - val_accuracy: 0.2524 - lr: 0.0010
        Epoch 2/10
        135/135 [==============================] - 717s 5s/step - loss: 1.3664 - accuracy: 0.2947 - val_loss: 1.3467 - val_accuracy: 0.3881 - lr: 0.0010
        Epoch 3/10
        135/135 [==============================] - 702s 5s/step - loss: 1.3584 - accuracy: 0.3123 - val_loss: 1.3715 - val_accuracy: 0.2993 - lr: 0.0010
        Epoch 4/10
        135/135 [==============================] - 698s 5s/step - loss: 1.3425 - accuracy: 0.3365 - val_loss: 1.3175 - val_accuracy: 0.4063 - lr: 0.0010
        Epoch 5/10
        135/135 [==============================] - 695s 5s/step - loss: 1.3288 - accuracy: 0.3523 - val_loss: 1.3081 - val_accuracy: 0.4091 - lr: 0.0010
        Epoch 6/10
        135/135 [==============================] - 682s 5s/step - loss: 1.3256 - accuracy: 0.3402 - val_loss: 1.3112 - val_accuracy: 0.4140 - lr: 0.0010
        Epoch 7/10
        135/135 [==============================] - 700s 5s/step - loss: 1.3070 - accuracy: 0.3695 - val_loss: 1.2892 - val_accuracy: 0.4140 - lr: 0.0010
        Epoch 8/10
        135/135 [==============================] - 694s 5s/step - loss: 1.2948 - accuracy: 0.3870 - val_loss: 1.2498 - val_accuracy: 0.4692 - lr: 0.0010
        Epoch 9/10
        9/135 [=>............................] - ETA: 12:30 - loss: 1.2948 - accuracy: 0.3264
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(image_size, image_size, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((3, 3)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((4, 4)))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((5, 5)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='softmax'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5)
    checkpoint = ModelCheckpoint(filepath=dataset_path + '/modello_cnn_exp7.h5', monitor='val_loss', save_best_only=True)

    callbacks_list = [reduce_lr, checkpoint]
    history = model.fit(train_generator, epochs=10, validation_data=val_generator, callbacks=callbacks_list)

    return model

def Cnn8():
    """
        Eighth CNN model with four convolutional layers and four max pooling layers at the biginning.
        Now each of the max pooling layer have a different pool size.
        Than one flatten layer and one dense layer with 512 units and softmax activation function.
        We add a dropout layer with 0.5 probability to avoid overfitting.
        At the end we add a dense layer with 4 units (one for each class) and softmax activation function.

        Epoch 1/20
        135/135 [==============================] - 36s 266ms/step - loss: 0.1496 - accuracy: 0.9447 - val_loss: 0.4183 - val_accuracy: 0.8713 - lr: 0.0010
        Epoch 2/20
        135/135 [==============================] - 39s 286ms/step - loss: 0.1417 - accuracy: 0.9491 - val_loss: 0.3930 - val_accuracy: 0.8804 - lr: 0.0010
        Epoch 3/20
        135/135 [==============================] - 35s 260ms/step - loss: 0.0988 - accuracy: 0.9653 - val_loss: 0.4798 - val_accuracy: 0.8517 - lr: 0.0010
        Epoch 4/20
        135/135 [==============================] - 35s 262ms/step - loss: 0.0888 - accuracy: 0.9693 - val_loss: 0.4874 - val_accuracy: 0.8804 - lr: 0.0010
        Epoch 5/20
        135/135 [==============================] - 37s 273ms/step - loss: 0.0772 - accuracy: 0.9742 - val_loss: 0.6510 - val_accuracy: 0.8601 - lr: 0.0010
        Epoch 6/20
        135/135 [==============================] - 38s 281ms/step - loss: 0.0826 - accuracy: 0.9698 - val_loss: 0.4128 - val_accuracy: 0.8979 - lr: 0.0010
        Epoch 7/20
        135/135 [==============================] - 35s 259ms/step - loss: 0.0711 - accuracy: 0.9735 - val_loss: 0.3950 - val_accuracy: 0.8937 - lr: 0.0010
        Epoch 8/20
        135/135 [==============================] - 37s 274ms/step - loss: 0.0250 - accuracy: 0.9937 - val_loss: 0.3758 - val_accuracy: 0.9000 - lr: 1.0000e-04
        Epoch 9/20
        135/135 [==============================] - 36s 266ms/step - loss: 0.0128 - accuracy: 0.9970 - val_loss: 0.3860 - val_accuracy: 0.9028 - lr: 1.0000e-04
        Epoch 10/20
        135/135 [==============================] - 38s 283ms/step - loss: 0.0097 - accuracy: 0.9984 - val_loss: 0.3896 - val_accuracy: 0.9063 - lr: 1.0000e-04
        Epoch 11/20
        135/135 [==============================] - 35s 259ms/step - loss: 0.0080 - accuracy: 0.9991 - val_loss: 0.3986 - val_accuracy: 0.9070 - lr: 1.0000e-04
        Epoch 12/20
        135/135 [==============================] - 36s 269ms/step - loss: 0.0061 - accuracy: 0.9988 - val_loss: 0.4169 - val_accuracy: 0.9091 - lr: 1.0000e-04
        Epoch 13/20
        135/135 [==============================] - 35s 261ms/step - loss: 0.0053 - accuracy: 0.9995 - val_loss: 0.4176 - val_accuracy: 0.9091 - lr: 1.0000e-04
        Epoch 14/20
        135/135 [==============================] - 38s 281ms/step - loss: 0.0045 - accuracy: 0.9995 - val_loss: 0.4209 - val_accuracy: 0.9098 - lr: 1.0000e-05
        Epoch 15/20
        135/135 [==============================] - 35s 256ms/step - loss: 0.0039 - accuracy: 0.9995 - val_loss: 0.4217 - val_accuracy: 0.9105 - lr: 1.0000e-05
        Epoch 16/20
        135/135 [==============================] - 38s 280ms/step - loss: 0.0041 - accuracy: 0.9995 - val_loss: 0.4243 - val_accuracy: 0.9098 - lr: 1.0000e-05
        Epoch 17/20
        135/135 [==============================] - 36s 264ms/step - loss: 0.0045 - accuracy: 0.9991 - val_loss: 0.4272 - val_accuracy: 0.9126 - lr: 1.0000e-05
        Epoch 18/20
        135/135 [==============================] - 38s 278ms/step - loss: 0.0040 - accuracy: 0.9998 - val_loss: 0.4294 - val_accuracy: 0.9112 - lr: 1.0000e-05
        Epoch 19/20
        135/135 [==============================] - 37s 276ms/step - loss: 0.0040 - accuracy: 0.9995 - val_loss: 0.4295 - val_accuracy: 0.9112 - lr: 1.0000e-06
        Epoch 20/20
        135/135 [==============================] - 36s 267ms/step - loss: 0.0038 - accuracy: 0.9993 - val_loss: 0.4298 - val_accuracy: 0.9112 - lr: 1.0000e-06
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(image_size, image_size, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((3, 3)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((4, 4)))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((5, 5)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5)
    checkpoint = ModelCheckpoint(filepath=dataset_path + '/modello_cnn_exp8.h5', monitor='val_loss', save_best_only=True)

    callbacks_list = [reduce_lr, checkpoint]
    history = model.fit(train_generator, epochs=10, validation_data=val_generator, callbacks=callbacks_list)

    return model