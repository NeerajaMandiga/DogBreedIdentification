import os
import json
import argparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


def build_and_train(dataset_dir='dataset', img_size=(128, 128), batch_size=32, epochs=6, output_model='dogbreed.h5'):
    train_dir = os.path.join(dataset_dir, 'train')
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_generator = datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    num_classes = train_generator.num_classes

    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(*img_size, 3))
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        ModelCheckpoint(output_model, monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    ]

    model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks
    )

    # Save class names (index -> class)
    class_indices = train_generator.class_indices
    inv_map = {v: k for k, v in class_indices.items()}
    with open('class_names.json', 'w') as f:
        json.dump(inv_map, f)

    print(f"Training finished. Best model saved to {output_model} and classes to class_names.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VGG19-based dog breed classifier')
    parser.add_argument('--dataset', type=str, default='dataset', help='Dataset root folder')
    parser.add_argument('--epochs', type=int, default=6, help='Number of epochs (6-10 recommended)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--output', type=str, default='dogbreed.h5', help='Output model filename')
    args = parser.parse_args()

    build_and_train(dataset_dir=args.dataset, img_size=(128, 128), batch_size=args.batch_size, epochs=args.epochs, output_model=args.output)
