from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data augmentation for training set
train_datagen = ImageDataGenerator(
    rescale=1./255,           # normalize pixels
    rotation_range=20,        # rotate up to 20 degrees
    width_shift_range=0.2,    # shift horizontally
    height_shift_range=0.2,   # shift vertically
    shear_range=0.2,          # shear transform
    zoom_range=0.2,           # zoom in/out
    horizontal_flip=True,     # flip horizontally
    fill_mode='nearest'       # fill pixels after transformation
)

# No augmentation for validation set, only rescaling
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'datasets/train',
    target_size=(224, 224),
    batch_size=8,
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    'datasets/val',
    target_size=(224, 224),
    batch_size=8,
    class_mode='categorical'
)
