from keras.preprocessing.image import ImageDataGenerator

def getImageAndMaskGenerator(image_dir, mask_dir, shuffle=True, batch_size=8, seed=123):
    """
    Based on here: https://github.com/keras-team/keras/issues/3059, zip two generators
    to form a generator to produce images and masks in a synchronous way.
    :return: a zipped generator can be fed to model.fit_generator method in Keras.
    """
    image_gen = ImageDataGenerator()
    mask_gen = ImageDataGenerator()

    image_generator = image_gen.flow_from_directory(image_dir, class_mode=None, seed=seed, batch_size=batch_size, color_mode='grayscale', shuffle=shuffle)
    mask_generator = mask_gen.flow_from_directory(mask_dir, class_mode=None, seed=seed, batch_size=batch_size, color_mode='grayscale', shuffle=shuffle)

    return zip(image_generator, mask_generator)


