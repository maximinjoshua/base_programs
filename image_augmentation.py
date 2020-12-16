# import os
# from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# os.chdir('/home/maximin/tact_labs/tact_lab_synthetic_data/images/inputimages')
# pathlist=[]
# for i in os.listdir():
#     pathlist.append(os.path.abspath(i))

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=40,
        height_shift_range=0.2,
        width_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


i = 0
for batch in datagen.flow_from_directory('/home/maximin/tact_labs/data_augmentation/images/inputimages', batch_size=6,target_size=(256,256),
                          save_to_dir='/home/maximin/tact_labs/data_augmentation/images/outputimages', save_prefix='cornflakes', save_format='jpg'):
    i += 1
    if i > 5:
        break


