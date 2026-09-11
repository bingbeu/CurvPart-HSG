import os
import shutil

train_test_split_f = '/raid/datasets/cub-200/CUB_200_2011/train_test_split.txt'
f = open(train_test_split_f, 'r')
train_test_splits = f.readlines()

images_f = '/raid/datasets/cub-200/CUB_200_2011/images.txt'
f = open(images_f, 'r')
images = f.readlines()

image_data_folder = '/raid/datasets/cub-200/CUB_200_2011/images'
save_folder = '/raid/datasets/cub-200/CUB_200_2011/images_split'

for i in range(len(images)):
    image_id, image_name = images[i].strip().split(' ') #001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg
    train_id, istrain = train_test_splits[i].strip().split(' ')
    istrain = int(istrain)
    image_folder = image_name.split('/')[0]
    assert image_id == train_id
    if istrain:
        if not os.path.exists(os.path.join(save_folder, 'train', image_folder)):
            os.makedirs(os.path.join(save_folder, 'train', image_folder))
        #os.rename(os.path.join(image_data_folder, image_name), os.path.join(save_folder, 'train', image_name))
        shutil.copy(os.path.join(image_data_folder, image_name), os.path.join(save_folder, 'train', image_name))
    else:
        if not os.path.exists(os.path.join(save_folder, 'test', image_folder)):
            os.makedirs(os.path.join(save_folder, 'test', image_folder))
        #os.rename(os.path.join(image_data_folder, image_name), os.path.join(save_folder, 'test', image_name))
        shutil.copy(os.path.join(image_data_folder, image_name), os.path.join(save_folder, 'test', image_name))


    