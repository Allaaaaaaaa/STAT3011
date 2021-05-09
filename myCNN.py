import tensorflow as tf
import pathlib
import random

data_root = pathlib.Path('data/')
a_dir_path = data_root/'部落艺术'
a_image_paths = list(a_dir_path.glob('*.jpg'))
a_image_paths = [str(path) for path in a_image_paths]
random.shuffle(a_image_paths)

b_dir_path = data_root/'古典主义'
b_image_paths = list(b_dir_path.glob('*.jpg'))
b_image_paths = [str(path) for path in b_image_paths]
random.shuffle(b_image_paths)

c_dir_path = data_root/'极简主义'
c_image_paths = list(c_dir_path.glob('*.jpg'))
c_image_paths = [str(path) for path in c_image_paths]
random.shuffle(c_image_paths)

image_count = len(a_image_paths)
print(a_image_paths[1])