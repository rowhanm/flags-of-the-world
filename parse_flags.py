from os import listdir
from os.path import isfile, join
import numpy as np
import cv2

#Locating files
img_path_africa = './flags-normal/africa/'
img_path_asia = './flags-normal/asia/'
img_path_europe = './flags-normal/europe/'
img_path_north_america = './flags-normal/north_america/'
img_path_oceania = './flags-normal/oceania/'
img_path_south_america = './flags-normal/south_america/'

#Adding flag images in a list as per continent
flags_asia = [img_path_asia + f for f in listdir(img_path_asia) if isfile(join(img_path_asia, f))]
flags_africa = [img_path_africa + f for f in listdir(img_path_africa) if isfile(join(img_path_africa, f))]
flags_europe = [img_path_europe + f for f in listdir(img_path_europe) if isfile(join(img_path_europe, f))]
flags_north_america = [img_path_north_america + f for f in listdir(img_path_north_america) if isfile(join(img_path_north_america, f))]
flags_oceania = [img_path_oceania + f for f in listdir(img_path_oceania) if isfile(join(img_path_oceania, f))]
flags_south_america = [img_path_south_america + f for f in listdir(img_path_south_america) if isfile(join(img_path_south_america, f))]

def show(image):
    cv2.imshow('image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def calc_split_mean(image):
    b, g, r = cv2.split(image)
    return np.array((np.mean(b), np.mean(g), np.mean(r)))

def normalize_flags(flags):
    flag_heights = []
    normalized_flags = []
    for idx in range(len(flags)):
        flag_heights.append(flags[idx].shape[0])
    max_flag_height = max(flag_heights)  #Calculate maximum height of all flags
    for idx in range(len(flags)):
        mean_value = calc_split_mean(flags[idx])
        mean_b, mean_g, mean_r = mean_value[0], mean_value[1], mean_value[2]
        new_image = [0]*3 #Empty list for 3 channels
        new_image_b = np.full((max_flag_height, flags[idx].shape[1]), mean_b)
        new_image_g = np.full((max_flag_height, flags[idx].shape[1]), mean_g)
        new_image_r = np.full((max_flag_height, flags[idx].shape[1]), mean_r)
        new_image[0] = new_image_b
        new_image[1] = new_image_g
        new_image[2] = new_image_r
        new_image = np.array(new_image, dtype=np.float32)
        new_image = np.moveaxis(new_image,0,-1)
        new_image[:flags[idx].shape[0],:flags[idx].shape[1]] = flags[idx]
        normalized_flags.append(new_image)
    return np.array(normalized_flags, dtype=np.float32)

def combine(flags):
    add = np.zeros(flags[0].shape)
    for idx in range(len(flags)):
        add = add + flags[idx]
    return add/len(flags)

# Change these and play around with several combination
# flags = flags_africa + flags_asia + flags_europe + flags_north_america + flags_south_america + flags_oceania

flags = flags_africa


#Creating list of 3D numpy arrays
flag_array = [0]*len(flags)
for i in range(len(flags)):
    image = cv2.imread(flags[i])
    image = image/255
    flag_array[i] = image
flags_np_array = np.array(flag_array, dtype=object)

#Since flags are of irregular heights, we gotta normalize
normalized_flags = normalize_flags(flags_np_array)

combined_image = combine(normalized_flags)
b, g, r = cv2.split(combined_image)
print(np.mean(b),np.mean(g),np.mean(r))
# show(combined_image)
