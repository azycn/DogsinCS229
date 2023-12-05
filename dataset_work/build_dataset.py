import os
import random
import shutil

n_breeds = 25
n_pics_train = 15 # per breed
n_pics_test = 5 
n_pics_valid = 5
n_pics_total = n_pics_train + n_pics_test + n_pics_valid

def main():
    new_loc = "/Users/alice/Documents/Stanford/cs229/STANFORDDOGDATASET/medium_dataset"
    og_data_loc = "/Users/alice/Documents/Stanford/cs229/STANFORDDOGDATASET/Stanford_Dog_Dataset"

    os.chdir(og_data_loc + "/Images")
    image_dirs = os.listdir('.')
    rand_breeds = random.sample(image_dirs, k=n_breeds)
    print(rand_breeds)
    build_set(og_data_loc, new_loc, rand_breeds)

def build_set(cur_dir, new_dir, rand_breeds):
    # Images
    for breed in rand_breeds:
        os.mkdir(new_dir + '/train/Images/' + breed)
        os.mkdir(new_dir + '/test/Images/' + breed)
        os.mkdir(new_dir + '/valid/Images/' + breed)
        os.mkdir(new_dir + '/train/Annotations/' + breed)
        os.mkdir(new_dir + '/test/Annotations/' + breed)
        os.mkdir(new_dir + '/valid/Annotations/' + breed)

        os.chdir(cur_dir + '/Images/' + breed)
        all_images = os.listdir(".")
        for i in range(n_pics_total):
            if i < n_pics_train:
                # Train 
                shutil.copy2(all_images[i], new_dir + '/train/Images/' + breed + "/" + all_images[i])
            elif i < n_pics_train + n_pics_valid:
                # Test 
                shutil.copy2(all_images[i], new_dir + '/test/Images/' + breed + "/" + all_images[i])
            else:
                # Valid 
                shutil.copy2(all_images[i], new_dir + '/valid/Images/' + breed + "/" + all_images[i])

    # Annotations
        os.chdir(cur_dir + '/Annotation/' + breed)
        all_annotations = os.listdir(".")
        for i in range(n_pics_total):
            if i < n_pics_train:
                # Train 
                shutil.copy2(all_annotations[i], new_dir + '/train/Annotations/' + breed + "/" + all_annotations[i])
            elif i < n_pics_train + n_pics_valid:
                # Test 
                shutil.copy2(all_annotations[i], new_dir + '/test/Annotations/' + breed + "/" + all_annotations[i])
            else:
                # Valid 
                shutil.copy2(all_annotations[i], new_dir + '/valid/Annotations/' + breed + "/" + all_annotations[i])

if __name__ == "__main__":
    main()



