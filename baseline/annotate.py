import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
# import tkinter as tk
# from tkinter import Entry, Button
# from PIL import Image, ImageTk

DATASET_PATH = './small_stanforddogdataset/'

# def load_next_image():
    

# def submit_text():
#     pass

SELECT_DATASET = 'test'


def main():
    dataset_parts = os.listdir(DATASET_PATH + f'{SELECT_DATASET}_images')

    image_files = [part for part in dataset_parts if 'images' in part]
    
    # training data only
    id_to_breed = {}
    all_ids = set()
    for breed in image_files:
        if breed == '.DS_Store':
            continue
        ids_in_breed = os.listdir(DATASET_PATH + f'{SELECT_DATASET}_images/' + breed)
        ids_in_breed = [id for id in ids_in_breed if id != '.DS_Store']
        all_ids = all_ids.union(set(ids_in_breed))
        for id in ids_in_breed:
            id_to_breed[id] = breed
   
    rs = []
    num_dogs = len(all_ids)
    for i, id in enumerate(list(all_ids)):

        img = mpimg.imread(DATASET_PATH + f'{SELECT_DATASET}_images/' + id_to_breed[id] + '/' + id)
        imgplot = plt.imshow(img)
        plt.show()
        label = None

        while label is None or label.lower()[0] not in ['y', 'n']:
            label = input(f'Do you like this dog {i/num_dogs*100:.3f} % (y/n): ')
        plt.close()

        new_row = {'id': id, 'label': 1 if label.lower()[0] == 'y' else 0}
        rs.append(new_row)

    labels = pd.DataFrame(rs)

    labels.to_csv(f'{SELECT_DATASET}_labels.csv', index=False)

            



if __name__ == "__main__":
    main()