import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import tkinter as tk
from PIL import Image, ImageTk
import os

# import tkinter as tk
# from tkinter import Entry, Button
# from PIL import Image, ImageTk

DATASET_PATH = './../small_stanforddogdataset/'

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
        if i > 5:
            break
        win = tk.Tk()
        win.geometry("700x500")
        frame = tk.Canvas(win, width=750, height=750)
        frame.place(anchor='center', relx=0.5, rely=0.5)
        frame.pack()

        # Show image + resize to max size
        image = Image.open(DATASET_PATH + f'{SELECT_DATASET}_images/' + id_to_breed[id] + '/' + id)
        max_width = 250
        pixels_x, pixels_y = tuple([int(max_width/image.size[0] * x) for x in image.size])
        img = ImageTk.PhotoImage(image.resize((pixels_x, pixels_y))) 
        l = tk.Label(frame, text='insert personality here', image = img, compound='bottom').pack()

        def label_dog(m):
            # m = 1 for yes, m = 0 for no
            new_row = {'id': id, 'label': m}
            rs.append(new_row)
            win.destroy()

        yes = tk.Button(frame, text="yes", command=lambda m=1: label_dog(m)).pack()
        no = tk.Button(frame, text="no", command=lambda m=0: label_dog(m)).pack()
        win.mainloop()

    labels = pd.DataFrame(rs)

    labels.to_csv(f'{SELECT_DATASET}_labels.csv', index=False)

            



if __name__ == "__main__":
    main()