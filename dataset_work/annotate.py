import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import tkinter as tk
from PIL import Image, ImageTk
import os
import random
import json

# import tkinter as tk
# from tkinter import Entry, Button
# from PIL import Image, ImageTk

TARGET_DATASET_PATH = './../medium_dataset/'

SELECT_DATASETS = ['valid'] 

PERSONALITY_TOGGLE = False
IMAGE_TOGGLE = True

def main():

    rs = []
    img_files = []
    # get all dog ids in dataset
    for folder in [TARGET_DATASET_PATH + f'{p}/Images'for p in SELECT_DATASETS]:
        for breed in os.listdir(folder):
            if breed != '.DS_Store':
                for i in os.listdir(folder + f'/{breed}'):
                    if i != ".DS_Store":
                        img_files += [folder + f'/{breed}/' + i ]

    personalityfile = open('./personalities.json', 'r')
    personalities = json.load(personalityfile)

    random.shuffle(img_files)    

    for i, img_file in enumerate(img_files):
        if i % 10 == 0:
            print(str(i) + " out of " + str(len(img_files)) + " images")

        id = img_file[-(str(img_file[::-1]).find('/')) : -4]

        win = tk.Tk()
        win.geometry("800x800")
        frame = tk.Canvas(win, width=750, height=750)
        frame.place(anchor='center', relx=0.5, rely=0.5)
        frame.pack()

        image = Image.open(img_file)
        max_width = 250
        pixels_x, pixels_y = tuple([int(max_width/image.size[0] * x) for x in image.size])
        img = ImageTk.PhotoImage(image.resize((pixels_x, pixels_y))) 
        if not IMAGE_TOGGLE:
            img = ImageTk.PhotoImage(image.resize((1,1)))

        p_vec = personalities[id]

        personality = ''
        if PERSONALITY_TOGGLE:
            personality = f'playfulness: {p_vec[0]}\nchase-proneness: {p_vec[1]}\ncuriosity: {p_vec[2]}\nsociability: {p_vec[3]}\naggressiveness: {p_vec[4]}\nshyness: {p_vec[5]}\n'

        l = tk.Label(frame, text=f'{personality}', image = img, compound='bottom').pack()

        def label_dog(m):
            # m = 1 for yes, m = 0 for no
            new_row = {'id': id, 'label': m}
            rs.append(new_row)
            win.destroy()

        yes = tk.Button(frame, text="yes", command=lambda m=1: label_dog(m)).pack()
        no = tk.Button(frame, text="no", command=lambda m=0: label_dog(m)).pack()
        win.mainloop()

    labels = pd.DataFrame(rs)

    out = '_'.join(SELECT_DATASETS)
    labels.to_csv(f'./labels/vrushank_{out}_personality{PERSONALITY_TOGGLE}_image{IMAGE_TOGGLE}_labels.csv', )

    personalityfile.close()



if __name__ == "__main__":
    main()