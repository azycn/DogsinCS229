import os
import json
import numpy as np

TARGET_DATASET_PATH = './medium_dataset'

def main():
    """
    [playfulness, chase-proneness, curiosity, sociability, aggressiveness, shyness]
    """

    dog_ids = []

    personalities = {}

    # get all dog ids in dataset
    for folder in [TARGET_DATASET_PATH + f'/{p}/Images'for p in ['test', 'train', 'valid']]:
        for breed in os.listdir(folder):
            if breed != '.DS_Store':
                dog_ids += [i[:-4] for i in os.listdir(folder + f'/{breed}')]

    # generate personalities
    for dog_id in dog_ids:
        personalities[dog_id] = [round(min(max(np.random.normal(5, 2), 1), 10)) for _ in range(6)]
    
    # write data
    with open('personalities.json', '+w') as outfile:
        json.dump(personalities, outfile)

if __name__ == "__main__":
    main()