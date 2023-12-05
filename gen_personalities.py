import os
import random
import json


TARGET_DATASET_PATH = './medium_dataset'

def main():
    """
    [playfulness, chase-proneness, curiosity, sociability, aggressiveness, shyness]
    """

    # first, get all of the dog IDs in the dataset

    dog_ids = []

    personalities = {}

    for folder in [TARGET_DATASET_PATH + f'/{p}/Images'for p in ['test', 'train', 'valid']]:
        for breed in os.listdir(folder):
            if breed != '.DS_Store':
                dog_ids += [i[:-4] for i in os.listdir(folder + f'/{breed}')]

    for dog_id in dog_ids:
        personalities[dog_id] = [random.randint(1,10) for _ in range(6)]
    
    with open('personalities.json', '+w') as outfile:
        json.dump(personalities, outfile)

if __name__ == "__main__":
    main()