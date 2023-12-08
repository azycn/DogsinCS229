import os
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

# breed => idx
# for each datapoint: one-hot vector with 1 at index of breed
# labels = y/n

DATASET_PATH = '../medium_dataset'

def get_breed_map(SELECT_DATASET):
    dataset_parts = os.listdir(DATASET_PATH + f'/{SELECT_DATASET}/Images')
    image_files = [part for part in dataset_parts if part != ".DS_Store"]
    
    # training data only
    id_to_breed = {}
    all_ids = set()
    for breed in image_files:
        if breed == '.DS_Store':
            continue
        breed_name = breed.split("-")[1]
        ids_in_breed = os.listdir(DATASET_PATH + f'/{SELECT_DATASET}/Images/' + breed)
        ids_in_breed = [id for id in ids_in_breed if id != '.DS_Store']
        all_ids = all_ids.union(set(ids_in_breed))
        for id in ids_in_breed:
            id_to_breed[id] = breed_name

    return id_to_breed


def prep_data(person, DATASET):
    imgid_to_label = pd.read_csv(f'../dataset_work/labels/image_only/{person}_{DATASET}_personalityFalse_imageTrue_labels.csv').set_index("id").to_dict()
    imgid_to_breed = get_breed_map(DATASET)
    
    breed_to_idx = {}
    count = 0
    for img in imgid_to_breed:
        breed = imgid_to_breed[img]
        if breed not in breed_to_idx:
            breed_to_idx[breed] = count
            count += 1

    # looks like: [imgidx, 0, 0, ..., 1, ..., 0] where 1 is at the BREEDIDX of the vector
    x_data = {}
    for img_id in imgid_to_breed:
        breed = imgid_to_breed[img_id]
        breed_idx = breed_to_idx[breed]
        x_data[img_id] = [0 for _ in range(len(breed_to_idx))]
        x_data[img_id][breed_idx] = 1


    X_df = pd.DataFrame(x_data.values(),index=x_data.keys())
    X_df.columns = [breed for breed in breed_to_idx]
    
    return (X_df, imgid_to_label['label'])

# naive bayes on that 

# TODO: keep the same breed_to_idx map for train and test. ok for now bc both maps r the same rn
def main():
    train_X_df, train_Y = prep_data('alice', 'train')
    test_X_df, test_Y = prep_data('alice', 'valid')

    train_x, train_y = train_X_df.to_numpy(), np.array([train_Y[k] for k in train_Y])
    test_x, test_y = test_X_df.to_numpy(), np.array([test_Y[k] for k in test_Y])

    print("NAIVE BAYES ====================================")

    BNB = BernoulliNB()
    BNB.fit(train_x, train_y)


    print("TRAIN==================")
    bnb_train_preds = BNB.predict(train_x)
    print(classification_report(train_y, bnb_train_preds))
    bnb_score = BNB.score(train_x, train_y)
    print("BNB Train Score: ", bnb_score)


    print("TEST==================")
    bnb_preds = BNB.predict(test_x)
    print(classification_report(test_y, bnb_preds))
    # print(bnb_preds)
    bnb_score = BNB.score(test_x, test_y)
    print("BNB Test Score: ", bnb_score)


    print("LOGISTIC REGRESSION ====================================")
    LR = LogisticRegression()
    LR.fit(train_x, train_y)

    print("TEST==================")
    lr_train_preds = LR.predict(train_x)
    print(classification_report(train_y, lr_train_preds))

    print("TEST==================")
    lr_preds = LR.predict(test_x)
    print(classification_report(test_y, lr_preds))

    # print(lr_preds)

    lr_score = LR.score(test_x, test_y)
    print("LR: ", lr_score)


if __name__ == "__main__":
    main() 