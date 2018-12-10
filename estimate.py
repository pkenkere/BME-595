import sys
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from train import Network
from torch.autograd import Variable
from torchvision import transforms, utils

import cv2

from train import Network

GENDER_MODEL_PATH = "./gender_est_model.pt"
RACE_MODEL_PATH = "./race_est_model.pt"
AGE_MODEL_PATH = "./age_est_model.pt"

def get_gender(image):
    model = Network(2)
    model.load_state_dict(torch.load(GENDER_MODEL_PATH))
    estimated_gender = model(Variable(image))
    estimated_gender = estimated_gender.detach().numpy().tolist()
    estimated_gender = estimated_gender[0]
    print(estimated_gender)
    estimated_gender = estimated_gender.index(max(estimated_gender))
    print(estimated_gender)
    labels = ["male", "female"]
    print("Gender predicted: ", labels[estimated_gender])

def get_age(image):
    model = Network(12)
    model.load_state_dict(torch.load(AGE_MODEL_PATH))
    estimated_age = model(Variable(image))
    estimated_age = estimated_age.detach().numpy().tolist()
    estimated_age = estimated_age[0]
    print(estimated_age)
    estimated_age = estimated_age.index(max(estimated_age))
    print(estimated_age)
    labels = ["1-10", "11-20", "21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "81-90", "91-100", "101-110", "111-120"]
    print("Age predicted: ", labels[estimated_age])

def get_race(image):
    model = Network(5)
    model.load_state_dict(torch.load(RACE_MODEL_PATH))
    estimated_race = model(Variable(image))
    estimated_race = estimated_race.detach().numpy().tolist()
    estimated_race = estimated_race[0]
    print(estimated_race)
    estimated_race = estimated_race.index(max(estimated_race))
    print(estimated_race)
    labels = ["White", "Black", "Asian", "Indian", "Others"]
    print("Race predicted: ", labels[estimated_race])

def main(argv):
    if len(argv) != 1:
        print("Usage: python3 estimate.py imagepath")
        exit()

    image_path = argv[0]
    image = cv2.imread(image_path)
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    image = trans(image)
    image = image.reshape(-1, 3, 200, 200)

    gender = get_gender(image)
    age = get_age(image)
    race = get_race(image)

    return

if __name__ == "__main__":
    main(sys.argv[1:])