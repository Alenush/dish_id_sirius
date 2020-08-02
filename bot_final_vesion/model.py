import torch
import torch.nn as nn
from torchvision import transforms, models
from collections import defaultdict
import matplotlib.pyplot as plt
from PIL import Image
device = torch.device('cpu')
import json


en2ru = {}
with open('/home/dishid_bot/clean_ingred.txt', 'r') as f:
  en=f.readlines()
with open('/home/dishid_bot/rus_clean_ingred.txt', 'r',encoding='utf-8') as f:
  ru=f.readlines()

for e, r in zip(en, ru):
  en2ru[e.rstrip('\n')] = r.rstrip('\n')
with open('/home/dishid_bot/en2ru_ing.json', 'w') as f:
  json.dump(en2ru, f)


with open('/home/dishid_bot/clean_ingred.txt', 'r') as f:
  clean_ingredients = list(f.read().split('\n'))
  print(len(clean_ingredients))

id2word = defaultdict()
for i, ingr in enumerate(clean_ingredients):
  id2word[i+1] = ingr

rus_id2word = defaultdict()
for i, ingr in enumerate(clean_ingredients):
  rus_id2word[i+1] = ingr

word2id = {v:k for k,v in id2word.items()}
def words2ids(ingreds):
  return [word2id[ing] for ing in ingreds]

model = models.densenet161(pretrained=True)

def init_model():

    for param in model.parameters():
        param.requires_grad = False
    num_feat = model.classifier.in_features

    model.classifier = nn.Sequential(
        nn.Linear(num_feat, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, len(word2id)),
        nn.Sigmoid())
    model.to(device)

    model.load_state_dict(torch.load("/home/dishid_bot/model_encoder_classifier_best.pth",map_location=torch.device('cpu')))
    model.eval()


def predict_image(path_to_img, transform):
    with torch.no_grad():
        img = Image.open(path_to_img).convert('RGB')
    plt.figure(figsize=(12,8))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    img = transform(img).to(device).unsqueeze(0)

    ingred_pred = model(img) > 0.25
    ingred_pred = ingred_pred.nonzero()[:, 1].tolist()
    ingred_pred = [id2word[ing + 1] for ing in ingred_pred]
    ingred_pred = [en2ru[ing] for ing in ingred_pred]
    return '\t' + '\n\t'.join(ingred_pred)