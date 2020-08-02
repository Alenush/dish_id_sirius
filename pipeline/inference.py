import os
import json
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from collections import defaultdict
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict

def predict_image(img_path, df, ingred_model, word_model, k=10, threshold=0.25):
  ingred_pred = ingred_model.predict(img_path)
  cos_sim, best_indices = word_model.most_similar(ingred_pred, k)

  return cos_sim, df.loc[best_indices]

def main(args):
  img = Image.open(args.img_path)

  plt.figure(figsize=(12,8))
  plt.imshow(img)
  plt.axis('off')
  plt.show()

  cos_sim, best_rows = predict_image(img_path, df, chefnet, elmo_model)

  for i, row in enumerate(best_rows.iterrows()):
    row = row[1]
    print(f"{i + 1}) {row['name']}")
    print('\t' + row['ingreds'])
    print(f"\t{row['url']}")
