from abc import ABC, abstractmethod
from PIL import Image

class IngredModel(ABC):
  @abstractmethod
  def predict(self, img_path):
    pass


class DenseChefNet(IngredModel):
  def __init__(self, model, threshold=0.25):
    self.model = model
    self.threshold = threshold

  def predict(self, img_path):
    img = Image.open(img_path)
    img = transform_val(img).to(device).unsqueeze(0)
    
    with torch.no_grad():
      ingred_pred = self.model(img) > self.threshold
    
    ingred_pred = ingred_pred.nonzero()[:, 1].tolist()
    ingred_pred = [id2word[ing+1] for ing in ingred_pred]
    ingred_pred = [en2ru[ing] for ing in ingred_pred]

    return ingred_pred