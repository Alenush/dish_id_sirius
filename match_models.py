from abc import ABC, abstractmethod
from scipy.spatial.distance import cosine

class MatchModel(ABC):
  def __init__(self, mean_recipes, word_model):
    self.mean_recipes = mean_recipes
    self.word_model = word_model

  @abstractmethod
  def mean_embedding(self, ingred_list):
    pass

  def most_similar(self, ingred_list, k):
    mean_vector = self.mean_embedding(ingred_list)
    cos_sim = self.mean_recipes.apply(lambda v: cosine(v, mean_vector)).\
                  sort_values()[:k]
    return cos_sim, cos_sim.index


class Word2Vec(MatchModel):
  def __init__(self, mean_recipes, word_model):
    super().__init__(mean_recipes, word_model)

  def mean_embedding(self, ingred_list):
    words = ' '.join(tag_ud(ingred_list)).split()
    vectors = [self.word_model[word] for word in words]
    return np.mean(vectors, axis=0)

class FastText(MatchModel):
  def __init__(self, mean_recipes, word_model):
    super().__init__(mean_recipes, word_model)

  def mean_embedding(self, ingred_list):
    words = ' '.join(tag_ud(ingred_list, keep_pos=False)).split()
    vectors = [self.word_model[word] for word in words if word in self.word_model.vocab]
    return np.mean(vectors, axis=0)

class Elmo(MatchModel):
  def __init__(self, mean_recipes, word_model):
    super().__init__(mean_recipes, word_model)

  def mean_embedding(self, ingred_list):
    return self.word_model([ingred_list])[0]