from scipy.spatial.distance import cosine

pos_morphy_to_vec = {'ADJF':'ADJ'} # перевод части речи из pymorphy в word2vec


def match_recipes(ingred_list, recipes_df, word_model, k=10):
  """
  Возвращает k ближайших рецептов по косинусному расстоянию.
  :param ingred_list: список ингедиенто
  :param recipes_df: датасет для матчинга
  :param word_model: векторная модель (word2vec, fasttext, elmo)
  :param k: кол-во возвращаемых рецептов
  :returns: k рецептов
  """
  vectors = []
  for word in ingred_list:
    normal_word = morph.parse(word)[0].normalized
    pos = normal_word.tag.POS
    if pos in pos_morphy_to_vec:
      pos = pos_morphy_to_vec[normal_word.tag.POS]
    word = normal_word.word
    if pos != None:
      word_pos = f'{word}_{pos}'
      if word_pos in word_model:
        vectors.append(word_model[word_pos])

  mean_vector = np.mean(vectors, axis=0)
  cos_sim = recipes_df['word2vec_mean'].apply(lambda v: cosine(v, mean_vector)).sort_values()[:k]
  closest_recipes = recipes_df.loc[cos_sim.index]
  closest_recipes['cos_sim'] = cos_sim
  
  return closest_recipes
