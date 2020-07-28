import numpy as np
import pandas as pd
import pickle as pkl

from sklearn.feature_extraction.text import TfidfVectorizer
import pymorphy2
orph = pymorphy2.MorphAnalyzer()

FIT_COLUMN = 'spl'
PREDICT_COLUMN = 'nor_ingridients'
N_BEST = 10

PATH = '/mnt/c/Users/Saidash/Documents/yanius/saydashtat/offtop/'

def splinst(inst): #это функция, которая принимает на вход текст инструкции и дает обратно текст с нормальными формами слов. Ну и чистит всякие предлоги, цифры и т.д. по возможности.
    I = ' '.join(inst) #это нужно!
    I = I.split(' ')
    #print (I)
    #I = inst
    for i in range(len(I)):
        I[i] = I[i].lower()
        if len(I[i])>2:
            if ord(I[i][-1])<1072 or ord(I[i][-1])>1103:
                I[i] = I[i][:-1]
            if ord(I[i][0])<1072 or ord(I[i][0])>1103:
                I[i] = I[i][1:]
            #print(orph.parse(I[i]))
            p = orph.parse(I[i])[0]
            I[i] = (p.normal_form)
            if I[i]==None:
                I[i] = ''
        else:
            I[i] = ''
    return ' '.join(I)


def create_text_for_vectorizer(lst_base):
    lst = lst_base.copy()
    for i in range(len(lst_base)):
        lst.extend([lst_base[i]] * int(max(6 // (0.5 * i + 1), 1)))
    return " ".join([''.join(elem.split()) for elem in lst])


def prepare_dataset(df):
    df = df.copy()

    print('preparing dataset')
    df.index = [i for i in range(len(df))]
    df['splinstr'] = df.instructions.apply(splinst)

    print('splinstr applied')
    df['nor_ingridients'] = df.pure_ingridients.apply(lambda x: list(map(lambda y: orph.parse(y)[0].normal_form, x)))
    df['nor_ingridients'] = df.nor_ingridients.apply(lambda x: create_text_for_vectorizer(x))

    df['splinctr2'] = df.splinstr.apply(lambda x: x.split(' '))

    spl = []  # запихнуть все в листы как мудак -- могу умею практикую
    for i in range(len(df)):
        dd = df.splinstr[i].split(' ')
        # for s in df2.splinstr[i]:
        #    dd = dd + ' ' + s
        # dd = dd.split(' ')
        nn = df.nor_ingridients[i]
        # for a in df2.nor_ingridients[i]:
        #    nn = nn + a
        sss = []
        # print (list(df2.nor_ingridients[i]))
        for j in range(len(dd)):
            # print (dd[j])
            if dd[j] in list(nn):
                sss.append(dd[j])
        spl.append(sss)
        # df2.splinctr2[i] = sss
    df['spl'] = spl  # нормализованные данные (по упоминанию слов в тексте рецепта)

    return df


# для каждого элемента тестовой выборки берем лучшие мэтчи из трэйна
def get_best_matches(array, n):
    #     print(array[0])
    x = np.argsort(np.array(array))
    #     print(x)
    return (x[-n:])


def fit_predict(train, test):
    print('fit_predict_started')
    vectorizer = TfidfVectorizer(min_df=2)

    train_tfidf = vectorizer.fit_transform(
        train[FIT_COLUMN].apply(lambda x: " ".join([''.join(elem.split()) for elem in x])))
    with open('vectorizer.pkl', 'wb') as file:
        pkl.dump(vectorizer, file, protocol=4)
    with open('train_tfidf.pkl', 'wb') as file:
        pkl.dump(train_tfidf, file, protocol=4)

    test_tfidf = vectorizer.transform(test[PREDICT_COLUMN])  # .apply(lambda x: " ".join(x)))
    #     display(test_tfidf.todense())
    #     print(train_tfidf.shape, test_tfidf.shape)
    result = np.array(test_tfidf).dot(np.array(train_tfidf.T))
    result = pd.DataFrame(np.array(result.todense()))
    best_matches = result.apply(lambda x: get_best_matches(x, N_BEST), axis=1)
    return best_matches


def main(train, test):
    best_matches = fit_predict(train, test)
    best_matches.index = test.nor_ingridients
    best_matches = pd.DataFrame(best_matches)
    # best_matches = pd.DataFrame(best_matches.apply(lambda x: [train['name'][i] for i in x]))
    best_matches.columns = ['recipes']

    final_best_matches = pd.DataFrame(best_matches.recipes.tolist(), index=best_matches.index)
    melt_matches = final_best_matches.reset_index()\
        .melt(id_vars=['nor_ingridients'])\
        .rename({'variable': 'priority', 'nor_ingridients': 'test_ingridients'}, axis=1)\
        .set_index(['test_ingridients', 'priority'])

    melt_matches = melt_matches['value'].apply(
        lambda x: train[['name', 'pure_ingridients', 'instructions', 'img_url', 'recipe_link']].iloc[x])

    melt_matches.sort_index().to_csv('matches_gold_fulldb_pred.csv')


if __name__ == '__main__':
    # print(pd.read_csv('best_matches_gold_pred.csv').iloc[:, :2])
    # with open(PATH + 'recipes_final.pkl', 'rb') as f:
    #     df = pkl.load(f)
    # train = prepare_dataset(df)

    with open(PATH + 'eda_povar.pkl', 'rb') as f:
        train = pkl.load(f)
    print(train.head(), train.columns, train.shape)

    test = pd.read_csv(PATH + 'df_matching_win.csv').rename({'prediction': 'nor_ingridients'}, axis=1)

    # train_base = df_norm.sample(frac=1).reset_index()
    # # Пока что тест - сэмпл из выборки рецептов.
    # # Нужно будет поменять при тесте
    # train = train_base[100:].reset_index()
    # test = train_base[:100]

    main(train, test)
