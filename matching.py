import numpy as np
import pandas as pd
import pickle as pkl

from sklearn.feature_extraction.text import TfidfVectorizer
import pymorphy2

orph = pymorphy2.MorphAnalyzer()

FIT_COLUMN = "spl"
PREDICT_COLUMN = "not_ingridients"
N_BEST = 10


# NOTE: Есть такая штука как docstrings (ко всем функциям добавить)
# Не стоит писать странные "это нужно", объясни сразу что идет на вход (вместе с типом) функции, а что на выход
def splinst(inst):
    """
    это функция, которая принимает на вход текст инструкции и
    дает обратно текст с нормальными формами слов.
    Ну и чистит всякие предлоги, цифры и т.д. по возможности.

    :param array inst: описание что на вход
    :return: что возвращает в каком типе данных прописать
    """
    I = " ".join(inst)  # это нужно!
    I = I.split(" ")
    # NOTE: довольно странная махинация со сплитом. Я понимаю что после парсинга там все странно, но после джойна сделай тогда токенизациб чище
    # Есть вот такая штука, чтобы разбить на предложения рецепт https://pypi.org/project/rusenttokenize/
    # Далее по токенам есть https://github.com/aatimofeev/spacy_russian_tokenizer и много других
    # они разбивают уже даже отдельно пунткуацию (не должно быть "token,", а после сплита у тебя грязные токены с лишними символами)
    for i in range(len(I)):
        I[i] = I[i].lower()
        if len(I[i]) > 2:
            if ord(I[i][-1]) < 1072 or ord(I[i][-1]) > 1103:
                I[i] = I[i][:-1]
            if ord(I[i][0]) < 1072 or ord(I[i][0]) > 1103:
                I[i] = I[i][1:]
            # NOTE: интересный подход, я бы использовала обычные regex, чтобы наверняка
            # чтобы убрать еще мусор всякий (ссылки, лишние пробелы и прочее)
            # с помошью pymorphy убрать стоп слова (те части речи хотя бы такие как PREP, CONJ, PRCL, INTJ)
            p = orph.parse(I[i])[0]
            I[i] = p.normal_form
            if I[i] == None:
                I[i] = ""
        else:
            I[i] = ""
    return " ".join(I)


def create_text_for_vectorizer(lst_base):
    lst = lst_base.copy()
    for i in range(len(lst_base)):
        lst.extend([lst_base[i]] * int(max(6 // (0.5 * i + 1), 1)))
    return " ".join(["".join(elem.split()) for elem in lst])


def prepare_dataset(df):
    df = df.copy()

    df.index = [i for i in range(len(df))]
    df["splinstr"] = df.instructions.apply(splinst)

    df["nor_ingridients"] = df.pure_ingridients.apply(
        lambda x: list(map(lambda y: orph.parse(y)[0].normal_form, x))
    )
    df["not_ingridients"] = df.nor_ingridients.apply(
        lambda x: create_text_for_vectorizer(x)
    )

    df["splinctr2"] = df.splinstr.apply(lambda x: x.split(" "))

    spl = []  # запихнуть все в листы как мудак -- могу умею практикую
    for i in range(len(df)):
        dd = df.splinstr[i].split(" ")
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
        # NOTE: да сразу бы в строки тогда уж, чтобы потом не join-ить
        # Мысли такие: у тебя тут добавляются из рецептов только те слова,
        # которые есть в составе нормализованном (кстати состав проверить что все к lower привидено и чистые без пунктуации и тд),
        # это хорошо для подсчета пересечения множеств (попробовать сделать
        # не только такой искусственный tf-idf а прям пересечение ручками и выбирать наилучшие 10)
        # Попробуйте как вариант еще не исключать всё остальное в tf-idf, а прям нормализованный чистый текст отправлять в fit
        spl.append(sss)
        # df2.splinctr2[i] = sss
    df["spl"] = spl  # нормализованные данные (по упоминанию слов в тексте рецепта)
    # NOTE:  Название рецепта бы тоже учитывала (в нем тоже могут быть упоминание ингредиентов)

    return df


# для каждого элемента тестовой выборки берем лучшие мэтчи из трэйна
def get_best_matches(array, n):
    #     print(array[0])
    x = np.argsort(np.array(array))
    #     print(x)
    return x[-n:]


def fit_predict(train, test):
    vectorizer = TfidfVectorizer(min_df=2)

    train_tfidf = vectorizer.fit_transform(
        train[FIT_COLUMN].apply(
            lambda x: " ".join(["".join(elem.split()) for elem in x])
        )
    )
    test_tfidf = vectorizer.transform(
        test[PREDICT_COLUMN]
    )  # .apply(lambda x: " ".join(x)))
    # NOTE: посмотри еще get_feature_names() на что реагирует

    #     display(test_tfidf.todense())
    #     print(train_tfidf.shape, test_tfidf.shape)
    result = np.array(test_tfidf).dot(np.array(train_tfidf.T))
    result = pd.DataFrame(np.array(result.todense()))
    best_matches = result.apply(lambda x: get_best_matches(x, N_BEST), axis=1)
    return best_matches


if __name__ == "__main__":
    with open("recipes_final.pkl", "rb") as f:
        df = pkl.load(f)
    df_norm = prepare_dataset(df)

    train_base = df_norm.sample(frac=1).reset_index()

    # Пока что тест - сэмпл из выборки рецептов.
    # Нужно будет поменять при тесте
    train = train_base[100:].reset_index()
    test = train_base[:100]

    best_matches = fit_predict(train, test)
    best_matches = pd.DataFrame(
        best_matches.apply(lambda x: [train["name"][i] for i in x])
    )
    best_matches.index = test.name
    best_matches.columns = ["names"]

    final_best_matches = pd.DataFrame(
        best_matches.names.tolist(), index=best_matches.index
    )
    final_best_matches.to_csv("best_matches.csv")

# NOTES: Не хватает размеченной выборки и метрик


# NOTES: Есть такая штука https://github.com/psf/black
# она по PEP-8 все делает, я применила к файлу, он отформотировался
