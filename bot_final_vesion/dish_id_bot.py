# To-do
# # Вывести в отдельный работающий пайплайн
# # Прикрутить распознование изображения
# # Прикрутить ранжирование
# # Прикрутить логирование


import aiohttp
import io
import logging
import os
import time
import numpy as np
import pandas as pd
import pickle as pkl
import typing as tp

from PIL import Image
from aiogram import Bot, types
from aiogram.utils import executor
from aiogram.dispatcher import Dispatcher
from aiogram.utils.helper import Helper, HelperMode, ListItem
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.types import ReplyKeyboardRemove, \
    ReplyKeyboardMarkup, KeyboardButton, \
    InlineKeyboardMarkup, InlineKeyboardButton
from aiofiles import os as aio_os
from torchvision import transforms
from model import init_model, predict_image


N_BEST = 5
best_recipes = dict()
TYPE = 'image'


# bot = Bot(token=os.environ.get('TOKEN', None),
TOKEN = 'TOKEN'
bot = Bot(token=TOKEN)
dp = Dispatcher(bot, storage=MemoryStorage())


init_model()
logging.info("Model was init")
logging.basicConfig(filename='log.txt',
                    filemode='a',
                    format='%(asctime)s, %(msecs) d %(name)s %(levelname) s %(message) s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

inline_keyboard_markup = types.InlineKeyboardMarkup()
inline_keyboard_markup.add(types.InlineKeyboardButton('Применить стиль Сезанна', callback_data='sezanne'))


class TestStates(Helper):
    mode = HelperMode.snake_case

    TEST_STATE_0 = ListItem()
    TEST_STATE_1 = ListItem()
    TEST_STATE_2 = ListItem()


button0 = KeyboardButton('/help')
buttons = [KeyboardButton('1')]
buttons.append(KeyboardButton('2'))
buttons.append(KeyboardButton('3'))
buttons.append(KeyboardButton('4'))
buttons.append(KeyboardButton('5'))
markup0 = ReplyKeyboardMarkup(resize_keyboard=True,
                              one_time_keyboard=True).add(
    button0)
markup1 = ReplyKeyboardMarkup(resize_keyboard=True,
                              one_time_keyboard=True).row(
    *buttons[:2])
markup5 = ReplyKeyboardMarkup(resize_keyboard=True,
                              one_time_keyboard=True).row(
    *buttons)


# Обработка 3 команд - старт, хелп и тим
@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message) -> None:
    state = dp.current_state(user=message.from_user.id)
    await state.set_state(TestStates.all()[0])
    await message.answer("Привет, {}!\n".format(message.from_user.first_name) +
                         "Я бот, который поможет тебе найти рецепт для блюда из фотографии")


@dp.message_handler(commands=['help'])
async def send_help(message: types.Message) -> None:
    logging.info("User asked for help")
    state = dp.current_state(user=message.from_user.id)
    await state.set_state(TestStates.all()[0])
    await message.answer("Нужна помощь? Решение очень простое!\n" +
                         "Просто отправь фотку блюда и алгоритм подберет оптимальный рецепт")


@dp.message_handler(state=TestStates.all(), commands=['start'])
async def send_welcome_state(message: types.Message) -> None:
    await send_welcome(message)


@dp.message_handler(state=TestStates.all(), commands=['help'])
async def send_help_state(message: types.Message) -> None:
    await send_help(message)


@dp.message_handler(state=TestStates.all(), content_types=['photo'])
async def handle_photo(message):
    if not os.path.isdir(f'/home/dishid_bot/photo/{message.from_user.id}'):
        await aio_os.mkdir(f'/home/dishid_bot/photo/{message.from_user.id}')
    await message.photo[-1].download(
        f'/home/dishid_bot/photo/{message.from_user.id}/{TYPE}.jpg')
    if TYPE == 'image':
        await bot.send_message(message.chat.id, "Отлично, картинка загружена! Сейчас скажу, какие ингредиенты я тут вижу")
    await apply_model(message)


@dp.message_handler(state=TestStates.TEST_STATE_1[0], commands=['run'])
async def apply_model(message: types.Message):
    logging.info("Get image from user {}".format(message.chat.id))
    transform = transforms.Compose([
        transforms.Resize((250, 250)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    path_image = f'/home/dishid_bot/photo/{message.chat.id}/{TYPE}.jpg'
    result = predict_image(path_image, transform)
    res_list = result.replace('\t', '').split('\n')

    if len(res_list) == 0:
        await message.answer("Я на фото ничего не нашел...\nПопробуй загрузить фото лучшего качества")

    logging.info("Found ingredients: {}".format(", ".join(res_list)))
    await message.answer("Я выделил ингредиенты на фото.\nВот что я тут вижу: {}\n\nСейчас по этому списку подберу рецептики".format(", ".join(res_list)))

    await match_recipes(" ".join(res_list), message)


@dp.message_handler(state=TestStates.TEST_STATE_1[0])
async def team_1(message: types.Message, redirect: bool = False) -> None:
    await message.answer("Отправь фото блюда, чтобы получить релевантный рецепт\n" +
                         "Если хочешь посмотреть другие рецепты по предыдущей фотографии, просто отправь её ещё раз")


async def match_recipes(ingridients, message):
    t0 = time.time()
    logging.info("Matching with ingridients list: {}".format(ingridients))
    with open('/home/dishid_bot/vectorizer.pkl', 'rb') as f:
        vectorizer = pkl.load(f)
    with open('/home/dishid_bot/botrain_tfidf.pkl', 'rb') as f:
        train_tfidf = pkl.load(f)
    with open('/home/dishid_bot/eda_povar.pkl', 'rb') as f:
        train = pkl.load(f)

    test_tfidf = vectorizer.transform([ingridients])
    result = np.array(test_tfidf).dot(np.array(train_tfidf.T))

    loc_best_matches = np.argsort(np.array(result.todense()))[0][-N_BEST:]
    loc_best_matches = pd.DataFrame(loc_best_matches)
    loc_best_matches.columns = ['idx']


    short_list = loc_best_matches['idx'].apply(
        lambda x: train[['name', 'ingridients', 'instructions', 'img_url', 'recipe_link']].iloc[x])

    logging.info('Recipes matched: {}'.format(list(short_list.name)))

    best_recipes[message.from_user.id] = short_list

    await message.answer("Выберите номер наиболее понравившегося рецепта")
    await message.answer(
        "\n".join([f"{i}. {j}\n" for i, j in zip(range(1, 6), best_recipes[message.from_user.id].name)]),
        reply_markup=markup5)
    logging.info("Time spent from receiving the photo till proposing 5 recipes: {}".format(time.time() - t0))
    state = dp.current_state(user=message.from_user.id)
    await state.set_state(TestStates.all()[2])


# team 2
@dp.message_handler(content_types=['text'], state=TestStates.TEST_STATE_2[0])
async def team_2_txt(message: types.Message) -> None:

    dct_keys = {f'{i}': i for i in range(1, 6)}
    dct_keys.update({'1️': 1, '2️': 2, '3️': 3, '4️⃣': 4, '5️⃣': 5})

    flag = dct_keys.get(message.text.strip())  ### Добавил .strip()
    if (best_recipes.get(message.from_user.id) is None) or (flag is None):
        await message.answer(
            "Немного не понял. Выбери число еще раз (кнопки ниже)")  # Поменял текст, закомментил 2 строки ниже
        return
    recipe = best_recipes[message.from_user.id].iloc[flag - 1]
    logging.info("User chose {}".format(recipe['name']))
    await message.answer("Название: {}\n\nСсылка: {}\n\nИнгредиенты:\n{}\n\nРецепт: {}".format(
        recipe['name'], recipe['recipe_link'], "\n".join(recipe['ingridients'][2:-2].split("', '")),
        "\n\n".join(recipe['instructions'][2:-2].split("', '"))),
        reply_markup=ReplyKeyboardRemove()
    )
    state = dp.current_state(user=message.from_user.id)
    await state.set_state(TestStates.all()[1])

@dp.message_handler()
async def state0(message: types.Message) -> None:
    await first_message(message)


@dp.message_handler(state=TestStates.TEST_STATE_0[0])
async def first_message(message: types.Message) -> None:
    state = dp.current_state(user=message.from_user.id)
    await state.set_state(TestStates.all()[1])

    await message.answer("Ты работаешь с проектом Dish-ID.\n" +
                         "Просто отправь мне фото блюда, и я подберу лучший рецепт",
                         reply_markup=ReplyKeyboardRemove())


if __name__ == '__main__':
    executor.start_polling(dp)
