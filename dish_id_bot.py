# To-do
# # Вывести в отдельный работающий пайплайн
# # Прикрутить распознование изображения
# # Прикрутить ранжирование
# # Прикрутить логирование


import aiohttp
import io
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
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.types import ReplyKeyboardRemove, \
    ReplyKeyboardMarkup, KeyboardButton, \
    InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.utils.helper import Helper, HelperMode, ListItem


N_BEST = 10


proxy_host = os.environ.get('PROXY', None)
proxy_credentials = os.environ.get('PROXY_CREDS', None)
if proxy_credentials:
    login, password = proxy_credentials.split(':')
    proxy_auth = aiohttp.BasicAuth(login=login, password=password)
else:
    proxy_auth = None

# bot = Bot(token=os.environ.get('TOKEN', None),
bot = Bot(token='1270289945:AAENLIacppChZzxlh9MjfdNmY2eWrPWb6dI',
          proxy=proxy_host, proxy_auth=proxy_auth)
dp = Dispatcher(bot, storage=MemoryStorage())


class TestStates(Helper):
    mode = HelperMode.snake_case

    TEST_STATE_0 = ListItem()
    TEST_STATE_1 = ListItem()
    TEST_STATE_2 = ListItem()

#
button0 = KeyboardButton('/help')
buttons = [KeyboardButton('1️⃣')]
buttons.append(KeyboardButton('2️⃣'))
buttons.append(KeyboardButton('3️⃣'))
buttons.append(KeyboardButton('4️⃣'))
buttons.append(KeyboardButton('5️⃣'))
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
    await message.answer("Hi, {}!\nI'm bot!\nChoose team number".format(message.from_user.first_name),
                         reply_markup=markup1)


@dp.message_handler(commands=['help'])
async def send_help(message: types.Message) -> None:
    state = dp.current_state(user=message.from_user.id)
    await state.set_state(TestStates.all()[0])
    await message.answer("Hi, wanna help?\n" +
                         "Just send your photo and wait for recipe generation\n" +
                         "Type /team to choose another team's project",
                         reply_markup=markup0)


@dp.message_handler(commands=['team'])
async def send_team(message: types.Message) -> None:
    await message.answer("Please, choose the team number", reply_markup=markup1)


@dp.message_handler(state=TestStates.all(), commands=['start'])
async def send_welcome_state(message: types.Message) -> None:
    await send_welcome(message)


@dp.message_handler(state=TestStates.all(), commands=['help'])
async def send_help_state(message: types.Message) -> None:
    await send_help(message)


@dp.message_handler(state=TestStates.all(), commands=['team'])
async def send_team_state(message: types.Message) -> None:
    await message.answer("Please, choose the team number", reply_markup=markup1)


# Основной функционал бота
# team 1
@dp.message_handler(state=TestStates.TEST_STATE_1[0])
async def team_1(message: types.Message, redirect: bool = False) -> None:
    # process image and match with recipies
    await message.answer("This is team 1")


# Предобработка изображения
def prepare_image(image):
    image = Image.open(image)
    image = image.resize((128, 128), Image.ANTIALIAS)
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format='PNG')
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr


# Здесь по фотке будем ингридиенты выводить
def get_ingridients(ready_image):
    return "яблоки масло мука яйца сахар молоко соль"


# Здесь будет ранжирование, пока просто берем 5 лучших
def ranking(df):
    return df.iloc[5:]


def match_recipes(ingridients):
    t0 = time.time()
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pkl.load(f)
    with open('train_tfidf.pkl', 'rb') as f:
        train_tfidf = pkl.load(f)
    with open('eda_povar.pkl', 'rb') as f:
        train = pkl.load(f)

    test_tfidf = vectorizer.transform([ingridients])
    result = np.array(test_tfidf).dot(np.array(train_tfidf.T))

    best_matches = np.argsort(np.array(result.todense()))[0][-N_BEST:]
    best_matches = pd.DataFrame(best_matches)
    best_matches.columns = ['idx']
    # print(best_matches)

    melt_matches = best_matches['idx'].apply(
        lambda x: train[['name', 'pure_ingridients', 'instructions', 'img_url', 'recipe_link']].iloc[x])

    short_list = ranking(melt_matches)
    print(time.time() - t0)
    return short_list


# team 2
@dp.message_handler(content_types=['text'], state=TestStates.TEST_STATE_2[0])
async def team_2_txt(message: types.Message) -> None:
    dct_keys = {f'{i}': i for i in range(1, 6)}
    dct_keys.update({'1️⃣': 1, '2️⃣': 2, '3️⃣': 3, '4️⃣': 4, '5️⃣': 5})
    flag = dct_keys.get(message.text) - 1
    if (best_recipes is None) or (not flag):
        await message.answer("Ошибка")
        return
    recipe = best_recipes.iloc[flag]
    await message.answer("Title: {}\n\nLink: {}\n\nIngredients: {}\n\nRecipe: {}".format(
        recipe['name'], recipe['recipe_link'], recipe['pure_ingridients'], recipe['instructions']
    ))



best_recipes = None
@dp.message_handler(content_types=['photo'], state=TestStates.TEST_STATE_2[0])
async def team_2(message: types.Message) -> None:
    # process image and match with recipes

    # problem - it takes several seconds to get into this point
    global best_recipes
    await message.answer("Please wait. I'm working")
    # print('hey')
    file_info = await bot.get_file(message.photo[-1].file_id)
    photo = await bot.download_file(file_info.file_path)
    # print(photo)
    try:
        ready_image = prepare_image(photo)
    except:
        await message.answer('photo problem {}'.format(file_info))
        return

    ingridients = get_ingridients(ready_image)

    best_recipes = match_recipes(ingridients)

    # print(best_recipes)
    await message.answer("Выберите номер наиболее понравившегося рецепта")
    await message.answer("\n".join([f"{i}. {j}\n" for i, j in zip(range(1, 6), best_recipes.name)]),
                         reply_markup=markup5)


    # try to get ingredients from the photo

    # let user change ingredients list

    # show the best recipes


# Обработка начальных сообщений
@dp.message_handler()
async def state0(message: types.Message) -> None:
    await first_message(message)


@dp.message_handler(state=TestStates.TEST_STATE_0[0])
async def first_message(message: types.Message) -> None:
    if message.text in ('1️⃣', '1'):
        team = 0
    elif message.text in ('2️⃣', '2'):
        team = 1
    else:
        await message.answer("Could not understand, please choose the team once more",
                             reply_markup=markup1)
        return
    state = dp.current_state(user=message.from_user.id)
    await state.set_state(TestStates.all()[team + 1])

    await message.answer("You're working with team {} project.\n".format(team + 1) +
                         "Just send me a photo, I'll send you the best recipes of the dish",
                         reply_markup=ReplyKeyboardRemove())


if __name__ == '__main__':
    executor.start_polling(dp)
