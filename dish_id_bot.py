import aiohttp
from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.types import ReplyKeyboardRemove, \
    ReplyKeyboardMarkup, KeyboardButton, \
    InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.utils.helper import Helper, HelperMode, ListItem
import os
import typing as tp


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


# team 2
@dp.message_handler(state=TestStates.TEST_STATE_2[0])
async def team_2(message: types.Message) -> None:
    # process image and match with recipies
    await message.answer("This is team 2")


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
    await message.answer("You're working with team {} project.\n".format(team + 1) +
                         "Just send me a photo, I'll send you the best recipes of the dish",
                         reply_markup=ReplyKeyboardRemove())
    state = dp.current_state(user=message.from_user.id)
    await state.set_state(TestStates.all()[team + 1])


if __name__ == '__main__':
    executor.start_polling(dp)
