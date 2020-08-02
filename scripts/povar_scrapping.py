import pandas as pd
import pickle as pkl
from selenium.webdriver import Chrome
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions
# import wget

#####################
##### Поставить путь до драйвера вашего браузера
driver = Chrome(executable_path="/path/to/chromedriver.exe")
option = webdriver.ChromeOptions()
option.add_argument(" — incognito")

BASE = 'https://povar.ru/list/'

#####################
##### Поставить срез категорий (между 0 и 271), которые будут скрепиться
### У меня 1 категория = 3-4 минуты скрепинга (35-40 рецептов)
##### Поставить путь до директории сохранения данных
MIN_CATEGS = 150
MAX_CATEGS = 271
DIR_NAME = '/path/to/save/data'
# 271 - это количество категорий, которые нужно заскреппить
# NUM_CATEGS = [i.text for i in categories].index('Соленое') + 1


def main():
    inds = []
    titles = []
    ingridient_lists = []
    img_urls = []
    step_lists = []
    links = []

    driver.get(BASE)
    # categories = driver.find_elements_by_css_selector(".ingredientItem div a")
    # print([i.text for i in categories])
    for categ_num in range(MIN_CATEGS, MAX_CATEGS):
        # Чекаем, есть ли вообще ссылки на категории
        try:
            WebDriverWait(driver, 8).until(
                expected_conditions.presence_of_element_located(
                (By.CSS_SELECTOR, ".ingredientItemH2")))
        except:
            print('no-category')
            continue

        # Переходим по найденной ссылке на категорию
        try:
            driver.find_elements_by_css_selector('.ingredientItem div a')[categ_num].click()
        except:
            continue

        # Пытаемся не упасть в ситуации, когда рецептов меньше 40 (не целая страница)
        try:
            ttl_recipes = WebDriverWait(driver, 4).until(
                expected_conditions.presence_of_element_located(
                (By.CSS_SELECTOR, ".total")))
            num_recipes = int(ttl_recipes.text.split()[-1])
        except:
            driver.get(BASE)
            print('less than 40 recipes')
            continue
        print(f"Рецептов в категории: {num_recipes}")

        categ_url = driver.current_url
        for page_num in range(2, min(num_recipes // 40 + 1, 11)):
            try:
                driver.get(categ_url + str(page_num))
            except:
                print('page is not clicked')
                driver.get(categ_url)
                continue
        # На одной странице максимум 40 рецептов
            for recipe_num in range(40):

                # Ждем открытия страницы и переходим на страничку с рецептом
                try:
                    WebDriverWait(driver, 4).until(
                        expected_conditions.presence_of_element_located(
                            (By.CSS_SELECTOR, ".listRecipieTitle")))
                    driver.find_elements_by_css_selector(".listRecipieTitle")[recipe_num].click()
                except:
                    print('skipped_1')
                    # Если вдруг оказались на неправильной странице, надо попробовать обратно зайти, и потом уже забить
                    driver.get(categ_url + str(page_num))
                    continue

                # Качаем название и ингридиенты
                try:
                    title = WebDriverWait(driver, 4).until(
                        expected_conditions.presence_of_element_located(
                            (By.CSS_SELECTOR, ".detailed"))).text.strip()
                except:
                    title = None
                try:
                    ingridients = [elem.text.strip() for elem
                                   in driver.find_elements_by_css_selector('.detailed_ingredients li')]
                except:
                    ingridients = None
                try:
                    steps = [elem.text.strip() for elem in
                             driver.find_elements_by_css_selector(".detailed_step_description_big")]
                except:
                    steps = None
                cur_url = driver.current_url


                # Качаем фотку
                try:
                    img_src = WebDriverWait(driver, 5).until(
                        expected_conditions.presence_of_element_located(
                        (By.CSS_SELECTOR, ".bigImgBox img"))).get_attribute('src')
                except:
                    img_src = None

                # Тут еще можно саму фотку скачать, но это вроде не особо нужно
                #     print(img_src)
                #     try:
                #         wget.download(img_src, DIR_NAME + f'{categ_num}_{recipe_num}.png')
                #     except:
                #         driver.refresh()
                #         print('refreshing')
                #         img_src = WebDriverWait(driver, 2).until(
                #             expected_conditions.presence_of_element_located(
                #                 (By.CSS_SELECTOR, ".bigImgBox img"))).get_attribute('src')
                #         print('img_src')
                #         try:
                #             wget.download(img_src, DIR_NAME + f'{categ_num}_{recipe_num}.png')
                #         except:
                #             pass

                # Сохраняем в базу данные
                inds.append(f'{categ_num}_{recipe_num + 40 * page_num}')
                titles.append(title)
                img_urls.append(img_src)
                ingridient_lists.append(tuple(ingridients))
                step_lists.append(tuple(steps))
                links.append(cur_url)

                driver.get(categ_url + str(page_num))

        # Делаем бэкап, вдруг что пойдет не так
        df = pd.DataFrame([inds, titles, ingridient_lists, img_urls, step_lists, links]).transpose().set_index(0)
        if categ_num == 99:
            with open(DIR_NAME + 'backup_100.pkl', 'wb') as f:
                pkl.dump(df, f, protocol=4)
        else:
            with open(DIR_NAME + 'backup.pkl', 'wb') as f:
                pkl.dump(df, f, protocol=4)
        driver.get(BASE)

    # Закрываем драйвер и фиксируем резы в .pkl
    driver.close()
    df = pd.DataFrame([inds, titles, ingridient_lists, img_urls, step_lists, links]).transpose().set_index(0)
    with open(DIR_NAME + 'backup.pkl', 'wb') as f:
        pkl.dump(df, f, protocol=4)


if __name__ == '__main__':
    main()
    # 01:19 - начало скрепинга
    with open(DIR_NAME + 'backup.pkl', 'rb') as f:
        df = pkl.load(f)
    print(df.shape)