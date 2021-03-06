DISH ID 2020_final.pptx - презентация с описанием проекта Dish-ID (https://drive.google.com/file/d/1_Fg-ezBY7KnFsHnxjUJ8bHIi7i60mYGM/view?usp=sharing)

Итого: 1) данные с картинками и ингредиентами к ним
       2) данные с рецептами
       3) бот, который крутится на сервере
       4) pipeline от и до
       5) обученная модель по выделению ингредиентов на фотографии
       6) ноутбуки по обучению моделей
       7) скрипты с матчингом

data/ #папка с данными, которые удалось собрать - https://drive.google.com/drive/folders/1_3nfYMJH6fbME6c_bt68woR6Ia8hcRhQ?usp=sharing
	AllRecipes_images.zip  #данные с сайта allrecipes.com
                               #для обучения модели по выделению игредиентов по фотографии
                               #данные содержат изображения и ингредиенты

	recipes_final_povar_ru.pkl #рецепты с сайта povar.ru
                                   #датасет содержит данные о названиях, ингредиентах и рецептах блюд

	eda_all_recipes.csv #рецепты с сайта eda.ru
                                   #датасет содержит данные о названиях, ингредиентах и рецептах блюд
	
	recipes_all.csv #объединенные данные с eda.ru и povar.ru
                        #на этом датасете нужно тестировать gold

	eda_vectors.pkl #данные с eda.ru с заранее вычисленными эмбеддингами 
                        #word2vec, fasttext, elmo
                        
	gold.xlsx #золотой стандарт для проверки матчинга 
                  #если в колонке priority - 10, рецепт нерелевантен

	ingredients.json #словарь перевода ингредиентов с английского на русский
	dishes.json      #словарь перевода названий блюд с английского на русский

models/ #обученные модели по выделению ингредиентов на фотографии
        model_encoder_classifier_best.pth #модель, которая используется в боте 
                                          #лучшая из полученных моделей архитектуры encoder->classifier
					  # Можно найти по ссылке https://drive.google.com/file/d/1CKTwhkTrEJzU69wI11PdmX047p4ULCND/view?usp=sharing

bot_final_version/ #папка с ботом @dish_id_bot
                   #в папке также продублированы веса model_encoder_classifier_best.pth (по ссылке https://drive.google.com/file/d/1CKTwhkTrEJzU69wI11PdmX047p4ULCND/view?usp=sharing),
                   #чтобы было меньше трудностей с путями

pipeline/ #папка с pipeline от картинки до предсказания рецептов
	chefnet_dense.ipynb #ноутбук по обучению модели на данных AllRecipes_images.zip
        word_models.ipynb   #ноутбук с папйлайном по подбору рецептов
                            #можно использовать веса обученной модели models/model_encoder_classifier_best.pth

scripts/ #папка со скриптами
  	matching_povar.py  #алгоритм подбора рецептов по ингредиентам на данных recipes_final_povar_ru.pkl
	povar_scrapping.py #скрэппер сайта povar.ru



