# Detection of people with pre-trained ssd 300 from nvidia
Детекция людей при помощи предобученной сети ssd300 от NVIDIA
## Запуск проекта

При создании проекта использовался python версии 3.9.0.

1. скопировать на локальный пк
2. создать виртуальное окружение virtual environment python -m venv .venv
3. активировать его
4. установить зависимости install requirements.txt
5. запустить файл main.py

Для запуска приложения FastAP запустить файл deploy.py или ввести в командной строке uvicorn deploy:app --reload

Если имеются веса для данной модели, то положить их в папку weight.
При запуске проекта, если нет заданных весов для модели, веса автоматически скачиваются по ссылке:
   https://api.ngc.nvidia.com/v2/models/nvidia/ssd_pyt_ckpt_amp/versions/20.06.0/files/nvidia_ssdpyt_amp_200703.pt

Если использовать автоматически скаченные веса, то модель сможет детектировать только людей без головы!

Результаты всех детекций хранятся в папке new_data

### Запуск через докер контейнер

    - Создать docker image, используя docker build  -t detection_server .
    - Запуск docker container, используяdocker run --rm -it -p 80:8000 --name detection detection_server
    - Перейти http://127.0.0.1:80/docs для просмотра API

    Если необходимо произвести детекцию нескольких файлов, в docker-compose.yml указать пусть к папке с данными.
    Запустить контейнер docker-compose  -f docker-compose.yml up


## Структура проекта
```
repository
    └── data                          # папка, в которую необходимо поместить данные
    │
    └── detection                     # папка с кодом модели
        └── config.py
            └── config.py             # файл с настройками
        │
        └── ssd
        │                                 # все необходимые скрипты для обучения ssd300
        |       └── SSD_Transformers.py   # необходимые data_transformers для train/test
        │       └── create_model.py       # функция загрузки весов и создания модели
        │       └── dataloader.py         # реализован класс, выполняющий чтение и преподработку всех изображений
        │       └── decode_results.py     # преобразует выход модели в bbx (используется при оценке качества модели, а также при детекции)
        │       └── Detection_model.py    # обертка модели в класс, содержащий доп. информацию, необходимую для deploy
        │       └── loader_coco.py        # data loader для COCO dataset
        │       └── loader_crowdhuman.py  # data loader для crowdhuman dataset
        │       └── model.py              # код слоев модели, а также функции потерь
        │       └── model_eval.py         # для оценки качества модели
        │       └── train_one_loop.py     # код обучения одной эпохи
        │       └── utils_ssd300.py       # реализация функций, необходимых для обработки сырого выхода модели
        └── train results
        │                                      # отчеты по обучению модели на crowdhuman dataset
        |       └── example prepare data.ipynb # ноутбук с примером подготовки данных
        |       └── example_train_model.ipynb  # ноутбук с примером обучения
        |       └── report.ipynb.ipynb         # отчет по обучению
        └── utils
        │       └── utils.py                   # Утилитры для модели, функции для нанесения bbx на изображения или видео
        │
        └── detect_images.py              # Функции для детектирования изображения
        └── detect_video.py               # Функция для детектирования видео
        └── req.py                        # Пример запросов requests

    └── weight                             # Веса модели
    │
    └── deploy.py                     # обертка модели в fast api
    └── main.py                       # Запуск детекции

```
Функции detect_ ...  .py возвращают не изображения, а bbx для каждого из изображений или видео в виде словаря. Ключи словаря - пути до файлов или имена файлов (при детекции из папки). Для наненсения bbx используются функции, расположенные в utils. При детекции в деплое, возвращает bbx для каждого из классов.
## Обучение модели
Модель обучалась на crowdhuman dataset. После обучения модель сопобна детектировать людей и их головы. Подробности обучения модели содеражатся в папке train_results
