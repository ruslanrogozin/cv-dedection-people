# Detection of people with pre-trained ssd 300 from nvidia
Детекция людей при помощи предобученной сети ssd300 от NVIDIA
## Запуск проекта

При создании проекта использовался python версии 3.9.0.

1. скопировать на локальный пк
2. создать виртуальное окружение virtual environment python -m venv .venv
3. активировать его
4. установить зависимости install requirements.txt
5. заупстить файл main.py

Если имеются веса для данной модели, то положить их в папку weight в формате .pt.
При запуске проекта, если нет заданных весов для модели, веса автоматически скачиваются по ссылке:
   https://api.ngc.nvidia.com/v2/models/nvidia/ssd_pyt_ckpt_amp/versions/20.06.0/files/nvidia_ssdpyt_amp_200703.pt

Результаты всех детекций хранятся в папке new_data



## Структура проекта
```
repository
    └── config.py
            └── config.py                                # файл с настройками

    └── data                                             # папка, в которую необходимо поместить данные
    │
    └── ssd 
    │                                                    # все необходимые скрипты для обучения ssd300
    |       └── SSD_Transformers.py                      # необходимые data_transformers для train/test
    │       └── create_model.py                          # функция загрузки весов и создания модели
    │       └── dataloader.py                            # реализован класс, выполняющий чтение и преподработку всех изображений
    │       └── decode_results.py                        # преобразует выход модели в bbx (используется при оценке качества модели, а также при детекции)
    │       └── loader_coco.py                           # data loader для COCO dataset
    │       └── loader_crowdhuman.py                     # data loader для crowdhuman dataset
    │       └── model.py                                 # код слоев модели, а также функции потерь
    │       └── model_eval.py                            # для оценки качества модели
    │       └── train_one_loop.py                        # код обучения одной эпохи
    │       └── utils_ssd300.py                          # реализация функций, необходимых для обработки сырого выхода модели
    └── train results
    │                                                    # отчеты по обучению модели на crowdhuman dataset
    |       └── example prepare data.ipynb               # ноутбук с примером подготовки данных
    |       └── example_train_model.ipynb                # ноутбук с примером обучения
    |       └── report.ipynb.ipynb                       # отчет по обучению
    └── utils
    │       └── utils.py                                # Утилитры для модели
    └── weight                                          # Веса модели

    │   
    └── detect_batch_image_cv2.py                       # детекция изображений, прочитанных cv2 (используется в детекции видео)            
    └── detect_images_from_folder.py                    # Функция для детектирования всех изображений из папки data
    └── detect_video.py                                 # Функция для детектирования всех видео из папки data
    └── main.py                                         # Запуск детекции
```
## Обучение модели
Модель обучалась на crowdhuman dataset. Подробности об обучения модели содеражатся в папке train_results
