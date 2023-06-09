<div align="center">
<h1>
   <svg xmlns="http://www.w3.org/2000/svg" height="1em" viewBox="0 0 512 512"><defs><linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" style="stop-color:#4A00E0;"/><stop offset="100%" style="stop-color:#00B4D8;"/></linearGradient></defs><path d="M96 256H128V512H0V352H32V320H64V288H96V256zM512 352V512H384V256H416V288H448V320H480V352H512zM320 64H352V448H320V416H192V448H160V64H192V32H224V0H288V32H320V64zM288 128H224V192H288V128z" fill="url(#gradient)"/></svg>
   <span style="margin-left: .25em">Вояджер-108</span>
</h1>

<div style="display: flex; justify-content: center; align-items: begin; gap: 2.5em; margin-bottom: 2em">
    <div>
        <strong>Айхем Буабид</strong><br>a.bouabid@innopolis.university
    </div>
    <div>
        <strong>Александр Лобанов</strong><br>a.lobanov@innopolis.university
    </div>
    <div>
        <strong>Вячеслав Синий</strong><br>v.sinii@innopolis.university<br>v.siniy@tinkoff.ru
    </div>
</div>
</div>

<div align="center" style="display: flex; gap: 1em; justify-content: center; align-items: center; margin-bottom: 1em">
<a href="https://docs.voyager108.ru/" target="_blank">
  <button style="background: linear-gradient(45deg, #5e5ce6, #8b00ff);
                  color: #fff;
                  font-size: 16px;
                  border: none;
                  padding: 10px 20px;
                  border-radius: 5px;
                  cursor: pointer;">
    Документация
  </button>
</a>

<a href="https://voyager108.ru:7080" target="_blank">
  <button style="background: linear-gradient(-45deg, #5e5ce6, #8b00ff);
                  color: #fff;
                  font-size: 16px;
                  border: none;
                  padding: 10px 20px;
                  border-radius: 5px;
                  cursor: pointer;">
    Попробовать
  </button>
</a>

<a href="https://youtube.com" target="_blank">
  <button style="background: linear-gradient(135deg, #5e5ce6, #8b00ff);
                  color: #fff;
                  font-size: 16px;
                  border: none;
                  padding: 10px 20px;
                  border-radius: 5px;
                  cursor: pointer;">
    YouTube
  </button>
</a>
</div>

<div align="center" style="margin-bottom: 2em">
<hr style="width: 40%; height: 2px; background: linear-gradient(to right, rgba(94, 92, 230, 0.1), rgba(139, 0, 255, 0.8), rgba(94, 92, 230, 0.1)); margin: 0 auto; display: block;">

</div>

В этом репозитории содержится часть решения команды Вояджер-108 для конкурса Лидеры Цифровой Трансформации 2023 в рамках задачи ["Интерактивная платформа для мониторинга внутренней отделки квартиры"](https://leaders2023.innoagency.ru/task_9). Эта часть отвечает за вычислительный сервер и решение задачи компьютерного зрения и машинного обучения.  

# :green_book: Задача

Перед нами стояли следующие задачи:

1. Определить степень готовности помещений внутри строящегося здания. 
2. Определить приналежность помещений к той или иной категории (жилые помещения, санузлы и т.д.)
3. Используя полученную информацию, определить степень готовности здания в целом.   

Мы разбили задачу на несколько составляющих, и разработали следующие компоненты:

1. Компонент для распознавания элементов внутренней отделки помещений.
2. Сегментатор временного ряда.
3. Классификатор типов помещений.

# :bulb: Решение   

Подробно решение описано в [документации](https://docs.voyager108.ru/overview/reshenie) проекта. Ниже приведены лишь краткие описания компонентов.

## Компонент для распознавания элементов внутренней отделки помещений

Для распознавания объектов мы использовали данные, предоставленные организаторами, и модель YOLOv8. Мы разметили предоставленные данные, выделив все классы, которые запросили организаторы. Для стен, потолка и пола мы сделали по классу на каждый тип отделки. Изображение ниже прекрасно иллюстрирует качество работы обученной модели: она способна справляться даже со сложными случаями, где части одной стены находятся на разных стадиях готовности.

Было обучено два варианта YOLOv8:

<div align="center">
    
Базовая модель | mAP 50 | mAP 50-95 | Время на одно изображение (GPU)
--- | --- | --- | ---
`yolov8n` | 0.668 | 0.401 | 0.01 ms
`yolov8x` | 0.716 | 0.479 | 0.02 ms

</div>

# :rocket: Запуск

Запуск сервера производится через готовый Docker-образ. Следуйте следующим шагам для запуска:

1. Склонируйте репозиторий и перейдите в папку с проектом.
   
    ```bash
    git clone https://github.com/voyager108/ml 
    cd ml
    ```

2. Создайте `.env` файл и заполните его переменными окружения. 

    ```
    HF_AUTH_TOKEN=<ваш токен от HF>
    OBJECT_STORE_BUCKET=<название бакета на Yandex.Cloud>
    ```

3. Соберите и запустите Docker-контейнер.

    ```bash
    docker build -t voyager108/video .
    docker run -d --env-file .env -p "0.0.0.0:7080:8000" --name voyager-108-video voyager108/video bash scripts/serve.sh
    ```

4. (Опционально) Измените конфигурацию сервера по желанию (например, увеличьте количество воркеров). 

    ```
    docker run -d --env-file .env -p "0.0.0.0:7080:8000" --name voyager-108-video voyager108/video hypercorn apps.fastapi.app:app --bind 0.0.0.0:8000 -w 16
    ```

    или используйте `uvicorn`:

    ```
    docker run -d --env-file .env -p "0.0.0.0:7080:8000" --name voyager-108-video voyager108/video uvicorn apps.fastapi.app:app --host 0.0.0.0:8000 --workers 16
    ```

