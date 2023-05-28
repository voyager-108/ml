# ML
Этот репозиторий включает в себя наш подход к глубокому обучению для решения задачи обработки видео на хакатонах. 
## Структура хранилища
Основной движок нашего решения находится в папке apps. Последний далее разделен на следующие папки:
1. быстрый api: Код для нашего внутреннего сервера.
3. тесты: протестируйте общую структуру
3. утилита: используется для сегментации видео
4. nn: Основа нашего подхода к глубокому обучению:
  1. сегментация помещения: код для сегментации видео в соответствии с его временной шкалой, а также построения классификатора для прогнозирования типа помещения в данном кадре
  2. Yolov8: Код для обучения Yolov8 обнаруживать, классифицировать и отслеживать различные объекты в данном видео

# ML
This repository includes our Deep Learning approach to solve the Hackathon's video processing task.
## Repository Structure
The main engine of our solution lies in the apps folder. The latter is further divided into the following folders:
1. fastapi: The code for our Backend server.
3. tests: test the overall structure
3. utitlity: used for video segmentation
4. nn: The core of our Deep Learning approach:
  1. room segmentation: The code for segmenting a video according to its timeline as well as building a classifier for predicting the room type of a given frame
  2. Yolov8: The code for training Yolov8 to detect, classify and track different objects in a given video

