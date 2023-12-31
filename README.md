# Тестовое задание

Необходимо было смешать два изображения, чтобы придать первому изображению черты второго. Для этого мы воспользуемся image fuse и моделью с открытым кодом [Kandinsky 2.2](https://github.com/ai-forever/Kandinsky-2).

## Метод решения

Будем объединять два изображения с помощью `mix_images`, но учитывая, что нам надо отдать предпочтения первой картинке из двух, то веса смешивания будут не равными, а зафиксированными эмпирически 0.7 и 0.3 соответственно у первой и второй картинки.

## Установка зависимостей

Для нормального использования решения необходимо:
1. [Установить](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) `git`.
2. [Установить](https://pip.pypa.io/en/stable/installation/) `pip`.
3. Установить зависимости из `requirements.txt` - `pip install -r requirements.txt`.

Если же необходимо изолировать запуск в Docker, то необходимый контейнер можно найти в каталоге NVIDIA NGC [тут](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch).

## Как запустить

Для запуска смешивания есть файл `run.py`, который запускается следующим образом (выделенное в квадратные скобки - опционально и можно опустить):
```bash
python run.py [--device cpu] [--flash_attention true] [--output_name mix.png] [--output_width 512] [--output_height 512] [--random_state 42] img1.jpg img2.jpg
```
Выходное изображение будет записано в файл `output_name` параметр, который по умолчанию `mix.png`.

## Примеры работы программы

Ниже приведены примеры работы программы:

1. Исходные изображения:

| Левое | Правое |
| -- | --- |
| ![Left](imgs/image1.jpeg "Left") | ![Right](imgs/image2.jpeg "Right") |

Результат:

![First mix](imgs/mix1.png "First mix")

2. Исходные изображения:

| Левое | Правое |
| -- | --- |
| ![Left](imgs/image3.jpeg "Left") | ![Right](imgs/image4.jpeg "Right") |

Результат:

![Second mix](imgs/mix2.png "Second mix")
