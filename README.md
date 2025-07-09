
# Pet Verification

## Описание проекта

Проект для верификации домашних животных с использованием машинного обучения и API.

## Структура проекта

```
pet_verification/
├── api/                # API сервер
├── dataset/           # Файл для генерации описаний для картинок
├── ml/                # Файл для обучения
└── requirements.txt   # Зависимости проекта
```


## Установка

### 1. Создание окружения

Создайте новое окружение conda с Python 3.10:

```bash
conda create -n pet_verification python=3.10
conda activate pet_verification
```


### 2. Установка зависимостей

Установите необходимые пакеты из requirements.txt:

```bash
pip install -r requirements.txt
```


## Запуск

### API сервер

Для запуска API сервера перейдите в папку `api` и выполните:

```bash
cd api
uvicorn main:app --host 0.0.0.0 --port 8502
```

API будет доступен по адресу: `http://localhost:8502`

## Использование

После запуска API сервера вы можете:

- Просмотреть документацию API по адресу: `http://localhost:8502/docs`


## Требования

- Python 3.10
- Conda
- Зависимости из requirements.txt

