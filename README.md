# Telegram Bot для классификации фруктов и овощей

Этот бот для Telegram использует модель нейронной сети (VGG16) для классификации изображений фруктов и овощей. Он принимает изображения через чат и возвращает название распознанного объекта вместе с вероятностью.

## Установка и настройка

1. **Клонируйте репозиторий**
   ```bash
   git clone https://github.com/IvanPodyukov/fruits_vegetables_classifier_telegram_bot.git
2. **Создайте файл .env в корне проекта и добавьте в него токен вашего Telegram бота**
   ```makefile
   TELEGRAM_TOKEN=YOUR_TOKEN
   ```
   Замените `YOUR_TOKEN` на токен, полученный от BotFather в Telegram.
3. **Установите зависимости из файла requirements.txt**
   ```bash
   pip install -r requirements.txt
4. **[Скачайте](https://drive.google.com/file/d/1nyBiKV5cFGRjLqh6JAmj9_pRC2TRf0x-) веса модели и сохраните их в корневую директорию проекта**

   Убедитесь, что файл называется `vgg16_fruit_veg_classifier.pth`, или измените путь в коде, если используете другое имя файла.
5. **Запустите бота**
   ```bash
   python main.py