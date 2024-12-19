import io
import logging
import os
import torch

from dotenv import load_dotenv
from PIL import Image
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from torchvision import transforms
from timm import create_model

# Настройки логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Маппинг индексов классов на названия фруктов и овощей
IDX_TO_CLASS = {
    0: 'яблоко', 1: 'банан', 2: 'свекла', 3: 'болгарский перец', 4: 'капуста',
    5: 'стручковый перец', 6: 'морковь', 7: 'цветная капуста', 8: 'перец чили',
    9: 'кукуруза', 10: 'огурец', 11: 'баклажан', 12: 'чеснок', 13: 'имбирь',
    14: 'виноград', 15: 'халапеньо', 16: 'киви', 17: 'лимон', 18: 'салат',
    19: 'манго', 20: 'лук', 21: 'апельсин', 22: 'паприка', 23: 'груша',
    24: 'горох', 25: 'ананас', 26: 'гранат', 27: 'картофель', 28: 'редис',
    29: 'соевые бобы', 30: 'шпинат', 31: 'сахарная кукуруза', 32: 'батат',
    33: 'помидор', 34: 'репа', 35: 'арбуз'
}

# Загрузка модели
device = torch.device('cpu')
model = create_model('vgg16', pretrained=False, num_classes=36)
model.load_state_dict(torch.load('vgg16_fruit_veg_classifier.pth', map_location=device))
model = model.to(device)
model.eval()

# Токен бота
load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')


# Функция для классификации изображения
def classify_image(model, image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(image)

    prediction_probs = torch.nn.functional.softmax(predictions, dim=1)
    predicted_class = torch.argmax(prediction_probs, dim=1).item()

    return IDX_TO_CLASS[predicted_class], prediction_probs[0, predicted_class].item() * 100


# Обработчик команды /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        'Привет! Отправь мне изображение фрукта или овоща, и я попробую его определить.'
    )


# Обработчик изображений
async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    file = await update.message.photo[-1].get_file()
    image_bytes = await file.download_as_bytearray()
    image = Image.open(io.BytesIO(image_bytes))

    # Классификация изображения
    class_name, probability = classify_image(model, image)
    await update.message.reply_text(f'С вероятностью {probability:.2f}% это {class_name}.')


# Запуск бота
if __name__ == '__main__':
    application = ApplicationBuilder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))
    application.run_polling()
