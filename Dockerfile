FROM python:3.10-slim

RUN apt-get update && apt-get install -y git curl python3-venv

WORKDIR /app

RUN git clone https://github.com/ticpragma/fruits_vegetables_classifier_telegram_bot/ .

RUN python3 -m venv venv

RUN ./venv/bin/pip install --no-cache-dir -r requirements.txt

RUN curl -L -o vgg16_fruit_veg_classifier.pth 'https://github.com/ticpragma/fruits_vegetables_classifier_telegram_bot/releases/download/download/vgg16_fruit_veg_classifier.pth'

ENV PATH="/app/venv/bin:$PATH"

CMD ["python", "main.py"]