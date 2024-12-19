import unittest
from PIL import Image
import torch
from main import classify_image, IDX_TO_CLASS, model, device


class TestBotWithRealImage(unittest.TestCase):
    def test_classify_real_image(self):
        # Загрузка реального изображения
        image_path = 'test_asset/image_1.jpg'  # Укажите путь к изображению
        image = Image.open(image_path)

        # Классификация изображения
        class_name, probability = classify_image(model, image)

        # Проверка результатов
        testasset = {'class_name': 'банан', 'probability': '100.00'}
        self.assertEqual(f'Результат классификации: {class_name} с вероятностью {probability:.2f}%', f'Результат классификации: {testasset["class_name"]} с вероятностью {testasset["probability"]}%')
        #self.assertGreaterEqual(probability, 0, "Вероятность меньше 0.")
        #self.assertLessEqual(probability, 100, "Вероятность больше 100.")


if __name__ == '__main__':
    unittest.main()