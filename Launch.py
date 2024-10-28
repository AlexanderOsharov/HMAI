import chardet
import re
from HistoricalTextClassifier import HistoricalTextClassifier
from TextProcessor import TextProcessor

classifier = HistoricalTextClassifier()
processor = TextProcessor(classifier)

text = ""
# Определяем кодировку файла
with open('./booksProcessor/books_1.txt', 'rb') as file:
    raw_data = file.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']

# Читаем файл с обработкой возможных ошибок
with open('./booksProcessor/books_1.txt', 'r', encoding=encoding, errors='replace') as file:
    text = file.read()

# Удаляем нежелательные символы
text = re.sub(r'[^a-zA-Zа-яА-Я0-9\s]', ' ', text)  # Заменяем все недопустимые символы на пробел
text = re.sub(r'[ \t]+', ' ', text)  # Заменяем подряд идущие пробелы на один пробел
text = re.sub(r'(\n)+', '\n', text)  # Удаление строк, состоящих из пробелов

# Обработка текста с помощью скользящего окна
long_text = """В 1757 г. для размещения Университета, основанного М.В. Ломоносовым в 1755 г., была приобретена усадьба князя Репнина на Моховой – современный участок Старого Университета. В середине 1780х гг."""
processed_text = processor.process_with_sliding_window(text)
print(processed_text)
