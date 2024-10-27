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

# Используем регулярное выражение для замены нежелательных символов
# Например, заменяем все символы, которые не являются буквами, цифрами или пробелами
text = re.sub(r'[^a-zA-Zа-яА-Я0-9\s]', ' ', text)  # Заменяем все недопустимые символы на пробел

# Заменяем подряд идущие пробелы и табуляции
text = re.sub(r'[ \t]+', ' ', text)

# Удаление строк, состоящих из пробелов
text = re.sub(r'(\n)+', '\n', text)
print(text)
long_text = """В 1757 г. для размещения Университета, основанного М.В. Ломоносовым в 1755 г., была приобретена усадьба князя Репнина на Моховой – современный участок Старого Университета. В середине 1780х гг."""
processed_text = processor.process_long_text(text)
print(processed_text)