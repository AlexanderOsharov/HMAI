class TextProcessor:
    def __init__(self, classifier):
        self.classifier = classifier

    def process_long_text(self, long_text, chunk_size=512):
        # Удаление лишних пробелов
        long_text = re.sub(r'[ \t]+', ' ', long_text).strip()  # Заменяем подряд идущие пробелы
        long_text = re.sub(r'(\n)+', '\n', long_text)  # Удаление строк, состоящих из пробелов

        chunks = [long_text[i:i + chunk_size] for i in range(0, len(long_text), chunk_size)]
        results = [self.classifier.predict(chunk) for chunk in chunks]

        # Объединение результатов
        combined_result = ""
        current_label = None
        for chunk, label in zip(chunks, results):
            if label != current_label:
                combined_result += "\n\n" + label.capitalize() + ":\n"
                current_label = label
            combined_result += chunk.strip() + "\n"

        return combined_result.strip()

    def process_with_sliding_window(self, text, window_size=150, overlap=50):
        start = 0
        text_chunks = []

        while start < len(text):
            end = min(start + window_size, len(text))
            text_chunks.append(text[start:end])
            start += window_size - overlap  # сдвигаем начало с учетом перекрытия

        results = [self.classifier.predict(chunk) for chunk in text_chunks]
        return self.combine_results(text_chunks, results)

    def combine_results(self, chunks, results):
        combined_result = ""
        current_label = None
        for chunk, label in zip(chunks, results):
            if label != current_label:
                combined_result += "\n\n" + label.capitalize() + ":\n"
                current_label = label
            combined_result += chunk.strip() + "\n"

        return combined_result.strip()