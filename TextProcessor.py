class TextProcessor:
    def __init__(self, classifier):
        self.classifier = classifier

    def process_long_text(self, long_text, chunk_size=512):
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