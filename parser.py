import pypdf
import re

class SyllabusParser:
    def extract_keywords(self, pdf_path):
        """Extracts key topics and dates from a university syllabus."""
        text = ""
        with open(pdf_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text()

        # Regex to find common academic headers
        topics = re.findall(r'(?:Week|Lesson|Unit)\s*\d+[:\-]\s*(.*)', text)
        return topics if topics else ["General Review"]