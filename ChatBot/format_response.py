import re
def clean_response(text):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = text.strip()
    text = re.sub(r'\[\s*(.*?)\s*\]', r'$$\1$$', text, flags=re.DOTALL)
    return text
