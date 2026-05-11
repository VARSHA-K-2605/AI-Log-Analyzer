FROM python:3.11

WORKDIR /app

COPY . .

RUN pip install pandas numpy scikit-learn nltk spacy flask streamlit

CMD ["python", "log_analyzer.py"]