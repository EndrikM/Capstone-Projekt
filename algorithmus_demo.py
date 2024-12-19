import json
import os
from bs4 import BeautifulSoup
import requests


class DataManager:
    def __init__(self) -> None:
        self._file_path = "data.json"

        if not os.path.exists(self._file_path):
            self._store({})

        self._data: dict[str, tuple[str, str]] = self._load()

    def get_data(self) -> dict[str, tuple[str, str]]:
        return self._data.copy()

    def add_all_urls(self, urls: dict[str, tuple[str, str]], update_existing: bool) -> None:
        for name, (label, url) in urls.items():
            if update_existing or (name not in self._data):
                self.add_url(name, label, url)

    def add_url(self, name: str, label: str, url: str) -> None:
        text = self._get_text_from_url(url)
        text = text.replace("\xa0", " ")
        text = text.replace("\t", " ")
        text = text.strip()
        if len(text) == 0:
            print(
                f"The text for the URL {url} is empty or could not be loaded! Please check the URL.")
            return

        self._data[name] = (label, text)
        self._store(self._data)

    def _get_text_from_url(self, url: str) -> str:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            paragraphs = soup.find_all('p')
            text = ' '.join(p.get_text() for p in paragraphs)
            return text
        except Exception as e:
            print(f"Error retrieving the URL {url}: {e}")
            return ""

    def _store(self, data: dict[str, tuple[str, str]]) -> None:
        with open(self._file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False)

    def _load(self) -> dict[str, tuple[str, str]]:
        with open(self._file_path, "r", encoding="utf-8") as file:
            return json.load(file)

data_manager = DataManager()

urls = {
    "whitehouse": ("Legal & Regulatory aspects of AI", "https://www.whitehouse.gov/briefing-room/presidential-actions/2023/10/30/executive-order-on-the-safe-secure-and-trustworthy-development-and-use-of-artificial-intelligence/?utm_source=chatgpt.com"),
    "kpmg": ("Legal & Regulatory aspects of AI", "https://kpmg.com/us/en/articles/2023/landmark-actions-coming-the-ai-act-and-growing-us-regulations-reg-alert.html"),
    "technologyreview": ("Legal & Regulatory aspects of AI", "https://www.technologyreview.com/2024/02/15/1087815/responsible-technology-use-in-the-ai-age/#:~:text=AI%20presents%20distinct%20social%20and,singular%20opportunity%20for%20responsible%20adoption.&text=The%20sudden%20appearance%20of%20application,challenging%20social%20and%20ethical%20questions."),
    "pwc": ("Legal & Regulatory aspects of AI", "https://www.pwc.ch/en/insights/regulation/ai-act-demystified.html"),
    "bbc": ("Legal & Regulatory aspects of AI", "https://www.bbc.com/news/technology-68546450"),
    "european comission": ("Legal & Regulatory aspects of AI", "https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai"),
    "pecan": ("Next-Generation Regression Models and Data Calculation Algorithms", "https://www.pecan.ai/blog/predictive-modeling/"),
    "databricks": ("Next-Generation Regression Models and Data Calculation Algorithms", "https://www.databricks.com/blog/introduction-time-series-forecasting-generative-ai"),
    "bostoninstituteofanalytics": ("Next-Generation Regression Models and Data Calculation Algorithms", "https://bostoninstituteofanalytics.org/blog/the-role-of-ai-in-predictive-analytics-how-machine-learning-is-transforming-forecasting/"),
    "discoverdatascience": ("Next-Generation Regression Models and Data Calculation Algorithms", "https://www.discoverdatascience.org/articles/breaking-down-the-top-data-science-algorithms-methods/"),
    "pickl": ("Next-Generation Regression Models and Data Calculation Algorithms", "https://www.pickl.ai/blog/regression-in-machine-learning-types-examples/"),
    "nature": ("Next-Generation Regression Models and Data Calculation Algorithms", "https://www.nature.com/articles/s41598-024-55243-x"),
    "ibm": ("Large Language Models (LLMs) and Generative AI", "https://www.ibm.com/blog/generative-ai-vs-predictive-ai-whats-the-difference/"),
    "computerworld": ("Large Language Models (LLMs) and Generative AI", "https://www.computerworld.com/article/1627101/what-are-large-language-models-and-how-are-they-used-in-generative-ai.html"),
    "toloka": ("Large Language Models (LLMs) and Generative AI", "https://toloka.ai/blog/difference-between-ai-ml-llm-and-generative-ai/"),
    "ieeexplore": ("Large Language Models (LLMs) and Generative AI", "https://ieeexplore.ieee.org/abstract/document/10669603"),
    "mckinsey": ("Large Language Models (LLMs) and Generative AI", "https://www.mckinsey.com/capabilities/mckinsey-digital/our-insights/the-economic-potential-of-generative-ai-the-next-productivity-frontier"),
    "appian": ("Large Language Models (LLMs) and Generative AI", "https://appian.com/blog/acp/process-automation/generative-ai-vs-large-language-models"),
    "leewayhertz": ("Predictive Analytics and Forecasting", "https://www.leewayhertz.com/ai-for-predictive-analytics/"),
    "mckinsey": ("Predictive Analytics and Forecasting", "https://www.mckinsey.com/capabilities/operations/our-insights/ai-driven-operations-forecasting-in-data-light-environments"),
    "forbes": ("Predictive Analytics and Forecasting", "https://www.forbes.com/sites/amazonwebservices/2021/12/03/predicting-the-future-of-demand-how-amazon-is-reinventing-forecasting-with-machine-learning/"),
    "dlabi": ("Predictive Analytics and Forecasting", "https://dlabi.org/index.php/journal/article/view/115"),
    "geniusee": ("Predictive Analytics and Forecasting", "https://geniusee.com/single-blog/ai-and-predictive-analytics"),
    "SAP": ("Predictive Analytics and Forecasting", "https://learning.sap.com/learning-journeys/leveraging-sap-analytics-cloud-functionality-for-enterprise-planning/forecasting-with-predictive-analytics_de7bdcb3-f8eb-4dca-bac1-00ebef2b7bee"),
    "thomsonreuters": ("Ethics and Responsible Usage", "https://legal.thomsonreuters.com/blog/how-to-responsibly-use-ai-to-address-ethical-and-risk-challenges/"),
    "ssir": ("Ethics and Responsible Usage", "https://ssir.org/articles/entry/8_steps_nonprofits_can_take_to_adopt_ai_responsibly"),
    "atlassian": ("Ethics and Responsible Usage", "https://www.atlassian.com/blog/artificial-intelligence/responsible-ai"),
    "springer-s10676-021-09606-x": ("Ethics and Responsible Usage", "https://link.springer.com/article/10.1007/s10676-021-09606-x"),
    "springer-s43681-023-00330-4": ("Ethics and Responsible Usage", "https://link.springer.com/article/10.1007/s43681-023-00330-4"),
    "forbes": ("Ethics and Responsible Usage", "https://www.forbes.com/councils/forbestechcouncil/2024/11/08/navigating-the-ethics-of-ai-is-it-fair-and-responsible-enough-to-use/")
}

data_manager.add_all_urls(urls, update_existing=False)

data = data_manager.get_data()

import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd


# Function to extract Text from Websites

def get_text_from_url(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join(p.get_text() for p in paragraphs)
        return text
    except Exception as e:
        print(f"Error retrieving the URL {url}: {e}")
        return ""



# Load data from data.json file
data_manager = DataManager()
data = data_manager.get_data()
labels, texts = zip(*(data.values()))

# Create DataFrame
df = pd.DataFrame({'text': texts, 'label': labels})

# Transform Text into Features (TF-IDF)
vectorizer = TfidfVectorizer(
    max_features=5000, ngram_range=(1, 2), stop_words='english')
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Train-Test-Split
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
except ValueError:
    print("Warning: Not enough data for stratified split. Perform a normal split instead.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)


# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_test))

# Cross-Validation
print("Do Cross-Validation:")
scores = cross_val_score(model, X, y, cv=3)
print("Cross-Validation Scores:", scores)
print("Average Accuracy:", scores.mean())

# Example: Analyze a new URL
new_url = "https://www.nvidia.com/en-us/glossary/large-language-models/"
new_text = get_text_from_url(new_url)
if new_text:
    new_text_vectorized = vectorizer.transform([new_text])
    predicted_trend = model.predict(new_text_vectorized)
    print(f"The predicted trend for the URL is: {predicted_trend[0]}")

