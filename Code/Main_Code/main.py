import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report


# Load the dataset
df = pd.read_csv('C:/Users/parth/PycharmProjects/Guvi_Mentor/Project_2/Dataset/FinalBalancedDataset.csv/FinalBalancedDataset.csv')

# Convert text to Bag of Words
bow_vectorizer = CountVectorizer()
X_bow = bow_vectorizer.fit_transform(df['tweet'])

# Convert text to TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(df['tweet'])

# Split the dataset into training and testing sets
y = df['Toxicity']
X_train_bow, X_test_bow, y_train, y_test = train_test_split(X_bow, y, test_size=0.2, random_state=42)
X_train_tfidf, X_test_tfidf, _, _ = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Define a function to train and evaluate models
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"Model: {model.__class__.__name__}")
    print(classification_report(y_test, predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    # Calculate ROC-AUC
    if hasattr(model, "predict_proba"):
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        print(f"ROC-AUC: {roc_auc}")
    else:
        print("ROC-AUC: Not applicable for this model")
    print("\n")

# Initialize models
models = [
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    MultinomialNB(),
    KNeighborsClassifier(),
    SVC(probability=True)  # Enable probability for ROC-AUC
]

# Train and evaluate models using Bag of Words
print("Evaluating models with Bag of Words")
for model in models:
    train_and_evaluate_model(model, X_train_bow, X_test_bow, y_train, y_test)

# Train and evaluate models using TF-IDF
print("Evaluating models with TF-IDF")
for model in models:
    train_and_evaluate_model(model, X_train_tfidf, X_test_tfidf, y_train, y_test)