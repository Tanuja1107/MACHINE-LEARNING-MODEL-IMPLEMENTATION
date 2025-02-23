import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 🟢 STEP 2: Load the dataset
file_path = r"C:\Users\TANUJA\OneDrive\Desktop\projects\spam_dataset.csv"

try:
    data = pd.read_csv(file_path)
    print("✅ Dataset loaded successfully!")
except FileNotFoundError:
    print("❌ Error: File not found. Check the file path.")
    exit()

# Display first few rows
print("📊 First 5 Rows of Data:\n", data.head())

# Rename columns if needed
expected_columns = {'message', 'label'}
if not expected_columns.issubset(data.columns):
    print("❌ Column names might be incorrect. Check your dataset.")
    exit()

# Remove missing values
data.dropna(inplace=True)

# Convert columns to string
data['message'] = data['message'].astype(str)
data['label'] = data['label'].astype(str)

# 🟢 STEP 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

# 🟢 STEP 4: Text vectorization
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 🟢 STEP 5: Train the model using Naive Bayes
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 🟢 STEP 6: Make predictions
y_pred = model.predict(X_test_vec)

# 🟢 STEP 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\n✅ Accuracy: {accuracy * 100:.2f}%")
print("\n📄 Classification Report:\n", report)

# 🟢 STEP 8: Test with a new message
new_message = ["Congratulations! You have won a free iPhone. Click here to claim."]
new_message_vec = vectorizer.transform(new_message)
prediction = model.predict(new_message_vec)

print(f"\n🔍 New Message Prediction: {prediction[0]}")
