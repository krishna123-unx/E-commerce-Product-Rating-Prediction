import pandas as pd

df = pd.read_csv("data/train.csv")

print(df.head())
print(df.columns)
print("Average Rating:", df['Rating'].mean())
print("Average Price:", df['price1'].mean())
print("\nProducts per Category:")
print(df['maincateg'].value_counts())
df['discount'] = df['actprice1'] - df['price1']
print("\nTop Discounts:")
print(df[['title', 'discount']].sort_values(by='discount', ascending=False).head())
import matplotlib.pyplot as plt

df['Rating'].hist()
plt.title("Rating Distribution")
plt.show()
plt.scatter(df['price1'], df['Rating'])
plt.xlabel("Price")
plt.ylabel("Rating")
plt.title("Price vs Rating")
plt.show()
print("\nFulfilled vs Rating:")
print(df.groupby('fulfilled1')['Rating'].mean())
# ================= ML PART =================#

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# 1. Sabse pehle 'data' variable create karo
data = df.copy() 

# 2. Cleaning Offer %
# Hum .astype(str) isliye karte hain taaki numeric conversion mein error na aaye
data['Offer %'] = pd.to_numeric(data['Offer %'].astype(str).str.replace('%', ''), errors='coerce')
data['Offer %'] = data['Offer %'].fillna(0)

# 3. Encoding (Dummies)
# Categorical columns ko numeric flags mein badalna
data_ml = pd.get_dummies(data, columns=['maincateg', 'platform'], drop_first=True)

# 4. Final Clean: Remove any row that has a NaN value
data_ml = data_ml.dropna()

# 5. Features & Target
# Hum 'id', 'title' aur 'Rating' (target) ko features se hata rahe hain
X = data_ml.drop(['id', 'title', 'Rating', 'discount'], axis=1, errors='ignore')
y = data_ml['Rating']

# 6. Split data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)
lr_pred = model.predict(X_test)
print("\nLinear Regression MAE:", mean_absolute_error(y_test, lr_pred))

# 8. Decision Tree Model
tree_model = DecisionTreeRegressor(max_depth=5, random_state=42)
tree_model.fit(X_train, y_train)
tree_pred = tree_model.predict(X_test)
print("Decision Tree MAE:", mean_absolute_error(y_test, tree_pred))

# 9. Feature Importance Visualization
importance = tree_model.feature_importances_
features = X.columns
plt.figure(figsize=(10,6))
plt.barh(features, importance, color='teal')
plt.title("Importance of Features in Predicting Ratings")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()