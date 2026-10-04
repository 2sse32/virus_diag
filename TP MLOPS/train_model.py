import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 1. Chargement des données
df = pd.read_csv('insurance.csv')

# 2. Séparation Features / Target
X = df.drop('charges', axis=1)
y = df['charges']

# Définition des colonnes
num_features = ['age', 'bmi', 'children']
cat_features = ['sex', 'smoker', 'region']

# 3. Pipelines de Preprocessing
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ]
)

# Séparation Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Comparaison des Modèles ML (Étape 5)
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

best_model = None
best_r2 = -float('inf')
best_model_name = ""

print("=== ÉVALUATION DES MODÈLES ===")
for name, model in models.items():
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\nModel: {name}")
    print(f"R2 Score: {r2:.4f}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    
    if r2 > best_r2:
        best_r2 = r2
        best_model = clf
        best_model_name = name

print(f"\n--> Le meilleur modèle sélectionné est : {best_model_name} (R2 = {best_r2:.4f})")

# 5. Sauvegarde du modèle dans model.pkl
with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("Modèle sauvegardé avec succès dans 'model.pkl'.")