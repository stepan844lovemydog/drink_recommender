# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(current_dir, 'готовый_файл_с_напитками_расширенный.csv'), sep=';', encoding='utf-8-sig')

X = df.drop(columns=['Отметка времени', 'Напиток'])
y = df['Напиток']

# Кодируем
encoders = {}
for col in X.columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

le_target = LabelEncoder()
y_enc = le_target.fit_transform(y)

# Обучаем
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f"🎯 НОВАЯ ТОЧНОСТЬ: {acc*100:.1f}%")

# Сохраняем обновлённую модель
joblib.dump(model, os.path.join(current_dir, 'model_drink_improved.pkl'))
joblib.dump(encoders, os.path.join(current_dir, 'encoders_questions_improved.pkl'))
joblib.dump(le_target, os.path.join(current_dir, 'encoders_target_improved.pkl'))

print("✅ Обновлённая модель сохранена!")
