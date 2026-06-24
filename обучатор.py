# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# ---- ЗАГРУЗКА ФАЙЛА (он лежит на Рабочем столе) ----
desktop = os.path.join(os.path.expanduser('~'), 'Desktop')
file_path = os.path.join(desktop, 'готовый_файл_с_напитками_FINAL.csv')
df = pd.read_csv(file_path, sep=';', encoding='utf-8-sig')

print("✅ Данные загружены! Всего строк:", len(df))

# ---- ОТДЕЛЯЕМ ВОПРОСЫ (X) ОТ ОТВЕТОВ (y) ----
X = df.drop(columns=['Отметка времени', 'Напиток'])
y = df['Напиток']

# ---- КОДИРУЕМ ТЕКСТ В ЦИФРЫ ----
label_encoders = {}
for col in X.columns:
    le = LabelEncoder()
    X[col] = X[col].astype(str)
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

# ---- ДЕЛИМ НА ОБУЧАЮЩУЮ (80%) И ТЕСТОВУЮ (20%) ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ---- ОБУЧАЕМ КЛАССИФИКАТОР ----
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# ---- ПРОВЕРЯЕМ ТОЧНОСТЬ ----
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*50)
print(f"🎯 ТОЧНОСТЬ МОДЕЛИ: {accuracy * 100:.1f}%")
print("="*50)

print("\n📊 Отчёт по каждому напитку:")
print(classification_report(y_test, y_pred, target_names=le_target.classes_, zero_division=1))

# ---- СОХРАНЯЕМ МОДЕЛЬ НА РАБОЧИЙ СТОЛ ----
joblib.dump(model, os.path.join(desktop, 'model_drink.pkl'))
joblib.dump(label_encoders, os.path.join(desktop, 'encoders_questions.pkl'))
joblib.dump(le_target, os.path.join(desktop, 'encoders_target.pkl'))

print(f"\n💾 Модель сохранена на Рабочий стол:")
print(f"   - model_drink.pkl")
print(f"   - encoders_questions.pkl")
print(f"   - encoders_target.pkl")

# ---- ТЕСТ ДЛЯ ПРИМЕРА ----
print("\n🧪 ТЕСТ: Предсказание для первого человека из выборки:")
sample = X_test.iloc[0:1]
pred_num = model.predict(sample)[0]
pred_drink = le_target.inverse_transform([pred_num])[0]
true_drink = le_target.inverse_transform([y_test[0]])[0]
print(f"   Модель предсказала: {pred_drink}")
print(f"   Правильный ответ:  {true_drink}")

print("\n✅ ВСЁ ГОТОВО! Модель можно использовать.")
