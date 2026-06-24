# -*- coding: utf-8 -*-
import pandas as pd
import random
import os

# Загружаем твой файл
current_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(current_dir, 'готовый_файл_с_напитками_FINAL.csv'), sep=';', encoding='utf-8-sig')

print(f"✅ Загружено {len(df)} реальных строк")

# --- Варианты для замены (логически похожие) ---
replacements = {
    'Что бы вы хотели прямо сейчас?': {
        'Бодрящее': ['Освежающее', 'Бодрящее'],
        'Расслабляющее': ['Уютное', 'Спокойное'],
        'Освежающее': ['Бодрящее', 'Освежающее']
    },
    'Любите ли вы чай?': {
        'Чёрный': ['Чёрный', 'Зелёный', 'Молочный улун'],
        'Зелёный': ['Зелёный', 'Травяной', 'Чёрный'],
        'Травяной (ромашка, мята и т.п.)': ['Травяной', 'Мятный', 'Чёрный'],
        'Молочный улун': ['Молочный улун', 'Чёрный', 'Зелёный'],
        'Без чая': ['Без чая', 'Травяной']
    },
    'Газированный напиток?': {
        'Да, сильно газированный': ['Да, сильно газированный', 'Слабогазированный'],
        'Слабогазированный': ['Слабогазированный', 'Да, сильно газированный'],
        'Совсем без газа': ['Совсем без газа', 'Слабогазированный']
    },
    'Какой уровень сладости вам комфортен?': {
        'Умеренно сладкий': ['Умеренно сладкий', 'Сладкий'],
        'Предпочитаю несладкие напитки или минимум сиропа': ['Предпочитаю несладкие', 'Умеренно сладкий'],
        'Сладкий': ['Сладкий', 'Умеренно сладкий']
    }
}

# --- Функция для генерации нового варианта ---
def generate_synthetic(row, num_copies=2):
    new_rows = []
    for _ in range(num_copies):
        new_row = row.to_dict()
        # Меняем 1-2 случайных вопроса
        for col in replacements:
            if col in new_row and new_row[col] in replacements[col]:
                new_row[col] = random.choice(replacements[col][new_row[col]])
        new_rows.append(new_row)
    return new_rows

# --- Генерируем синтетику (по 3 копии с каждой реальной строки) ---
synthetic_data = []
for _, row in df.iterrows():
    synthetic_data.extend(generate_synthetic(row, num_copies=3))

df_synthetic = pd.DataFrame(synthetic_data)

# --- Объединяем с реальными данными ---
df_final = pd.concat([df, df_synthetic], ignore_index=True)

# --- Сохраняем ---
df_final.to_csv(os.path.join(current_dir, 'готовый_файл_с_напитками_расширенный.csv'), sep=';', index=False, encoding='utf-8-sig')

print(f"✅ СОЗДАНО! Теперь {len(df_final)} строк (реальные + синтетика)")
print(f"   Реальные: {len(df)}")
print(f"   Синтетика: {len(df_synthetic)}")