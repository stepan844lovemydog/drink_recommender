# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk
import pandas as pd
import joblib
import os

# --- ЗАГРУЖАЕМ УЛУЧШЕННУЮ МОДЕЛЬ (100%) ---
current_dir = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(current_dir, 'model_drink_improved.pkl'))
encoders_questions = joblib.load(os.path.join(current_dir, 'encoders_questions_improved.pkl'))
le_target = joblib.load(os.path.join(current_dir, 'encoders_target_improved.pkl'))

# --- ВОПРОСЫ (как в опроснике) ---
questions = [
    'Что бы вы хотели прямо сейчас?',
    'Вы хотите чтобы напиток: ',
    'Кофеин для вас—это…',
    'Какой уровень сладости вам комфортен?',
    'Какое молоко вы предпочитаете?',
    'Объём(если предпочитаете конкретный-укажите)',
    'Вам больше нравится:',
    'Любите ли вы чай?',
    'Газированный напиток?',
    'Сколько льда вы предпочитаете в прохладительных напитках?',
    'Как вы относитесь к необычным сочетаниям?'
]

# --- ВАРИАНТЫ ОТВЕТОВ ---
variants = {
    'Что бы вы хотели прямо сейчас?': ['Освежающее', 'Расслабляющее', 'Бодрящее'],
    'Вы хотите чтобы напиток: ': ['Удивил', 'Остудил', 'Согрел'],
    'Кофеин для вас—это…': ['Не желательно', 'Желательно, но не принципиально', 'Обязательно'],
    'Какой уровень сладости вам комфортен?': ['Умеренно сладкий', 'Предпочитаю несладкие напитки или минимум сиропа', 'Сладкий'],
    'Какое молоко вы предпочитаете?': ['Любое', 'Обычное', 'Растительное (миндальное, овсяное, кокосовое, соевое и т.п.)', 'Без молока'],
    'Объём(если предпочитаете конкретный-укажите)': ['Большой', 'Большой (более 350 мл)', 'Средний (200-350 мл)', 'Маленький (до 200 мл)'],
    'Вам больше нравится:': ['Ягодные ноты', 'Фруктовые ноты', 'Сладкие специи', 'Цветочные ноты', 'Ореховые ноты'],
    'Любите ли вы чай?': ['Чёрный', 'Фруктовый', 'Молочный улун', 'Травяной (ромашка, мята и т.п.)', 'Зелёный', 'Без чая'],
    'Газированный напиток?': ['Совсем без газа', 'Слабогазированный', 'Да, сильно газированный'],
    'Сколько льда вы предпочитаете в прохладительных напитках?': ['Мало льда', 'Нет льда', 'Много льда'],
    'Как вы относитесь к необычным сочетаниям?': ['Люблю экспериментировать', 'Люблю экзотику', 'Скорее придерживаюсь классики, но иногда пробую новинки', 'Только классика']
}

# --- ФУНКЦИЯ ПРЕДСКАЗАНИЯ ---
def predict_drink():
    # Собираем ответы
    answers = {}
    for q in questions:
        answers[q] = combo_boxes[q].get()
    
    # Превращаем в DataFrame
    df_user = pd.DataFrame([answers])
    
    # Кодируем ответы в цифры (как при обучении)
    for col in df_user.columns:
        if col in encoders_questions:
            le = encoders_questions[col]
            # Если модель не видела такой ответ — ставим 0
            df_user[col] = df_user[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
    
    # Предсказываем
    pred_num = model.predict(df_user)[0]
    drink = le_target.inverse_transform([pred_num])[0]
    
    # Показываем результат
    result_label.config(text=f"🍹 {drink}", fg="#2E7D32")
    result_frame.config(bg="#E8F5E9")

# --- СОЗДАЁМ ОКНО ---
root = tk.Tk()
root.title("🍹 ИИ-помощник для подбора напитка")
root.geometry("750x780")
root.configure(bg="#F5F5F5")

# Заголовок
tk.Label(root, text="🍹 Подбери свой идеальный напиток с помощью нейросети!", 
         font=("Arial", 14, "bold"), bg="#F5F5F5", fg="#1B5E20").pack(pady=15)

tk.Label(root, text="(Точность модели: 100% на тестовых данных)", 
         font=("Arial", 10), bg="#F5F5F5", fg="#757575").pack(pady=5)

# Рамка для вопросов
main_frame = tk.Frame(root, bg="#FFFFFF", relief="groove", bd=2)
main_frame.pack(padx=20, pady=10, fill="both", expand=True)

# Создаём выпадающие списки
combo_boxes = {}
for i, q in enumerate(questions):
    frame = tk.Frame(main_frame, bg="#FFFFFF")
    frame.pack(fill="x", padx=15, pady=8)
    
    tk.Label(frame, text=q, font=("Arial", 10, "bold"), bg="#FFFFFF", width=40, anchor="w").pack(side="left")
    
    cb = ttk.Combobox(frame, values=variants[q], state="readonly", width=30, font=("Arial", 10))
    cb.pack(side="right", padx=5)
    cb.set(variants[q][0])  # Ставим первый вариант по умолчанию
    combo_boxes[q] = cb

# Кнопка предсказания
btn_frame = tk.Frame(root, bg="#F5F5F5")
btn_frame.pack(pady=20)

btn = tk.Button(btn_frame, text="🍹 ПОДОБРАТЬ НАПИТОК", command=predict_drink,
                font=("Arial", 13, "bold"), bg="#43A047", fg="white", 
                padx=30, pady=12, relief="raised", bd=3)
btn.pack()

# Рамка для результата
result_frame = tk.Frame(root, bg="#F5F5F5", relief="sunken", bd=2)
result_frame.pack(padx=20, pady=15, fill="x")

result_label = tk.Label(result_frame, text="🔄 Выберите параметры и нажмите кнопку", 
                        font=("Arial", 16, "bold"), bg="#F5F5F5", fg="#424242")
result_label.pack(pady=20)

# Запускаем окно
root.mainloop().se
