# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk
import pandas as pd
import joblib
import os

# Загружаем модель
current_dir = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(current_dir, 'model_drink.pkl'))
encoders_questions = joblib.load(os.path.join(current_dir, 'encoders_questions.pkl'))
le_target = joblib.load(os.path.join(current_dir, 'encoders_target.pkl'))

# Вопросы и варианты (как в первом скрипте)
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

def predict():
    answers = {}
    for q in questions:
        answers[q] = combo_boxes[q].get()
    
    df_user = pd.DataFrame([answers])
    for col in df_user.columns:
        if col in encoders_questions:
            le = encoders_questions[col]
            df_user[col] = df_user[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
    
    pred_num = model.predict(df_user)[0]
    drink = le_target.inverse_transform([pred_num])[0]
    result_label.config(text=f"🍹 {drink}", fg="green")

# Окно
root = tk.Tk()
root.title("Нейросеть для подбора напитка")
root.geometry("700x750")
root.configure(bg="#f0f0f0")

tk.Label(root, text="Ответь на вопросы и получи свой идеальный напиток!", 
         font=("Arial", 14, "bold"), bg="#f0f0f0").pack(pady=15)

combo_boxes = {}
for q in questions:
    frame = tk.Frame(root, bg="#f0f0f0")
    frame.pack(fill="x", padx=20, pady=5)
    tk.Label(frame, text=q, font=("Arial", 10), bg="#f0f0f0", width=35, anchor="w").pack(side="left")
    cb = ttk.Combobox(frame, values=variants[q], state="readonly", width=30)
    cb.pack(side="right", padx=5)
    cb.set(variants[q][0])
    combo_boxes[q] = cb

btn = tk.Button(root, text="🍹 ПОДОБРАТЬ НАПИТОК", command=predict, 
                font=("Arial", 12, "bold"), bg="#4CAF50", fg="white", padx=20, pady=10)
btn.pack(pady=20)

result_label = tk.Label(root, text="", font=("Arial", 16, "bold"), bg="#f0f0f0")
result_label.pack(pady=10)

root.mainloop()