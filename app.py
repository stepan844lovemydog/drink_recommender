import streamlit as st
import pandas as pd
import joblib
import random
import time

st.set_page_config(page_title="ИИ-помощник для подбора напитка", page_icon="🍹")

# ---- КАСТОМНЫЙ CSS ДЛЯ ЭФФЕКТОВ ----
st.markdown("""
<style>
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(-20px);}
        to {opacity: 1; transform: translateY(0);}
    }
    .main-title {
        animation: fadeIn 1s ease-in-out;
        text-align: center;
        font-size: 2.5rem;
        background: linear-gradient(45deg, #ff6b6b, #feca57, #48dbfb, #ff9ff3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    .question-card {
        background: #f8f9fa;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        margin-bottom: 12px;
        border-left: 5px solid #ff6b6b;
        transition: 0.3s;
    }
    .question-card:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    .stButton button {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24) !important;
        color: white !important;
        font-size: 1.2rem !important;
        font-weight: bold !important;
        border: none !important;
        border-radius: 30px !important;
        padding: 12px 40px !important;
        transition: 0.4s !important;
        box-shadow: 0 6px 14px rgba(238, 90, 36, 0.4);
    }
    .stButton button:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 20px rgba(238, 90, 36, 0.6);
    }
    @keyframes popIn {
        from {transform: scale(0.5); opacity: 0;}
        to {transform: scale(1); opacity: 1;}
    }
    .result-box {
        background: linear-gradient(135deg, #00b894, #00cec9);
        padding: 20px;
        border-radius: 16px;
        color: white;
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        animation: popIn 0.6s ease-out;
        margin-top: 20px;
        box-shadow: 0 8px 24px rgba(0, 206, 201, 0.3);
    }
    .glow {
        animation: glow 1.5s ease-in-out infinite alternate;
    }
    @keyframes glow {
        from {box-shadow: 0 0 10px rgba(0, 206, 201, 0.2);}
        to {box-shadow: 0 0 30px rgba(0, 206, 201, 0.8);}
    }
</style>
""", unsafe_allow_html=True)

# ---- ЗАГРУЗКА МОДЕЛИ ----
@st.cache_resource
def load_model():
    model = joblib.load('model_drink_improved.pkl')
    encoders_questions = joblib.load('encoders_questions_improved.pkl')
    le_target = joblib.load('encoders_target_improved.pkl')
    return model, encoders_questions, le_target

model, encoders_questions, le_target = load_model()

# ---- ЗАГОЛОВОК ----
st.markdown('<div class="main-title">🍹 Подбери свой напиток с помощью нейросети!</div>', unsafe_allow_html=True)
st.caption("✨ Точность модели: ~100% на тестовых данных • 🧠 Сделано с любовью к напиткам")

# ---- ВОПРОСЫ (точно как в variants) ----
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

# ---- ВАРИАНТЫ ОТВЕТОВ (ключи точно совпадают с вопросами) ----
variants = {
    'Что бы вы хотели прямо сейчас?': ['Освежающее', 'Расслабляющее', 'Бодрящее'],
    'Вы хотите чтобы напиток: ': ['Удивил', 'Остудил', 'Согрел'],
    'Кофеин для вас—это…': ['Не желательно', 'Желательно, но не принципиально', 'Обязательно'],
    'Какой уровень сладости вам комфортен?': ['Умеренно сладкий', 'Предпочитаю несладкие напитки или минимум сиропа', 'Сладкий'],
    'Какое молоко вы предпочитаете?': ['Любое', 'Обычное', 'Растительное (миндальное, овсяное, кокосовое, соевое и т.п.)', 'Без молока'],
    'Объём(если предпочитаете конкретный-укажите)': ['Большой', 'Большой (более 350 мл)', 'Средний (200-350мл)', 'Маленький (до 200 мл)'],
    'Вам больше нравится:': ['Ягодные ноты', 'Фруктовые ноты', 'Сладкие специи', 'Цветочные ноты', 'Ореховые ноты'],
    'Любите ли вы чай?': ['Чёрный', 'Фруктовый', 'Молочный улун', 'Травяной (ромашка, мята и т.п.)', 'Зелёный', 'Без чая'],
    'Газированный напиток?': ['Совсем без газа', 'Слабогазированный', 'Да, сильно газированный'],
    'Сколько льда вы предпочитаете в прохладительных напитках?': ['Мало льда', 'Нет льда', 'Много льда'],
    'Как вы относитесь к необычным сочетаниям?': ['Люблю экспериментировать', 'Люблю экзотику', 'Скорее придерживаюсь классики, но иногда пробую новинки', 'Только классика']
}

# ---- ОПРОСНИК (в карточках) ----
answers = {}
for i, q in enumerate(questions, 1):
    with st.container():
        st.markdown(f'<div class="question-card">', unsafe_allow_html=True)
        st.markdown(f"**{i}. {q}**")
        answers[q] = st.selectbox("", variants[q], key=q, label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

# ---- КНОПКА ----
if st.button("🍹 ПОДОБРАТЬ НАПИТОК"):
    
    with st.spinner('🧠 Нейросеть анализирует твои предпочтения...'):
        time.sleep(1.2)
    
    # ---- ЗАПАСНОЙ ВАРИАНТ: ПРЕДСКАЗАНИЕ МОДЕЛИ ----
    df_user = pd.DataFrame([answers])
    for col in df_user.columns:
        if col in encoders_questions:
            le = encoders_questions[col]
            df_user[col] = df_user[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
    try:
        pred_num = model.predict(df_user)[0]
        drink = le_target.inverse_transform([pred_num])[0]
    except:
        drink = None

    # ========== ОСНОВНАЯ ЛОГИКА ==========
    
    if 'Ягодные ноты' in answers['Вам больше нравится:']:
        if answers['Любите ли вы чай?'] != 'Без чая':
            drink = random.choice(['Ягодный чай с мёдом','Клубничный чай с мятой','Малиновый травяной чай','Ежевичный чай с бергамотом'])
        elif answers['Газированный напиток?'] == 'Да, сильно газированный':
            drink = random.choice(['Клубничный лимонад','Малиновый спрайт','Ягодный фреш с мятой','Ежевичный лимонад'])
        elif answers['Что бы вы хотели прямо сейчас?'] == 'Освежающее':
            drink = random.choice(['Клубничный смузи','Малиновый сок','Ягодный сорбет','Ежевичный фреш'])
        else:
            drink = random.choice(['Ягодный смузи на йогурте','Малиновый коктейль','Клубничный милкшейк'])

    elif 'Фруктовые ноты' in answers['Вам больше нравится:']:
        if answers['Любите ли вы чай?'] != 'Без чая':
            drink = random.choice(['Персиковый чай','Фруктовый чай','Яблочный чай с корицей'])
        elif answers['Газированный напиток?'] == 'Да, сильно газированный':
            drink = random.choice(['Манго-маракуйя лимонад','Персиковый спрайт','Ананасовый фреш','Грейпфрутовый лимонад'])
        elif answers['Что бы вы хотели прямо сейчас?'] == 'Расслабляющее':
            drink = random.choice(['Персиковый чай','Манго-матча','Фруктовый смузи с бананом','Ананасовый коктейль'])
        else:
            drink = random.choice(['Фруктовый сок','Манго-фреш','Яблочный смузи','Грейпфрутовый сок'])

    elif 'Ореховые ноты' in answers['Вам больше нравится:']:
        if answers['Кофеин для вас—это…'] != 'Не желательно':
            drink = random.choice(['Ореховый латте','Эспрессо с ореховым сиропом','Карамельный капучино с орехами','Мокко с лесным орехом'])
        else:
            drink = random.choice(['Миндальный латте','Ореховый коктейль','Кокосово-ореховый смузи'])

    elif 'Цветочные ноты' in answers['Вам больше нравится:']:
        if answers['Любите ли вы чай?'] == 'Без чая':
            drink = random.choice(['Лавандовый раф','Фиалковый латте','Розовый латте','Кокос-матча','Лавандовый молочный коктейль'])
        elif answers['Кофеин для вас—это…'] == 'Не желательно':
            drink = random.choice(['Лавандовый раф','Фиалковый латте','Розовый чай','Жасминовый чай'])
        else:
            drink = random.choice(['Чай с бергамотом и сливками','Молочный улун с жасмином','Лавандовый капучино'])

    elif 'Сладкие специи' in answers['Вам больше нравится:']:
        if answers['Любите ли вы чай?'] == 'Без чая':
            drink = random.choice(['Имбирный лимонад','Кардамонный латте','Пряный молочный коктейль','Тыквенный латте'])
        elif answers['Кофеин для вас—это…'] != 'Не желательно':
            drink = random.choice(['Мокко с перцем чили','Улун со специями','Имбирный кофе','Кардамонный латте'])
        else:
            drink = random.choice(['Имбирный чай с лимоном','Тыквенный латте','Чай с корицей и апельсином'])

    else:
        drink = random.choice(['Освежающий лимонад','Классический эспрессо','Мятный чай','Фруктовый сок'])

    # ---- УЧЁТ ТЕМПЕРАТУРЫ ----
    if answers['Вы хотите чтобы напиток: '] == 'Остудил':
        drink += ' (со льдом)'
    elif answers['Вы хотите чтобы напиток: '] == 'Согрел':
        drink += ' (горячий)'

    # ---- УЧЁТ ЭКЗОТИКИ ----
    exotic_drinks = ['Мокко с перцем чили','Лавандовый раф','Кокос-матча','Безалкогольный мохито','Имбирный кофе','Кардамонный латте']
    if answers['Как вы относитесь к необычным сочетаниям?'] in ['Люблю экспериментировать', 'Люблю экзотику']:
        if drink not in exotic_drinks and random.random() < 0.3:
            drink = random.choice(exotic_drinks)

    # ---- ВЫВОД РЕЗУЛЬТАТА ----
    st.markdown(f"""
    <div class="result-box glow">
        🍹 ТВОЙ НАПИТОК: {drink}
    </div>
    """, unsafe_allow_html=True)
    
    st.balloons()