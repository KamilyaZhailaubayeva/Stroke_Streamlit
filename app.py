import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import pickle
import plotly.express as px


# Настройка страницы
st.set_page_config(
    page_title="Stroke Prediction App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Глобальные переменные
uploaded_file = None

# Создание переменных session state
if 'df_input' not in st.session_state:
    st.session_state['df_input'] = pd.DataFrame()

if 'df_predicted' not in st.session_state:
    st.session_state['df_predicted'] = pd.DataFrame()

if 'tab_selected' not in st.session_state:
    st.session_state['tab_selected'] = None

def reset_session_state():
    st.session_state['df_input'] = pd.DataFrame()
    st.session_state['df_predicted'] = pd.DataFrame()

# ML section start
numerical = ['age', 'avg_glucose_level', 'bmi']
categorical = ['gender',
 'hypertension',
 'heart_disease',
 'ever_married',
 'work_type',
 'residence_type',
 'smoking_status']

# logistic regression model
model_file_path = 'models/lr_model_stroke_prediction.sav'
model = pickle.load(open(model_file_path, 'rb'))

# encoding model DictVectorizer
encoding_model_file_path = 'models/encoding_model.sav'
encoding_model = pickle.load(open(encoding_model_file_path, 'rb'))

# Кэширование функции предсказания
@st.cache_data
def predict_stroke(df_input, treshold):

    scaler = MinMaxScaler()

    df_original = df_input.copy()
    df_input[numerical] = scaler.fit_transform(df_input[numerical])
    
    dicts_df = df_input[categorical + numerical].to_dict(orient='records')
    X = encoding_model.transform(dicts_df)
    y_pred = model.predict_proba(X)[:, 1]
    stroke_descision = (y_pred >= treshold).astype(int)
    df_original['stroke_predicted'] = stroke_descision
    df_original['stroke_predicted_probability'] = y_pred

    return df_original

@st.cache_data
def convert_df(df):
    # Функция для конвертации датафрейма в csv
    return df.to_csv(index=False).encode('utf-8')

# Sidebar section start
# Сайдбар блок
with st.sidebar:
    st.title('🗂 Ввод данных')
    
    tab1, tab2 = st.tabs(['📁 Данные из файла', '📝 Ввести вручную'])
    with tab1:
        # Вкладка с загрузкой файла, выбором порога и кнопкой предсказания (вкладка 1)
        uploaded_file = st.file_uploader("Выбрать CSV файл", type=['csv', 'xlsx'], on_change=reset_session_state)
        if uploaded_file is not None:
            treshold = st.slider('Порог вероятности оттока', 0.0, 1.0, 0.5, 0.01, key='slider1')
            prediction_button = st.button('Предсказать', type='primary', use_container_width=True, key='button1')
            st.session_state['df_input'] = pd.read_csv(uploaded_file)
            if prediction_button:
                # Предсказание и сохранение в session state
                st.session_state['df_predicted'] = predict_stroke(st.session_state['df_input'], treshold)
                st.session_state['tab_selected'] = 'tab1'

    with tab2:
        # Вкладка с вводом данных вручную, выбором порога и кнопкой предсказания (вкладка 2)
        id = st.text_input('Patient ID', placeholder='0000', help='Введите ID клиента')
        gender = st.selectbox( 'Пол', ('Женщина', 'Мужчина'))
        age = st.number_input('Возраст пациента', min_value=0, max_value=150, value=0)
        hypertension = st.selectbox('Гипертония', ('Да', 'Нет'))
        heart_disease = st.selectbox('Болезни сердца', ('Да', 'Нет'))
        ever_married = st.selectbox('Замужем/женаты или когда-то были', ('Да', 'Нет'))
        work_type = st.selectbox('Тип работы', ('private', 'self_employed', 'govt_job', 'children'))
        residence_type = st.selectbox('Место проживания', ('В городе', 'В деревне'))
        avg_glucose_level = st.number_input('Средний уровень глюкозы', min_value=0, max_value=300, value=0)
        bmi = st.number_input('Индекс массы тела', min_value=0, max_value=100, value=0)
        smoking_status = st.selectbox('Статус курения', ('formerly_smoked', 'never_smoked', 'smokes', 'unknown'))
             
        
        # Если введен ID пациента, то показываем слайдер с порогом и кнопку предсказания
        if id != '':
            treshold = st.slider('Порог вероятности инсульта', 0.0, 1.0, 0.5, 0.01, key='slider2')
            prediction_button_tab2 = st.button('Предсказать', type='primary', use_container_width=True, key='button2')
            
            if prediction_button_tab2:
                st.session_state['tab_selected'] = 'tab2'
                # Сохраняем введенные данные в session state в виде датафрейма
                st.session_state['df_input'] = pd.DataFrame({
                    'id': id,
                    'gender': 'female' if gender == 'Женщина' else 'male',
                    'age': int(age),
                    'hypertension': 1 if hypertension == 'Да' else 0,
                    'heart_disease': 1 if heart_disease == 'Да' else 0,
                    'ever_married': 'yes' if ever_married == 'Да' else 'no',       
                    'work_type': work_type,
                    'residence_type': 'urban' if residence_type == 'В городе' else 'rural',
                    'avg_glucose_level': float(avg_glucose_level),
                    'bmi': float(bmi),
                    'smoking_status': smoking_status
                }, index=[0])
                # Предсказание и сохранение в session state
                st.session_state['df_predicted'] = predict_stroke(st.session_state['df_input'], treshold)

                

# Sidebar section end

# Main section start
# Основной блок
st.image('https://www.verywellhealth.com/thmb/uVW7Rx19dvUe9oesk8BD6eCHWcY=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/GettyImages-670886585-25e3cf6a70884fd2b427810085301f8f.jpg', width=400)
st.title('Прогнозирование инсульта у пациентов')

with st.expander("Описание проекта"):
    st.write("""
    В данном проекте мы рассмотрим задачу прогнозирования инсульта у пациентов.
    Для этого мы будем использовать датасет из открытых источников.
    Датасет содержит информацию о пациентах, у которых уже был или еще не был инсульт.
    Наша задача - построить модель, которая будет предсказывать инсульт у пациентов.
    """)

# Вывод входных данных (из файла или введенных пациентом/врачом)
if len(st.session_state['df_input']) > 0:
    # Если предсказание еще не было сделано, то выводим входные данные в общем виде
    if len(st.session_state['df_predicted']) == 0:
        st.subheader('Данные из файла')
        st.write(st.session_state['df_input'])
    else:
        # Если предсказание уже было сделано, то выводим входные данные в expander
        with st.expander("Входные данные"):
            st.write(st.session_state['df_input'])
    # Примеры визуализации данных
    # st.line_chart(st.session_state['df_input'][['tenure', 'monthlycharges']])
    # st.bar_chart(st.session_state['df_input'][['contract']])


# Выводим результаты предсказания для отдельного пациента (вкладка 2)
if len(st.session_state['df_predicted']) > 0 and st.session_state['tab_selected'] == 'tab2':
    if st.session_state['df_predicted']['stroke_predicted'][0] == 0:
        st.image('https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExdjM1cWdkYjMydWFkZmd0d3d5eGgwaHVnZ2k5bG95c2Fta29mejJrOSZlcD12MV9naWZzX3NlYXJjaCZjdD1n/YkCcPWkMuUnRIwr6F1/giphy.gif', width=200)
        st.subheader(f'У пациента :green[не может] случится инсульт c вероятностью {(1 - st.session_state["df_predicted"]["stroke_predicted_probability"][0]) * 100:.2f}%')
    else:
        st.image('https://gifdb.com/images/high/doctor-with-bad-news-4ohm43md81pax66f.gif', width=200)
        st.subheader(f'У пациета :red[может] случится инсульт c вероятностью {(st.session_state["df_predicted"]["stroke_predicted_probability"][0]) * 100:.2f}%')


# Выводим результаты предсказания для пациентов из файла (вкладка 1)
if len(st.session_state['df_predicted']) > 0 and st.session_state['tab_selected'] == 'tab1':
    # Результаты предсказания для всех пациентов в файле
    st.subheader('Результаты прогнозирования')
    st.write(st.session_state['df_predicted'])
    # Скачиваем результаты предсказания для всех пациентов в файле
    res_all_csv = convert_df(st.session_state['df_predicted'])
    st.download_button(
        label="Скачать все предсказания",
        data=res_all_csv,
        file_name='df-stroke-predicted-all.csv',
        mime='text/csv',
    )

    # Гистограмма оттока для всех клиентов в файле
    fig = px.histogram(st.session_state['df_predicted'], x='stroke_predicted', color='stroke_predicted')
    st.plotly_chart(fig, use_container_width=True)

    # Пациенты с высоким риском инсульта
    risk_patients = st.session_state['df_predicted'][st.session_state['df_predicted']['stroke_predicted'] == 1]
    # Выводим пациентов с высоким риском инсульта
    if len(risk_patients) > 0:
        st.subheader('Пациенты с высоким риском инсульта')
        st.write(risk_patients)
        # Скачиваем пациентов с высоким риском инсульта
        res_risky_csv = convert_df(risk_patients)
        st.download_button(
            label="Скачать пациентов с высоким риском инсульта",
            data=res_risky_csv,
            file_name='df-stroke-predicted-risk-patients.csv',
            mime='text/csv',
        )