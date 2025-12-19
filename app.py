import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Предсказание цен авто", layout="wide")

st.markdown("# Предсказание цен на автомобили")
st.divider()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    with open(os.path.join(SCRIPT_DIR, 'model.pickle'), 'rb') as f:
        return pickle.load(f)

model_data = load_model()
model = model_data['model']
scaler = model_data['scaler']
feature_cols = model_data['feature_cols']
cat_cols = model_data['cat_cols']
train_columns = model_data['train_columns']
corr_pearson = model_data.get('corr_pearson')
corr_kendall = model_data.get('corr_kendall')
corr_phik = model_data.get('corr_phik')
model_comparison = model_data.get('model_comparison', {})
l0_results = model_data.get('l0_results', {})
business_results = model_data.get('business_results', {})
data_stats = model_data.get('data_stats', {})

@st.cache_data
def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv')
    df['brand'] = df['name'].str.split().str[0]
    return df

df = load_data()

tab1, tab2, tab3, tab4 = st.tabs(["Анализ данных", "Предсказание", "Веса модели", "Эксперименты"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(df, x='selling_price', nbins=50, title="Распределение цен")
        st.plotly_chart(fig, use_container_width=True)
        
        fig = px.scatter(df, x='max_power', y='selling_price', color='transmission', 
                        opacity=0.5, title="Цена vs Мощность")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(df, x='year', y='selling_price', title="Цена по годам")
        st.plotly_chart(fig, use_container_width=True)
        
        fig = px.box(df, x='fuel', y='selling_price', title="Цена по топливу")
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Корреляции")
    corr_type = st.radio("Выберите тип:", ["Пирсон", "Кендалл", "PhiK"], horizontal=True)
    
    if corr_type == "Пирсон" and corr_pearson is not None:
        fig = px.imshow(corr_pearson, text_auto='.2f', color_continuous_scale='RdBu_r')
    elif corr_type == "Кендалл" and corr_kendall is not None:
        fig = px.imshow(corr_kendall, text_auto='.2f', color_continuous_scale='RdBu_r')
    elif corr_type == "PhiK" and corr_phik is not None:
        fig = px.imshow(corr_phik, text_auto='.2f', color_continuous_scale='Viridis')
    else:
        corr = df.select_dtypes(include=[np.number]).corr()
        fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r')
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    input_method = st.radio("", ["Ручной ввод", "Загрузить CSV"], horizontal=True)
    
    if input_method == "Ручной ввод":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            year = st.number_input("Год выпуска", int(df['year'].min()), int(df['year'].max()), int(df['year'].median()))
            km_driven = st.number_input("Пробег (км)", 0, int(df['km_driven'].max()), int(df['km_driven'].median()))
            mileage = st.number_input("Расход (км/л)", 0.0, 50.0, 18.0)
            engine = st.number_input("Объём двигателя (CC)", 500, 5000, 1200)
        
        with col2:
            max_power = st.number_input("Мощность (bhp)", 30.0, 500.0, 80.0)
            torque = st.number_input("Крутящий момент (Nm)", 50.0, 800.0, 150.0)
            brand = st.selectbox("Марка", sorted(df['brand'].unique()))
            fuel = st.selectbox("Топливо", df['fuel'].unique())
        
        with col3:
            transmission = st.selectbox("Коробка", df['transmission'].unique())
            seller_type = st.selectbox("Продавец", df['seller_type'].unique())
            owner = st.selectbox("Владелец", df['owner'].unique())
        
        if st.button("Рассчитать цену", use_container_width=True):
            try:
                age = 2025 - year
                X_num = pd.DataFrame([[year, year**2, age, km_driven, np.log1p(km_driven), 
                                       km_driven/max(age,1), mileage, engine, max_power, torque,
                                       max_power/(engine/1000), 1 if owner=='First Owner' else 0]],
                                    columns=feature_cols)
                
                X_scaled = pd.DataFrame(scaler.transform(X_num), columns=feature_cols)
                cat_encoded = pd.get_dummies(pd.DataFrame([[brand, fuel, transmission, seller_type]], 
                                                          columns=cat_cols), drop_first=True)
                
                X_final = pd.concat([X_scaled, cat_encoded], axis=1)
                X_final = X_final.reindex(columns=train_columns, fill_value=0)
                
                pred = np.expm1(model.predict(X_final)[0])
                st.success(f"### Предсказанная цена: ₹{pred:,.0f}")
            except Exception as e:
                st.error(f"Ошибка: {e}")
    
    else:
        uploaded_file = st.file_uploader("Выберите CSV файл", type=['csv'])
        
        if uploaded_file and st.button("Рассчитать цены"):
            input_df = pd.read_csv(uploaded_file)
            
            input_df['age'] = 2025 - input_df['year']
            input_df['year_squared'] = input_df['year'] ** 2
            input_df['km_per_year'] = input_df['km_driven'] / input_df['age'].replace(0, 1)
            input_df['log_km'] = np.log1p(input_df['km_driven'])
            input_df['power_per_liter'] = input_df['max_power'] / (input_df['engine'] / 1000)
            input_df['is_first_owner'] = (input_df['owner'] == 'First Owner').astype(int)
            input_df['brand'] = input_df['name'].str.split().str[0]
            
            X_scaled = pd.DataFrame(scaler.transform(input_df[feature_cols]), columns=feature_cols)
            cat_encoded = pd.get_dummies(input_df[cat_cols], drop_first=True)
            
            X_final = pd.concat([X_scaled.reset_index(drop=True), cat_encoded.reset_index(drop=True)], axis=1)
            X_final = X_final.reindex(columns=train_columns, fill_value=0)
            
            predictions = np.expm1(model.predict(X_final))
            
            result = input_df[['name', 'year', 'km_driven']].copy()
            result['Предсказанная цена'] = predictions.astype(int)
            st.dataframe(result, use_container_width=True, hide_index=True)

with tab3:
    coef_df = pd.DataFrame({'Признак': train_columns, 'Коэффициент': model.coef_})
    coef_df['Абс'] = np.abs(coef_df['Коэффициент'])
    coef_df = coef_df.sort_values('Абс', ascending=True).tail(20)
    
    fig = go.Figure(go.Bar(
        y=coef_df['Признак'],
        x=coef_df['Коэффициент'],
        orientation='h',
        marker_color=['crimson' if x < 0 else 'seagreen' for x in coef_df['Коэффициент']]
    ))
    fig.update_layout(title="Топ-20 признаков", height=600)
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    if data_stats:
        st.subheader("Статистика данных")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Train", data_stats.get('train_size', '-'))
        col2.metric("Test", data_stats.get('test_size', '-'))
        col3.metric("Пропуски", data_stats.get('missing_values', '-'))
        col4.metric("Дубликаты", data_stats.get('duplicates_removed', '-'))
    
    if model_comparison:
        st.subheader("Сравнение моделей")
        
        models = list(model_comparison.keys())
        r2_train = [model_comparison[m]['r2_train'] for m in models]
        r2_test = [model_comparison[m]['r2_test'] for m in models]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='R2 Train', x=models, y=r2_train))
        fig.add_trace(go.Bar(name='R2 Test', x=models, y=r2_test))
        fig.update_layout(barmode='group')
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(pd.DataFrame(model_comparison).T, use_container_width=True)
    
    if l0_results:
        st.subheader("L0 регуляризация")
        
        fig = px.line(x=list(l0_results.keys()), 
                      y=[l0_results[k]['r2'] for k in l0_results.keys()], 
                      markers=True, labels={'x': 'Признаков', 'y': 'R2'})
        st.plotly_chart(fig, use_container_width=True)
        
        for k, v in l0_results.items():
            with st.expander(f"{k} признаков (R2={v['r2']:.2f})"):
                st.write(', '.join(v['features']))
    
    if business_results:
        st.subheader("Бизнес-метрики")
        
        models = list(business_results.keys())
        within_10 = [business_results[m]['within_10pct'] * 100 for m in models]
        
        fig = px.bar(x=models, y=within_10, text=[f'{v:.0f}%' for v in within_10])
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)