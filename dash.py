import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Painel Epidemiológico - Belém (PA)", layout="wide", page_icon="📊")

# carregamento de dados
@st.cache_data
def load_data():
    
    df = pd.read_csv('dados_sindrome_gripal_completo.csv')
    df['data_notificacao'] = pd.to_datetime(df['data_notificacao'])
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error(" Arquivo 'dados_sindrome_gripal_completo.csv' não encontrado.")
    st.stop()

# treinamento de modelo
@st.cache_resource
def treinar_modelo_sg(df):
    df_model = df.copy()
    df_model['target'] = df_model['classificacao_final'].apply(lambda x: 1 if x == 'COVID-19' else 0)
    
    sintomas_possiveis = ['Febre', 'Tosse', 'Dor de Garganta', 'Dispneia', 'Dor de Cabeça', 'Perda de Olfato/Paladar', 'Mialgia (Dor no corpo)', 'Coriza', 'Fadiga']
    for s in sintomas_possiveis:
        df_model[s] = df_model['sintomas'].apply(lambda x: 1 if s in str(x) else 0)
        
    features = sintomas_possiveis + ['idade']
    X_train, X_test, y_train, y_test = train_test_split(df_model[features], df_model['target'], test_size=0.2, random_state=42)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc, sintomas_possiveis

# layout e filtros
st.sidebar.title("Filtros Regionais")
cidade_filtro = st.sidebar.multiselect("Município", df['municipio'].unique(), default=df['municipio'].unique())
df_filtrado = df[df['municipio'].isin(cidade_filtro)]

st.title("📊 Painel de Vigilância Epidemiológica - Pará")


tab1, tab2, tab3, tab4 = st.tabs(["Visão Geral", "👥 Demografia & Social", "Análise Clínica", "🤖 IA Preditiva"])

# visão geral
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Notificações", len(df_filtrado))
    col2.metric("COVID-19", len(df_filtrado[df_filtrado['classificacao_final'] == 'COVID-19']))
    col3.metric("Influenza", len(df_filtrado[df_filtrado['classificacao_final'] == 'Influenza']))
    col4.metric("Óbitos", len(df_filtrado[df_filtrado['evolucao'] == 'Óbito']), delta_color="inverse")

    st.markdown("### Mapa de Calor - Notificações")
    st.map(df_filtrado[['lat', 'lon']], zoom=5)
    
    # 
    casos_tempo = df_filtrado.groupby([pd.Grouper(key='data_notificacao', freq='W'), 'classificacao_final']).size().reset_index(name='contagem')
    fig_line = px.line(casos_tempo, x='data_notificacao', y='contagem', color='classificacao_final', title="Evolução Semanal de Casos")
    st.plotly_chart(fig_line, use_container_width=True)

# tab2 - demografia 
with tab2:
    st.markdown("### Perfil Sociodemográfico da População Atingida")
    
    col_demo1, col_demo2 = st.columns(2)
    
    with col_demo1:
        # 1. distribuição etária por sexo
        st.markdown("**Distribuição Etária por Sexo**")
        fig_pyramid = px.histogram(df_filtrado, x="idade", color="sexo", 
                                   marginal="box", 
                                   nbins=20, barmode="overlay", opacity=0.7,
                                   title="Histograma de Idade: Masculino vs Feminino")
        st.plotly_chart(fig_pyramid, use_container_width=True)

    with col_demo2:
        # 2. raça/cor
        st.markdown("**Autodeclaração de Raça/Cor**")
        contagem_raca = df_filtrado['raca_cor'].value_counts().reset_index()
        contagem_raca.columns = ['Raça/Cor', 'Total']
        fig_raca = px.bar(contagem_raca, x='Total', y='Raça/Cor', orientation='h', text='Total', color='Total', color_continuous_scale='Blues')
        st.plotly_chart(fig_raca, use_container_width=True)
    
    st.divider()
    
    col_demo3, col_demo4 = st.columns(2)
    
    with col_demo3:
        
        st.markdown("**Desfecho Clínico por Faixa Etária**")
        
        df_faixa = df_filtrado.groupby(['faixa_etaria', 'evolucao']).size().reset_index(name='quantidade')
        fig_evolucao = px.bar(df_faixa, x="faixa_etaria", y="quantidade", color="evolucao", 
                              title="Impacto por Grupo Etário",
                              category_orders={"faixa_etaria": ["0-12 (Criança)", "13-19 (Adolescente)", "20-59 (Adulto)", "60+ (Idoso)"]})
        st.plotly_chart(fig_evolucao, use_container_width=True)

    with col_demo4:
       
        st.markdown("**Escolaridade e Diagnóstico**")
        # filtrar crianças para não poluir escolaridade
        df_adultos = df_filtrado[df_filtrado['escolaridade'] != 'Não se aplica (criança)']
        fig_sun = px.sunburst(df_adultos, path=['escolaridade', 'classificacao_final'], title="Relação Escolaridade x Infecção")
        st.plotly_chart(fig_sun, use_container_width=True)

# tab 3 - analise clinica
with tab3:
    st.markdown("### Análise de Sintomas e Testagem")
    col_clin1, col_clin2 = st.columns(2)
    
    # Sintomas mais comuns
    sintomas_series = df_filtrado['sintomas'].str.get_dummies(sep=',')
    sintomas_sum = sintomas_series.sum().sort_values(ascending=True)
    fig_sintomas = px.bar(x=sintomas_sum.values, y=sintomas_sum.index, orientation='h', title="Sintomas Mais Frequentes")
    col_clin1.plotly_chart(fig_sintomas, use_container_width=True)
    
    # Relação Sintoma x Diagnóstico Final (Heatmap simples via bolhas)
    # Explodir os sintomas para cruzar com diagnóstico
    df_exploded = df_filtrado.assign(sintoma=df_filtrado['sintomas'].str.split(',')).explode('sintoma')
    df_heatmap = df_exploded.groupby(['sintoma', 'classificacao_final']).size().reset_index(name='contagem')
    fig_heat = px.scatter(df_heatmap, x='classificacao_final', y='sintoma', size='contagem', color='contagem', title="Correlação Sintoma x Doença")
    col_clin2.plotly_chart(fig_heat, use_container_width=True)

# tab 4 - triagem IA
with tab4:
    st.markdown("### 🧬 Triagem")
    modelo, acuracia, feature_sintomas = treinar_modelo_sg(df)
    st.info(f"Modelo com **{acuracia*100:.1f}%** de precisão.")
    
    with st.form("ia_form"):
        c1, c2 = st.columns([1, 3])
        idade_in = c1.number_input("Idade", 0, 120, 30)
        
        checks = {}
        cols = c2.columns(3)
        for i, s in enumerate(feature_sintomas):
            with cols[i%3]: checks[s] = st.checkbox(s)
            
        if st.form_submit_button("Calcular Risco COVID"):
            input_vetor = [checks[s] for s in feature_sintomas] + [idade_in]
            prob = modelo.predict_proba([input_vetor])[0][1] * 100
            
            st.metric("Probabilidade COVID-19", f"{prob:.1f}%")
            if prob > 50: st.error("Alta compatibilidade com COVID-19.")
            else: st.success("Perfil compatível com outras síndromes gripais.")