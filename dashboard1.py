import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sqlalchemy import create_engine
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Painel Epidemiológico - Belém (PA)", layout="wide", page_icon="📊")

# --- 1. SUA FUNÇÃO DE CONEXÃO ---
def get_db_engine():
    try:
        # Credenciais
        user = 'postgres'
        password = 'nathy2004' 
        host = 'localhost'
        port = '5432'
        dbname = 'sindromegripal'
        
        # Cria a URL de conexão que o SQLAlchemy exige
        url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
        
        # Cria a Engine (O gerenciador de conexões)
        engine = create_engine(url)
        return engine
    except Exception as e:
        st.error(f"Erro ao configurar conexão: {e}")
        return None 
# --- 2. CONFIGURAÇÃO DE LATITUDE/LONGITUDE ---
# (Necessário pois sua tabela 'municipio' só tem o nome, não as coordenadas)
COORDS_PARA = {
    'Belém': {'lat': -1.4558, 'lon': -48.5044},
    'Ananindeua': {'lat': -1.3636, 'lon': -48.3734},
    'Santarém': {'lat': -2.4431, 'lon': -54.7083},
    'Marabá': {'lat': -5.3686, 'lon': -49.1174},
    'Parauapebas': {'lat': -6.0675, 'lon': -49.9042},
    'Castanhal': {'lat': -1.2964, 'lon': -47.9258},
    'Abaetetuba': {'lat': -1.7218, 'lon': -48.8858},
    'Cametá': {'lat': -2.2427, 'lon': -49.4965},
    'Bragança': {'lat': -1.0536, 'lon': -46.7656},
    'Altamira': {'lat': -3.2033, 'lon': -52.2025}
}

# --- 3. CARREGAMENTO DE DADOS (USANDO SUA CONEXÃO) ---
@st.cache_data(ttl=600)
def load_data():
    engine = get_db_engine()
    
    if engine is None:
        return pd.DataFrame()

    query = """
    SELECT 
        n.id,
        n.data_notificacao,
        n.idade,
        n.classificacaoFinal as classificacao_final,
        n.evolucaoCaso as evolucao,
        sx.descricao as sexo,
        rc.descricao as raca_cor,
        m.nome as municipio,
        (
            SELECT STRING_AGG(s.descricao, ',')
            FROM notificacao_sintoma ns
            JOIN sintoma s ON ns.sintoma_id = s.id
            WHERE ns.notificacao_id = n.id
        ) as sintomas,
        'Não Informado' as escolaridade 
    FROM notificacao n
    LEFT JOIN pessoa p ON n.pessoa_id = p.id
    LEFT JOIN sexo sx ON p.sexo_id = sx.id
    LEFT JOIN raca_cor rc ON p.raca_cor_id = rc.id
    LEFT JOIN notificacao_municipio nm ON nm.notificacao_id = n.id
    LEFT JOIN municipio m ON nm.municipio_id = m.id
    WHERE n.excluido IS FALSE OR n.excluido IS NULL
    LIMIT 500;
    """
    
    try:
        # AQUI MUDOU: passamos a 'engine' e não mais a 'conn' crua
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
            
    except Exception as e:
        st.error(f"Erro na execução da Query: {e}")
        return pd.DataFrame()
    # 3. Pandas lê direto do banco
    try:
        # AQUI MUDOU: O Pandas lê direto da 'engine'.
        # Isso evita o erro de "Connection is closed".
        df = pd.read_sql(query, engine)
        
    except Exception as e:
        st.error(f"Erro na execução da Query: {e}")
        return pd.DataFrame()

    # Tratamento dos dados (Só executa se o DF não estiver vazio)
    if not df.empty:
        df['data_notificacao'] = pd.to_datetime(df['data_notificacao'])
        df['idade'] = pd.to_numeric(df['idade'], errors='coerce').fillna(0)
        
        # Injetar Lat/Lon
        df['lat'] = df['municipio'].map(lambda x: COORDS_PARA.get(x, {}).get('lat', None))
        df['lon'] = df['municipio'].map(lambda x: COORDS_PARA.get(x, {}).get('lon', None))
        
        def categorizar_idade(i):
            if i <= 12: return '0-12 (Criança)'
            elif i <= 19: return '13-19 (Adolescente)'
            elif i <= 59: return '20-59 (Adulto)'
            else: return '60+ (Idoso)'
        df['faixa_etaria'] = df['idade'].apply(categorizar_idade)
        
        df['sintomas'] = df['sintomas'].fillna('Assintomático')
    
    return df

# Executa o carregamento
try:
    df = load_data()
    if df.empty:
        st.warning("⚠️ O banco conectou, mas a tabela está vazia ou a query não retornou dados.")
        st.stop()
except Exception as e:
    st.error(f"Erro crítico: {e}")
    st.stop()


# --- 4. FUNÇÃO DE IA (TREINAMENTO) ---
@st.cache_resource
def treinar_modelo_sg(df):
    df_model = df.copy()
    
    # 1. Tratamento mais flexível do Target (Aceita COVID, Positivo, Confirmado...)
    def classificar_target(valor):
        texto = str(valor).upper()
        # Se tiver qualquer uma dessas palavras, vira 1 (Positivo)
        if 'COVID' in texto or 'POSITIVO' in texto or 'CONFIRMADO' in texto or 'DETECTÁVEL' in texto:
            return 1
        return 0

    df_model['target'] = df_model['classificacao_final'].apply(classificar_target)
    
    # 2. VERIFICAÇÃO DE SEGURANÇA (O Pulo do Gato para corrigir seu erro)
    # Se só tiver 1 tipo de classe (só zeros ou só uns), aborta o treino para não travar
    unique_classes = df_model['target'].unique()
    if len(unique_classes) < 2:
        # Retorna None para avisar que não deu para treinar
        return None, 0, []
    
    # --- Continua o código normal se tiver dados suficientes ---
    sintomas_reais = df_model['sintomas'].str.get_dummies(sep=',').columns.tolist()
    sintomas_interesse = ['Febre', 'Tosse', 'Dor de Garganta', 'Dispneia', 'Dor de Cabeça', 'Perda de Olfato', 'Mialgia']
    features_sintomas = [s for s in sintomas_reais if any(sub in s for sub in sintomas_interesse)]
    
    if not features_sintomas:
        features_sintomas = sintomas_reais[:5]

    for s in features_sintomas:
        df_model[s] = df_model['sintomas'].apply(lambda x: 1 if s in str(x) else 0)
        
    features = features_sintomas + ['idade']
    X = df_model[features]
    y = df_model['target']
    
    try:
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        acc = model.score(X, y)
        return model, acc, features_sintomas
    except Exception as e:
        print(f"Erro no treino: {e}")
        return None, 0, []

# --- 5. LAYOUT E VISUALIZAÇÃO ---
st.sidebar.title("Filtros Regionais")
lista_cidades = df['municipio'].dropna().unique()
cidade_filtro = st.sidebar.multiselect("Município", lista_cidades, default=lista_cidades)

df_filtrado = df[df['municipio'].isin(cidade_filtro)]

st.title("📊 Painel de Vigilância Epidemiológica - Pará (DB Real)")

tab1, tab2, tab3, tab4 = st.tabs(["Visão Geral", "👥 Demografia & Social", "Análise Clínica", "🤖 IA Preditiva"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Notificações", len(df_filtrado))
    col2.metric("COVID-19", len(df_filtrado[df_filtrado['classificacao_final'] == 'COVID-19']))
    col3.metric("Influenza", len(df_filtrado[df_filtrado['classificacao_final'] == 'Influenza']))
    col4.metric("Óbitos", len(df_filtrado[df_filtrado['evolucao'] == 'Óbito']), delta_color="inverse")

    st.markdown("### Mapa de Calor - Notificações")
    map_data = df_filtrado.dropna(subset=['lat', 'lon'])
    if not map_data.empty:
        st.map(map_data[['lat', 'lon']], zoom=5)
    else:
        st.info("Sem coordenadas para exibir o mapa.")
    
    casos_tempo = df_filtrado.groupby([pd.Grouper(key='data_notificacao', freq='W'), 'classificacao_final']).size().reset_index(name='contagem')
    fig_line = px.line(casos_tempo, x='data_notificacao', y='contagem', color='classificacao_final', title="Evolução Semanal")
    st.plotly_chart(fig_line, use_container_width=True)

with tab2:
    st.markdown("### Perfil Sociodemográfico")
    c1, c2 = st.columns(2)
    
    # Pirâmide Etária
    fig_pyramid = px.histogram(df_filtrado, x="idade", color="sexo", marginal="box", 
                               nbins=20, barmode="overlay", opacity=0.7, title="Idade x Sexo")
    c1.plotly_chart(fig_pyramid, use_container_width=True)

    # Raça/Cor
    df_raca = df_filtrado['raca_cor'].value_counts().reset_index()
    df_raca.columns = ['Raça', 'Total']
    fig_raca = px.bar(df_raca, x='Total', y='Raça', orientation='h', title="Raça/Cor")
    c2.plotly_chart(fig_raca, use_container_width=True)

with tab3:
    st.markdown("### Análise de Sintomas")
    c_sint1, c_sint2 = st.columns(2)
    
    # Sintomas Frequentes
    sintomas_series = df_filtrado['sintomas'].str.get_dummies(sep=',')
    if not sintomas_series.empty:
        sintomas_sum = sintomas_series.sum().sort_values(ascending=True)
        fig_sint = px.bar(x=sintomas_sum.values, y=sintomas_sum.index, orientation='h', title="Sintomas + Comuns")
        c_sint1.plotly_chart(fig_sint, use_container_width=True)
    
    # Heatmap
    df_exploded = df_filtrado.assign(sintoma=df_filtrado['sintomas'].str.split(',')).explode('sintoma')
    df_heat = df_exploded.groupby(['sintoma', 'classificacao_final']).size().reset_index(name='contagem')
    if not df_heat.empty:
        fig_h = px.scatter(df_heat, x='classificacao_final', y='sintoma', size='contagem', color='contagem', title="Correlação")
        c_sint2.plotly_chart(fig_h, use_container_width=True)

with tab4:
    st.markdown("### 🤖 Triagem IA")
    
    # Chama a função nova
    modelo, acuracia, feature_sintomas = treinar_modelo_sg(df)
    
    # Se modelo for None, significa que caiu naquela proteção que criamos
    if modelo is None:
        st.warning("⚠️ **Dados insuficientes para treinar a IA.**")
        st.info("Para a IA funcionar, o banco de dados precisa ter exemplos variados (pelo menos um caso Positivo e um Negativo).")
        st.write("Diagnósticos encontrados no banco atualmente:", df['classificacao_final'].unique())
    
    else:
        # Se o modelo treinou, mostra o formulário normal
        st.success(f"Modelo treinado com sucesso! Acurácia histórica: **{acuracia*100:.1f}%**")
        
        with st.form("ia_form"):
            c1, c2 = st.columns([1, 3])
            idade_in = c1.number_input("Idade", 0, 120, 30)
            checks = {}
            cols = c2.columns(3)
            for i, s in enumerate(feature_sintomas):
                with cols[i%3]: checks[s] = st.checkbox(s)
                
            if st.form_submit_button("Calcular Risco"):
                vetor = [checks[s] for s in feature_sintomas] + [idade_in]
                prob = modelo.predict_proba([vetor])[0][1] * 100
                st.metric("Probabilidade COVID-19", f"{prob:.1f}%")
                if prob > 50: st.error("Alta Probabilidade")
                else: st.success("Baixa Probabilidade")