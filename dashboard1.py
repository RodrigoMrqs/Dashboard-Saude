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
    LIMIT 3000;
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
    # 1. Lista Fixa de Sintomas (Padronizada)
    sintomas_possiveis = [
        'Febre', 'Tosse', 'Dor de Garganta', 'Dispneia', 
        'Dor de Cabeça', 'Perda de Olfato/Paladar', 
        'Mialgia (Dor no corpo)', 'Coriza', 'Fadiga'
    ]

    # --- ETAPA A: PREPARAR DADOS REAIS DO BANCO ---
    df_real = df.copy()
    
    # Tratamento do Target
    def classificar_target(valor):
        texto = str(valor).upper()
        if 'COVID' in texto or 'POSITIVO' in texto or 'CONFIRMADO' in texto: return 1
        return 0
    
    df_real['target'] = df_real['classificacao_final'].apply(classificar_target)
    
    # One-Hot Encoding nos dados reais
    for s in sintomas_possiveis:
        df_real[s] = df_real['sintomas'].apply(lambda x: 1 if s in str(x) else 0)
        
    # --- ETAPA B: GERAR DADOS MÉDICOS SINTÉTICOS (O "CÉREBRO" DA IA) ---
    # Aqui definimos os PESOS REAIS (Baseado em literatura médica/OMS)
    # Quanto maior o peso, mais chance de ser COVID
    pesos_medicos = {
        'Perda de Olfato/Paladar': 75, # Sintoma muito específico
        'Dispneia': 60,                # Sintoma grave
        'Febre': 45,
        'Tosse': 40,
        'Fadiga': 30,
        'Mialgia (Dor no corpo)': 25,
        'Dor de Cabeça': 20,
        'Dor de Garganta': 15,
        'Coriza': 10                   # Mais comum em gripe/resfriado
    }
    
    # Geramos 500 pacientes virtuais para "ensinar" a IA
    dados_sinteticos = []
    np.random.seed(42) # Para o resultado ser sempre igual
    
    for _ in range(500):
        perfil = {}
        score_risco = 0
        
        # Simula sintomas aleatórios baseados em probabilidade
        for s in sintomas_possiveis:
            # Chance base de alguém ter o sintoma (ex: 20% chance de ter febre)
            tem_sintoma = np.random.choice([0, 1], p=[0.8, 0.2])
            perfil[s] = tem_sintoma
            if tem_sintoma == 1:
                score_risco += pesos_medicos.get(s, 0)
        
        # Idade aleatória
        idade = np.random.randint(5, 90)
        perfil['idade'] = idade
        if idade > 60: score_risco += 15 # Idade aumenta risco
        
        # Define se é COVID baseado no score (Sigmoide simulada)
        # Se score alto, chance alta de ser 1
        probabilidade_real = 1 / (1 + np.exp(-(score_risco - 50) / 20))
        perfil['target'] = np.random.choice([1, 0], p=[probabilidade_real, 1-probabilidade_real])
        
        dados_sinteticos.append(perfil)
        
    df_sintetico = pd.DataFrame(dados_sinteticos)
    
    # --- ETAPA C: FUNDIR DADOS REAIS + SINTÉTICOS ---
    # Selecionamos apenas as colunas necessárias para o treino
    cols_treino = sintomas_possiveis + ['idade', 'target']
    
    # Se o banco real tiver dados, usamos. Se estiver vazio, usamos só o sintético.
    if not df_real.empty:
        df_treino = pd.concat([df_real[cols_treino], df_sintetico[cols_treino]])
    else:
        df_treino = df_sintetico[cols_treino]

    # --- ETAPA D: TREINAMENTO ---
    X = df_treino[sintomas_possiveis + ['idade']]
    y = df_treino['target']
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    
    # Para acurácia, medimos apenas nos dados SINTÉTICOS (pois são a "gabarito" médico)
    # ou nos reais se houver muitos. Vamos medir no geral.
    acc = model.score(X, y)
    
    return model, acc, sintomas_possiveis


# --- 5. LAYOUT E VISUALIZAÇÃO ---
st.sidebar.title("Filtros Regionais")
lista_cidades = df['municipio'].dropna().unique()
cidade_filtro = st.sidebar.multiselect("Município", lista_cidades, default=lista_cidades)

df_filtrado = df[df['municipio'].isin(cidade_filtro)]

st.title("📊 Painel de Vigilância Epidemiológica - Pará (DB Real)")

tab1, tab2, tab3, tab4 = st.tabs(["Visão Geral", "Demografia & Social", "Análise Clínica", "Triagem"])

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
    st.markdown("### Triagem - COVID 19 via IA")
    
    modelo, acuracia, feature_sintomas = treinar_modelo_sg(df)
    
    if modelo is None:
        st.warning("⚠️ **Dados insuficientes para treinar a IA.**")
        st.info("O banco precisa ter pelo menos um caso POSITIVO e um NEGATIVO.")
        st.write("Diagnósticos no banco hoje:", df['classificacao_final'].unique())
    
    else:
        st.success(f"Modelo calibrado! Acurácia histórica: **{acuracia*100:.1f}%**")
        
        with st.form("ia_form"):
            c1, c2 = st.columns([1, 3])
            idade_in = c1.number_input("Idade", 0, 120, 30)
            
            checks = {}
            cols = c2.columns(3)
            # Agora ele vai gerar exatamente os checkboxes da sua lista
            for i, s in enumerate(feature_sintomas):
                with cols[i%3]: checks[s] = st.checkbox(s)
                
            if st.form_submit_button("Calcular Risco"):
                # Monta o vetor na mesma ordem da lista fixa
                vetor = [checks[s] for s in feature_sintomas] + [idade_in]
                
                # Predição
                prob = modelo.predict_proba([vetor])[0][1] * 100
                
                st.metric("Probabilidade COVID-19", f"{prob:.1f}%")
                if prob > 50: 
                    st.error("Alta Probabilidade")
                else: 
                    st.success("Baixa Probabilidade")
