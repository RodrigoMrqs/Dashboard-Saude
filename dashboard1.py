import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Painel Epidemiológico", layout="wide", page_icon="🏥")

# --- 1. DICIONÁRIO DE COORDENADAS ---
COORDS_PARA = {
    'Belém': {'lat': -1.4558, 'lon': -48.5044}, 'Ananindeua': {'lat': -1.3636, 'lon': -48.3734},
    'Santarém': {'lat': -2.4431, 'lon': -54.7083}, 'Marabá': {'lat': -5.3686, 'lon': -49.1174},
    'Parauapebas': {'lat': -6.0675, 'lon': -49.9042}, 'Castanhal': {'lat': -1.2964, 'lon': -47.9258},
    'Abaetetuba': {'lat': -1.7218, 'lon': -48.8858}, 'Cametá': {'lat': -2.2427, 'lon': -49.4965},
    'Bragança': {'lat': -1.0536, 'lon': -46.7656}, 'Altamira': {'lat': -3.2033, 'lon': -52.2025},
    'Tucuruí': {'lat': -3.7661, 'lon': -49.6722}, 'Barcarena': {'lat': -1.5058, 'lon': -48.6258},
    'Itaituba': {'lat': -4.2754, 'lon': -55.9869}, 'Redenção': {'lat': -8.0253, 'lon': -50.0317},
    'Breves': {'lat': -1.6826, 'lon': -50.4811}, 'Moju': {'lat': -1.8841, 'lon': -48.7678},
    'Novo Repartimento': {'lat': -4.2497, 'lon': -49.9482}, 'Oriximiná': {'lat': -1.7654, 'lon': -55.8661},
    'Santa Izabel do Pará': {'lat': -1.2975, 'lon': -48.1606}, 'Capanema': {'lat': -1.1969, 'lon': -47.1814}
}

# --- 2. CARREGAMENTO BLINDADO DE DADOS ---
@st.cache_data
def load_data():
    arquivo = 'datasetsindromegripal.csv'
    df = pd.DataFrame()
    
    # 1. TENTATIVA DE LEITURA (Auto-detectar separador)
    try:
        # Tenta ler com separador automático (python engine)
        df = pd.read_csv(arquivo, sep=None, engine='python', encoding='utf-8', on_bad_lines='skip')
    except:
        try:
            # Fallback para latin1 (comum no Brasil) e ponto e vírgula
            df = pd.read_csv(arquivo, sep=';', encoding='latin1', on_bad_lines='skip')
        except FileNotFoundError:
            st.error(f"❌ O arquivo '{arquivo}' não foi encontrado na pasta.")
            st.stop()
        except Exception as e:
            st.error(f"❌ Erro ao ler CSV: {e}")
            st.stop()
    
    if not df.empty:
        # 2. NORMALIZAÇÃO DAS COLUNAS (Tudo minúsculo e sem espaços)
        # Isso resolve o problema de 'dataNotificacao' vs 'DataNotificacao'
        df.columns = df.columns.str.lower().str.strip()
        
        # 3. MAPEAMENTO (Usando chaves em minúsculo)
        mapa = {
            'datanotificacao': 'data_notificacao',
            'classificacaofinal': 'resultado',
            'evolucaocaso': 'evolucao',
            'racacor': 'raca_cor',
            'codigorecebeuvacina': 'vacinado',
            'municipio': 'municipio',
            'idade': 'idade',
            'sexo': 'sexo',
            'sintomas': 'sintomas',
            'condicoes': 'comorbidades',
            'cbo': 'ocupacao',
            'codigotipoteste1': 'tipo_teste',
            'codigofabricanteteste1': 'fabricante_teste',
            'codigoresultadoteste1': 'res_teste'
        }
        
        # Renomeia apenas as que existem
        df = df.rename(columns=mapa)

        # 4. DEBUG DE SEGURANÇA (Se a coluna principal sumiu, avisa o usuário)
        if 'data_notificacao' not in df.columns:
            st.error("⚠️ Erro de Colunas: O sistema não encontrou a coluna de Data.")
            st.write("Colunas encontradas no seu arquivo:", df.columns.tolist())
            st.stop()

        # 5. TRATAMENTO DE DADOS
        # Datas
        df['data_notificacao'] = pd.to_datetime(df['data_notificacao'], errors='coerce')
        
        # Idade
        if 'idade' in df.columns:
            # Remove qualquer coisa que não seja número e converte
            df['idade'] = pd.to_numeric(df['idade'].astype(str).str.replace(r'\D', '', regex=True), errors='coerce').fillna(0)
            df['faixa_etaria'] = pd.cut(df['idade'], bins=[-1, 12, 19, 59, 120], labels=['0-12', '13-19', '20-59', '60+'])
            
        # Coordenadas
        if 'municipio' in df.columns:
            df['lat'] = df['municipio'].map(lambda x: COORDS_PARA.get(x, {}).get('lat', None))
            df['lon'] = df['municipio'].map(lambda x: COORDS_PARA.get(x, {}).get('lon', None))
            
        # Normalização de Texto
        for col in ['resultado', 'vacinado', 'res_teste', 'sexo']:
            if col in df.columns: 
                df[col] = df[col].astype(str).str.upper().str.strip()

        # Nulos
        if 'sintomas' in df.columns: df['sintomas'] = df['sintomas'].fillna('Assintomático')
        if 'ocupacao' in df.columns: df['ocupacao'] = df['ocupacao'].fillna('Não Informado')
        
    return df

# Carrega os dados
df = load_data()

# --- 3. IA PREDITIVA ---
@st.cache_resource
def treinar_modelo(df):
    df_mod = df.copy()
    
    # Target
    df_mod['target'] = df_mod['resultado'].apply(lambda x: 1 if 'COVID' in str(x) or 'POSITIVO' in str(x) or 'CONFIRMADO' in str(x) else 0)
    
    # Features
    sintomas_list = ['Febre', 'Tosse', 'Dor de Garganta', 'Dispneia', 'Dor de Cabeça', 'Perda de Olfato', 'Mialgia', 'Coriza', 'Fadiga']
    
    # Garante que as colunas existem antes de tentar acessar
    if 'sintomas' not in df_mod.columns: return None, 0, []

    for s in sintomas_list:
        termo = 'DOR' if s == 'Mialgia' else s
        termo = 'OLFATO' if s == 'Olfato' else termo.upper()
        df_mod[s] = df_mod['sintomas'].astype(str).str.upper().apply(lambda x: 1 if termo.upper() in x else 0)
    
    # Vacina e Sexo
    df_mod['vacina_feat'] = df_mod['vacinado'].apply(lambda x: 1 if 'SIM' in str(x) or '1' in str(x) else 0) if 'vacinado' in df_mod.columns else 0
    df_mod['sexo_feat'] = df_mod['sexo'].apply(lambda x: 1 if str(x).startswith('M') else 0) if 'sexo' in df_mod.columns else 0
    
    features = sintomas_list + ['idade', 'vacina_feat', 'sexo_feat']
    
    # Sintéticos para Robustez
    dados_sint = []
    pesos = {'Perda de Olfato': 80, 'Dispneia': 60, 'Febre': 40, 'Dor de Garganta': 30, 'Tosse': 30, 'Dor de Cabeça': 20, 'Mialgia': 20, 'Coriza': 10, 'Fadiga': 10}
    import numpy as np
    np.random.seed(42)
    
    for _ in range(300):
        p = {'idade': np.random.randint(10, 90), 'vacina_feat': np.random.choice([0,1]), 'sexo_feat': np.random.choice([0,1])}
        score = 0
        for s in sintomas_list:
            tem = np.random.choice([0, 1], p=[0.8, 0.2])
            p[s] = tem
            if tem: score += pesos.get(s, 10)
        prob = 1 / (1 + np.exp(-(score - 40) / 20))
        p['target'] = np.random.choice([1, 0], p=[prob, 1-prob])
        dados_sint.append(p)
        
    df_final = pd.concat([df_mod[features + ['target']].dropna(), pd.DataFrame(dados_sint)])
    
    X = df_final[features]
    y = df_final['target']
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model, model.score(X, y), sintomas_list

# --- 4. LAYOUT ---
st.sidebar.title("Filtros")
if 'municipio' in df.columns:
    mun_sel = st.sidebar.multiselect("Município", sorted(df['municipio'].dropna().unique()), default=[])
    df_filtro = df[df['municipio'].isin(mun_sel)] if mun_sel else df
else:
    df_filtro = df

st.title("Painel de Vigilância Epidemiológico ")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Evolução & Mapa", "Demografia", "Vacinas & Testes", "Estatística", "Triagem IA"])

# === TAB 1: GERAL ===
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Notificações", len(df_filtro))
    
    # Verifica colunas antes de calcular
    res_col = 'resultado' if 'resultado' in df_filtro.columns else None
    ev_col = 'evolucao' if 'evolucao' in df_filtro.columns else None
    
    if res_col:
        conf = len(df_filtro[df_filtro[res_col].str.contains('COVID|POSITIVO|CONFIRMADO', na=False)])
        flu = len(df_filtro[df_filtro[res_col].str.contains('INFLUENZA|GRIPAL', na=False)])
    else: conf, flu = 0, 0
    
    if ev_col:
        obitos = len(df_filtro[df_filtro['evolucao'].astype(str).str.contains('Óbito', case=False, na=False)])
    else: obitos = 0
    
    c2.metric("Confirmados", conf)
    c3.metric("Gripe/Influenza", flu)
    c4.metric("Óbitos", obitos, delta_color="inverse")
    
    col_A, col_B = st.columns([2, 1])
    with col_A:
        st.markdown("### Mapa de Calor - Notificações")
        if 'lat' in df_filtro.columns:
            map_data = df_filtro.dropna(subset=['lat', 'lon'])
            if not map_data.empty: st.map(map_data, latitude='lat', longitude='lon', size=20, color='#ff4b4b')
            else: st.warning("Sem coordenadas para os filtros atuais.")
            
    with col_B:
        if 'data_notificacao' in df_filtro.columns and res_col:
            # Agrupa e Conta
            df_t = df_filtro.groupby([pd.Grouper(key='data_notificacao', freq='W'), res_col]).size().reset_index(name='Casos')
            # Filtra top 5 resultados para não poluir
            top_res = df_t.groupby(res_col)['Casos'].sum().nlargest(5).index
            fig = px.line(df_t[df_t[res_col].isin(top_res)], x='data_notificacao', y='Casos', color=res_col, title="Curva Epidemiológica")
            st.plotly_chart(fig, use_container_width=True)

# === TAB 2: DEMOGRAFIA ===
with tab2:
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("**Distribuição Sexo e Faixa Etária**")
        if 'sexo' in df_filtro.columns and 'faixa_etaria' in df_filtro.columns:
            df_sun = df_filtro.dropna(subset=['sexo', 'faixa_etaria'])
            if not df_sun.empty:
                st.plotly_chart(px.sunburst(df_sun, path=['sexo', 'faixa_etaria']), use_container_width=True)
    
    with col_d2:
        st.markdown("**Raça/Cor**")
        if 'raca_cor' in df_filtro.columns:
            st.plotly_chart(px.bar(df_filtro['raca_cor'].value_counts().reset_index(), x='count', y='raca_cor', orientation='h'), use_container_width=True)

    st.divider()
    st.markdown("### Ocupações (CBO) Mais Afetadas")
    if 'ocupacao' in df_filtro.columns:
        top_cbo = df_filtro['ocupacao'].value_counts().head(10).reset_index()
        st.plotly_chart(px.bar(top_cbo, x='count', y='ocupacao', orientation='h', color='count'), use_container_width=True)

# === TAB 3: VACINAS ===
with tab3:
    c_vac, c_test = st.columns(2)
    
    with c_vac:
        st.subheader("Vacinação x Resultado Laboratorial")
        if 'vacinado' in df_filtro.columns and 'resultado' in df_filtro.columns:
            df_v = df_filtro.copy()
            df_v['status_vac'] = df_v['vacinado'].apply(lambda x: 'Vacinado' if 'SIM' in x or '1' in x else 'Não Vacinado')
            df_v['status_pos'] = df_v['resultado'].apply(lambda x: 'Positivo' if 'COVID' in x or 'POSITIVO' in x else 'Negativo')
            
            cross = pd.crosstab(df_v['status_vac'], df_v['status_pos'], normalize='index') * 100
            st.plotly_chart(px.bar(cross.reset_index(), x='status_vac', y=['Positivo', 'Negativo'], 
                                   title="Taxa de Positividade (%)"), use_container_width=True)
            
    with c_test:
        st.subheader("Tipos de Testes Realizados")
        if 'tipo_teste' in df_filtro.columns:
            top_testes = df_filtro['tipo_teste'].value_counts().head(5).reset_index()
            st.plotly_chart(px.pie(top_testes, names='tipo_teste', values='count', hole=0.4), use_container_width=True)

    st.markdown("### Eficácia por Fabricante de Teste")
    if 'fabricante_teste' in df_filtro.columns and 'res_teste' in df_filtro.columns:
        # Filtra apenas quem tem fabricante informado
        df_fab = df_filtro[~df_filtro['fabricante_teste'].isin(['NAN', 'NONE', 'nan'])]
        # Cria flag binária de positivo
        df_fab['is_pos'] = df_fab['res_teste'].apply(lambda x: 1 if 'REAGENTE' in x or 'POSITIVO' in x or 'DETECTAVEL' in x else 0)
        
        # Calcula média de positividade
        stats_fab = df_fab.groupby('fabricante_teste')['is_pos'].mean().nlargest(10).reset_index()
        stats_fab['is_pos'] = stats_fab['is_pos'] * 100 # vira porcentagem
        
        st.plotly_chart(px.bar(stats_fab, x='fabricante_teste', y='is_pos', title="Taxa de Positividade por Fabricante (%)"), use_container_width=True)
# === TAB 4: ESTATÍSTICA ===
with tab4:
    st.header("Estatísticas")
    if 'resultado' in df_filtro.columns:
        df_c = df_filtro.copy()
        df_c['TARGET'] = df_c['resultado'].apply(lambda x: 1 if 'POSITIVO' in x or 'COVID' in x else 0)
        sints = ['Febre', 'Tosse', 'Dispneia']
        if 'sintomas' in df_c.columns:
            for s in sints: df_c[s] = df_c['sintomas'].astype(str).str.upper().apply(lambda x: 1 if s.upper() in x else 0)
            corr = df_c[['TARGET', 'idade'] + sints].corr()
            st.plotly_chart(px.imshow(corr, text_auto=True), use_container_width=True)

# === TAB 5: IA ===
with tab5:
    st.header("Triagem Preditiva de COVID-19")
    modelo, acc, feats = treinar_modelo(df)
    if modelo:
        st.success(f"Acurácia: {acc*100:.1f}%")
        c1, c2 = st.columns([1, 2])
        with c1:
            idd = st.number_input("Idade", 0, 100, 30)
            vac = st.selectbox("Vacinado?", ["Sim", "Não"])
            sx = st.selectbox("Sexo", ["M", "F"])
        with c2:
            chks = {f: st.checkbox(f) for f in feats}
        
        if st.button("Calcular"):
            vec = [int(chks[f]) for f in feats] + [idd, 1 if vac=="Sim" else 0, 1 if sx=="M" else 0]
            prob = modelo.predict_proba([vec])[0][1] * 100
            st.metric("Probabilidade COVID", f"{prob:.1f}%")
