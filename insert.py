"""
Script para gerar INSERTs seguindo a Terceira Forma Normal (3FN)
Sem atributos multivalorados - cada sintoma/condição é um registro separado
Ordem: das pontas (tabelas independentes) para o centro (tabelas dependentes)
"""
import sys
import io
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
import os
from dotenv import load_dotenv
from tqdm import tqdm
from datetime import datetime

# Configurar encoding UTF-8 para Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

load_dotenv()

def connect_db():
    """Conectar ao PostgreSQL"""
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            database=os.getenv('DB_NAME', 'trabalho banco'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', 'rmap201bl1'),
            port=os.getenv('DB_PORT', '5432')
        )
        print("✅ Conectado ao PostgreSQL")
        return conn
    except Exception as e:
        print(f"❌ Erro ao conectar: {e}")
        return None

def parse_date(date_str):
    if pd.isna(date_str) or date_str in (None, "", "None"):
        return None

    s = str(date_str).strip()

    # Tenta formato ISO (AAAA-MM-DD)
    try:
        return datetime.strptime(s, '%Y-%m-%d').date()
    except:
        pass

    # Tenta formato padrão brasileiro (DD/MM/AAAA)
    try:
        return datetime.strptime(s, '%d/%m/%Y').date()
    except:
        pass

    return None


def parse_boolean(value):
    """Converte string para boolean"""
    if pd.isna(value):
        return False
    return str(value).upper() == 'VERDADEIRO'

# ============================================================================
# NÍVEL 1: TABELAS INDEPENDENTES (PONTAS DO GRAFO)
# ============================================================================

def insert_sexo(conn):
    """1. Inserir sexo (independente)"""
    print("\n1️⃣  SEXO (tabela independente)")
    cursor = conn.cursor()
    
    sexos = [('Masculino',), ('Feminino',), ('Ignorado',)]
    execute_batch(cursor, 
        "INSERT INTO sexo (descricao) VALUES (%s) ON CONFLICT (descricao) DO NOTHING",
        sexos
    )
    conn.commit()
    
    cursor.execute("SELECT id, descricao FROM sexo")
    result = {row[1]: row[0] for row in cursor.fetchall()}
    cursor.close()
    print(f"   ✅ {len(result)} registros")
    return result

def insert_raca_cor(conn, df):
    """2. Inserir raça/cor (independente) - PRESERVA NULOS"""
    print("\n2️⃣  RAÇA/COR (tabela independente)")
    cursor = conn.cursor()
    
    # Incluir valores únicos E valores nulos
    racas = df['racaCor'].unique()  # NÃO remove NaN
    for raca in racas:
        if pd.notna(raca):  # Só insere valores não-nulos na tabela
            cursor.execute(
                "INSERT INTO raca_cor (descricao) VALUES (%s) ON CONFLICT (descricao) DO NOTHING",
                (raca,)
            )
    conn.commit()
    
    cursor.execute("SELECT id, descricao FROM raca_cor")
    result = {row[1]: row[0] for row in cursor.fetchall()}
    result[None] = None  # Mapear None para None (permite FK nulo)
    cursor.close()
    print(f"   ✅ {len(result)-1} registros (+ valores nulos preservados)")
    return result

def insert_profissional_saude(conn):
    """3. Inserir profissional saúde (independente)"""
    print("\n3️⃣  PROFISSIONAL SAÚDE (tabela independente)")
    cursor = conn.cursor()
    
    valores = [('Sim',), ('Não',)]
    execute_batch(cursor,
        "INSERT INTO profissional_saude (descricao) VALUES (%s) ON CONFLICT (descricao) DO NOTHING",
        valores
    )
    conn.commit()
    
    cursor.execute("SELECT id, descricao FROM profissional_saude")
    result = {row[1]: row[0] for row in cursor.fetchall()}
    cursor.close()
    print(f"   ✅ {len(result)} registros")
    return result

def insert_profissional_seguranca(conn):
    """4. Inserir profissional segurança (independente)"""
    print("\n4️⃣  PROFISSIONAL SEGURANÇA (tabela independente)")
    cursor = conn.cursor()
    
    valores = [('Sim',), ('Não',)]
    execute_batch(cursor,
        "INSERT INTO profissional_seguranca (descricao) VALUES (%s) ON CONFLICT (descricao) DO NOTHING",
        valores
    )
    conn.commit()
    
    cursor.execute("SELECT id, descricao FROM profissional_seguranca")
    result = {row[1]: row[0] for row in cursor.fetchall()}
    cursor.close()
    print(f"   ✅ {len(result)} registros")
    return result

def insert_sintomas(conn, df):
    """5. Inserir sintomas (independente) - SEM MULTIVALORAÇÃO - PRESERVA NULOS"""
    print("\n5️⃣  SINTOMAS (tabela independente - atomizados)")
    cursor = conn.cursor()
    
    sintomas_set = set()
    # Processar TODOS os registros, incluindo os com sintomas nulos
    for sintomas_str in df['sintomas']:
        if pd.notna(sintomas_str):  # Só processa se não for nulo
            # Separar sintomas individuais
            for sintoma in str(sintomas_str).split(','):
                sintoma = sintoma.strip()
                if sintoma and sintoma != '':
                    sintomas_set.add(sintoma)
    
    for sintoma in sintomas_set:
        cursor.execute(
            "INSERT INTO sintoma (descricao) VALUES (%s) ON CONFLICT (descricao) DO NOTHING",
            (sintoma,)
        )
    conn.commit()
    
    cursor.execute("SELECT id, descricao FROM sintoma")
    result = {row[1]: row[0] for row in cursor.fetchall()}
    result[None] = None  # Permite sintomas nulos
    cursor.close()
    print(f"   ✅ {len(result)-1} sintomas únicos (+ nulos preservados)")
    return result

def insert_outro_sintoma(conn, df):
    """6. Inserir outros sintomas (independente) - PRESERVA NULOS"""
    print("\n6️⃣  OUTROS SINTOMAS (tabela independente)")
    cursor = conn.cursor()
    
    outros_set = set()
    for outros_str in df['outrosSintomas']:
        if pd.notna(outros_str):  # Só processa se não for nulo
            outros_str = str(outros_str).strip()
            if outros_str and outros_str != '':
                # Separar por vírgula se houver múltiplos
                for outro in outros_str.split(','):
                    outro = outro.strip()
                    if outro:
                        outros_set.add(outro)
    
    for outro in outros_set:
        cursor.execute(
            "INSERT INTO outro_sintoma (descricao) VALUES (%s) ON CONFLICT (descricao) DO NOTHING",
            (outro,)
        )
    conn.commit()
    
    cursor.execute("SELECT id, descricao FROM outro_sintoma")
    result = {row[1]: row[0] for row in cursor.fetchall()}
    result[None] = None  # Permite nulos
    cursor.close()
    print(f"   ✅ {len(result)-1} registros (+ nulos preservados)")
    return result

def insert_condicoes(conn, df):
    """7. Inserir condições (independente) - SEM MULTIVALORAÇÃO - PRESERVA NULOS"""
    print("\n7️⃣  CONDIÇÕES (tabela independente - atomizadas)")
    cursor = conn.cursor()
    
    condicoes_set = set()
    for condicoes_str in df['condicoes']:
        if pd.notna(condicoes_str):  # Só processa se não for nulo
            # Separar condições individuais
            for condicao in str(condicoes_str).split(','):
                condicao = condicao.strip()
                if condicao and condicao != '':
                    condicoes_set.add(condicao)
    
    for condicao in condicoes_set:
        cursor.execute(
            "INSERT INTO condicoes (descricao) VALUES (%s) ON CONFLICT (descricao) DO NOTHING",
            (condicao,)
        )
    conn.commit()
    
    cursor.execute("SELECT id, descricao FROM condicoes")
    result = {row[1]: row[0] for row in cursor.fetchall()}
    result[None] = None  # Permite nulos
    cursor.close()
    print(f"   ✅ {len(result)-1} condições únicas (+ nulos preservados)")
    return result

def insert_outras_condicoes(conn, df):
    """8. Inserir outras condições (independente) - PRESERVA NULOS"""
    print("\n8️⃣  OUTRAS CONDIÇÕES (tabela independente)")
    cursor = conn.cursor()
    
    outras_set = set()
    for outras_str in df['outrasCondicoes']:
        if pd.notna(outras_str):  # Só processa se não for nulo
            outras_str = str(outras_str).strip()
            if outras_str and outras_str != '':
                # Separar por vírgula se houver múltiplas
                for outra in outras_str.split(','):
                    outra = outra.strip()
                    if outra:
                        outras_set.add(outra)
    
    for outra in outras_set:
        cursor.execute(
            "INSERT INTO outras_condicoes (descricao) VALUES (%s) ON CONFLICT (descricao) DO NOTHING",
            (outra,)
        )
    conn.commit()
    
    cursor.execute("SELECT id, descricao FROM outras_condicoes")
    result = {row[1]: row[0] for row in cursor.fetchall()}
    result[None] = None  # Permite nulos
    cursor.close()
    print(f"   ✅ {len(result)-1} registros (+ nulos preservados)")
    return result

def insert_estados(conn, df):
    """9. Inserir estados (independente)"""
    print("\n9️⃣  ESTADOS (tabela independente)")
    cursor = conn.cursor()
    
    # Agrupar dados únicos de estados
    estados_df = df[['estado', 'estadoIBGE', 'estadoNotificacao', 'estadoNotificacaoIBGE']].drop_duplicates()
    estados_data = []
    
    for _, row in estados_df.iterrows():
        if pd.notna(row['estadoIBGE']):
            estados_data.append((
                row['estado'] if pd.notna(row['estado']) else None,
                row['estadoIBGE'],
                row['estadoNotificacao'] if pd.notna(row['estadoNotificacao']) else None,
                str(row['estadoNotificacaoIBGE']) if pd.notna(row['estadoNotificacaoIBGE']) else None,
                row['estadoIBGE']  # sigla
            ))
    
    execute_batch(cursor, """
        INSERT INTO estado (nome, estado_ibge, estado_notificacao, estado_notificacao_ibge, sigla)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (sigla) DO NOTHING
    """, estados_data, page_size=100)
    conn.commit()
    
    cursor.execute("SELECT id, sigla FROM estado")
    result = {row[1]: row[0] for row in cursor.fetchall()}
    cursor.close()
    print(f"   ✅ {len(result)} registros")
    return result

def insert_municipios(conn, df):
    print("\n🔟 MUNICÍPIOS (tabela independente)")
    cursor = conn.cursor()

    municipios_df = df[['municipioNotificacao', 'municipioNotificacaoIBGE']].drop_duplicates()
    municipios_data = []

    for _, row in municipios_df.iterrows():
        if pd.notna(row['municipioNotificacaoIBGE']):
            try:
                municipios_data.append((
                    row['municipioNotificacao'] if pd.notna(row['municipioNotificacao']) else None,
                    int(row['municipioNotificacaoIBGE'])
                ))
            except:
                pass

    execute_batch(cursor, """
        INSERT INTO municipio (nome, municipio_ibge)
        VALUES (%s, %s)
        ON CONFLICT (municipio_ibge) DO NOTHING
    """, municipios_data, page_size=100)

    conn.commit()

    cursor.execute("SELECT id, municipio_ibge FROM municipio")
    result = {row[1]: row[0] for row in cursor.fetchall()}

    cursor.close()
    print(f"   ✅ {len(result)} registros")
    return result


def insert_busca_ativa(conn, df):
    """11. Inserir busca ativa (independente)"""
    print("\n1️⃣1️⃣  BUSCA ATIVA (tabela independente)")
    cursor = conn.cursor()
    
    busca_df = df[['codigoEstrategiaCovid', 'codigoBuscaAtivaAssintomatico', 
                   'outroBuscaAtivaAssintomatico', 'codigoContemComunidadeTradicional']].drop_duplicates()
    
    busca_data = []
    for _, row in busca_df.iterrows():
        try:
            busca_data.append((
                int(row['codigoEstrategiaCovid']) if pd.notna(row['codigoEstrategiaCovid']) else None,
                int(row['codigoBuscaAtivaAssintomatico']) if pd.notna(row['codigoBuscaAtivaAssintomatico']) else None,
                int(row['outroBuscaAtivaAssintomatico']) if pd.notna(row['outroBuscaAtivaAssintomatico']) else None,
                int(row['codigoContemComunidadeTradicional']) if pd.notna(row['codigoContemComunidadeTradicional']) else None
            ))
        except:
            pass
    
    if busca_data:
        execute_batch(cursor, """
            INSERT INTO busca_ativa (codigo_Estrategia_Covid, codigo_BuscaAtiva_Assintomatico,
                                    outro_Busca_Ativa_Assintomatico, codigo_Contem_Comunidade_Tradicional)
            VALUES (%s, %s, %s, %s)
        """, busca_data, page_size=100)
        conn.commit()
    
    cursor.execute("SELECT id, codigo_Estrategia_Covid, codigo_BuscaAtiva_Assintomatico, outro_Busca_Ativa_Assintomatico, codigo_Contem_Comunidade_Tradicional FROM busca_ativa")
    result = {}
    for row in cursor.fetchall():
        key = (row[1], row[2], row[3], row[4])
        result[key] = row[0]
    cursor.close()
    print(f"   ✅ {len(result)} registros")
    return result

# ============================================================================
# NÍVEL 2: TABELAS QUE DEPENDEM DO NÍVEL 1
# ============================================================================

def insert_pessoas(conn, df, sexo_map, raca_map, prof_saude_map, prof_seg_map):
    """12. Inserir pessoas (depende de: sexo, raca_cor, profissional_*)"""
    print("\n1️⃣2️⃣  PESSOAS (depende de: sexo, raça/cor, profissionais)")
    cursor = conn.cursor()
    pessoas_data = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="   Preparando"):
        sexo_id = sexo_map.get(row['sexo']) if pd.notna(row['sexo']) else None
        raca_id = raca_map.get(row['racaCor']) if pd.notna(row['racaCor']) else None
        prof_saude_id = prof_saude_map.get(row['profissionalSaude']) if pd.notna(row['profissionalSaude']) else None
        prof_seg_id = prof_seg_map.get(row['profissionalSeguranca']) if pd.notna(row['profissionalSeguranca']) else None
        
        pessoas_data.append((sexo_id, raca_id, prof_saude_id, prof_seg_id))
    
    execute_batch(cursor, """
        INSERT INTO pessoa (sexo_id, raca_cor_id, profissional_saude_id, profissional_seguranca_id)
        VALUES (%s, %s, %s, %s)
    """, pessoas_data, page_size=1000)
    conn.commit()
    
    cursor.execute("SELECT id FROM pessoa ORDER BY id")
    result = [row[0] for row in cursor.fetchall()]
    cursor.close()
    print(f"   ✅ {len(result)} registros")
    return result

# ============================================================================
# NÍVEL 3: TABELAS QUE DEPENDEM DO NÍVEL 2
# ============================================================================

def insert_notificacoes(conn, df, estado_map, pessoa_ids):
    """13. Inserir notificações (depende de: estado, pessoa)"""
    print("\n1️⃣3️⃣  NOTIFICAÇÕES (depende de: estado, pessoa)")
    cursor = conn.cursor()
    notif_data = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="   Preparando"):
        estado_id = estado_map.get(row['estadoIBGE']) if pd.notna(row['estadoIBGE']) else None
        pessoa_id = pessoa_ids[idx] if idx < len(pessoa_ids) else None
        
        notif_data.append((
            estado_id,
            pessoa_id,
            row['cbo'] if pd.notna(row['cbo']) else None,
            row['origem'] if pd.notna(row['origem']) else None,
            row['evolucaoCaso'] if pd.notna(row['evolucaoCaso']) else None,
            row['classificacaoFinal'] if pd.notna(row['classificacaoFinal']) else None,
            int(row['totalTestesRealizados']) if pd.notna(row['totalTestesRealizados']) else 0,
            parse_date(row['dataNotificacao']),
            parse_date(row['dataInicioSintomas']),
            parse_date(row['dataEncerramento']),
            int(row['idade']) if pd.notna(row['idade']) else None,
            parse_boolean(row['excluido']),
            parse_boolean(row['validado'])
        ))
    
    execute_batch(cursor, """
        INSERT INTO notificacao (estado_id, pessoa_id, cbo, origem, evolucaoCaso, classificacaoFinal,
                                total_testes_realizados, data_notificacao, data_inicio_sintomas,
                                data_encerramento, idade, excluido, validado)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, notif_data, page_size=1000)
    conn.commit()
    
    cursor.execute("SELECT id FROM notificacao ORDER BY id")
    result = [row[0] for row in cursor.fetchall()]
    cursor.close()
    print(f"   ✅ {len(result)} registros")
    return result

# ============================================================================
# NÍVEL 4: TABELAS DE RELACIONAMENTO E COMPLEMENTARES
# ============================================================================

def insert_notificacao_sintomas(conn, df, notif_ids, sintoma_map, outro_sintoma_map):
    """14. Inserir notificação_sintoma (N:M) - ATOMIZADO"""
    print("\n1️⃣4️⃣  NOTIFICAÇÃO_SINTOMA (relacionamento N:M - atomizado)")
    cursor = conn.cursor()
    relacoes = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="   Processando"):
        if idx >= len(notif_ids):
            continue
        
        notif_id = notif_ids[idx]
        
        # Sintomas principais - CADA SINTOMA É UM REGISTRO
        if pd.notna(row['sintomas']):
            for sintoma in str(row['sintomas']).split(','):
                sintoma = sintoma.strip()
                sintoma_id = sintoma_map.get(sintoma)
                if sintoma_id:
                    relacoes.append((notif_id, None, sintoma_id))
        
        # Outros sintomas - CADA UM É UM REGISTRO
        if pd.notna(row['outrosSintomas']) and str(row['outrosSintomas']).strip() != '':
            for outro in str(row['outrosSintomas']).split(','):
                outro = outro.strip()
                outro_id = outro_sintoma_map.get(outro)
                if outro_id:
                    relacoes.append((notif_id, outro_id, None))
    
    if relacoes:
        execute_batch(cursor, """
            INSERT INTO notificacao_sintoma (notificacao_id, outro_sintoma_id, sintoma_id)
            VALUES (%s, %s, %s)
            ON CONFLICT DO NOTHING
        """, relacoes, page_size=1000)
        conn.commit()
    
    cursor.close()
    print(f"   ✅ {len(relacoes)} relações (atomizadas)")

def insert_notificacao_condicoes(conn, df, notif_ids, condicao_map, outra_condicao_map):
    """15. Inserir notificacao_condicao (N:M) - ATOMIZADO"""
    print("\n1️⃣5️⃣  NOTIFICAÇÃO_CONDIÇÃO (relacionamento N:M - atomizado)")
    cursor = conn.cursor()
    relacoes = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="   Processando"):
        if idx >= len(notif_ids):
            continue
        
        notif_id = notif_ids[idx]
        
        # Condições principais - CADA CONDIÇÃO É UM REGISTRO
        if pd.notna(row['condicoes']):
            for condicao in str(row['condicoes']).split(','):
                condicao = condicao.strip()
                condicao_id = condicao_map.get(condicao)
                if condicao_id:
                    relacoes.append((notif_id, None, condicao_id))
        
        # Outras condições - CADA UMA É UM REGISTRO
        if pd.notna(row['outrasCondicoes']) and str(row['outrasCondicoes']).strip() != '':
            for outra in str(row['outrasCondicoes']).split(','):
                outra = outra.strip()
                outra_id = outra_condicao_map.get(outra)
                if outra_id:
                    relacoes.append((notif_id, outra_id, None))
    
    if relacoes:
        execute_batch(cursor, """
            INSERT INTO notificacao_condicao (notificacao_id, outras_condicoes_id, condicoes_id)
            VALUES (%s, %s, %s)
            ON CONFLICT DO NOTHING
        """, relacoes, page_size=1000)
        conn.commit()
    
    cursor.close()
    print(f"   ✅ {len(relacoes)} relações (atomizadas)")

def insert_notificacao_municipio(conn, df, notif_ids, municipio_map):
    """16. Inserir notificacao_municipio (N:M)"""
    print("\n1️⃣6️⃣  NOTIFICAÇÃO_MUNICÍPIO (relacionamento N:M)")
    cursor = conn.cursor()
    relacoes = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="   Processando"):
        if idx >= len(notif_ids):
            continue
        
        notif_id = notif_ids[idx]
        
        if pd.notna(row['municipioNotificacaoIBGE']):
            try:
                mun_ibge = int(row['municipioNotificacaoIBGE'])
                mun_id = municipio_map.get(mun_ibge)
                if mun_id:
                    relacoes.append((notif_id, mun_id))
            except:
                pass
    
    if relacoes:
        execute_batch(cursor, """
            INSERT INTO notificacao_municipio (notificacao_id, municipio_id)
            VALUES (%s, %s)
            ON CONFLICT DO NOTHING
        """, relacoes, page_size=1000)
        conn.commit()
    
    cursor.close()
    print(f"   ✅ {len(relacoes)} relações")

def insert_notificacao_busca_ativa(conn, df, notif_ids, busca_map):
    """17. Inserir notificacao_busca_ativa (N:M)"""
    print("\n1️⃣7️⃣  NOTIFICAÇÃO_BUSCA_ATIVA (relacionamento N:M)")
    cursor = conn.cursor()
    relacoes = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="   Processando"):
        if idx >= len(notif_ids):
            continue
        
        notif_id = notif_ids[idx]
        
        try:
            key = (
                int(row['codigoEstrategiaCovid']) if pd.notna(row['codigoEstrategiaCovid']) else None,
                int(row['codigoBuscaAtivaAssintomatico']) if pd.notna(row['codigoBuscaAtivaAssintomatico']) else None,
                int(row['outroBuscaAtivaAssintomatico']) if pd.notna(row['outroBuscaAtivaAssintomatico']) else None,
                int(row['codigoContemComunidadeTradicional']) if pd.notna(row['codigoContemComunidadeTradicional']) else None
            )
            busca_id = busca_map.get(key)
            if busca_id:
                relacoes.append((notif_id, busca_id))
        except:
            pass
    
    if relacoes:
        execute_batch(cursor, """
            INSERT INTO notificacao_busca_ativa (notificacao_id, busca_ativa_id)
            VALUES (%s, %s)
            ON CONFLICT DO NOTHING
        """, relacoes, page_size=1000)
        conn.commit()
    
    cursor.close()
    print(f"   ✅ {len(relacoes)} relações")

def insert_testes(conn, df, notif_ids):
    """18. Inserir testes (depende de: notificacao)"""
    print("\n1️⃣8️⃣  TESTES (depende de: notificação)")
    cursor = conn.cursor()
    testes_data = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="   Processando"):
        if idx >= len(notif_ids):
            continue
        
        notif_id = notif_ids[idx]
        
        # Teste 1
        if pd.notna(row['codigoEstadoTeste1']):
            testes_data.append((
                notif_id, 1,
                row['codigoEstadoTeste1'],
                row['codigoTipoTeste1'] if pd.notna(row['codigoTipoTeste1']) else None,
                row['codigoFabricanteTeste1'] if pd.notna(row['codigoFabricanteTeste1']) else None,
                row['codigoResultadoTeste1'] if pd.notna(row['codigoResultadoTeste1']) else None,
                parse_date(row['dataColetaTeste1'])
            ))
        
        # Teste 2
        if pd.notna(row['codigoEstadoTeste2']):
            testes_data.append((
                notif_id, 2,
                row['codigoEstadoTeste2'],
                row['codigoTipoTeste2'] if pd.notna(row['codigoTipoTeste2']) else None,
                row['codigoFabricanteTeste2'] if pd.notna(row['codigoFabricanteTeste2']) else None,
                row['codigoResultadoTeste2'] if pd.notna(row['codigoResultadoTeste2']) else None,
                parse_date(row['dataColetaTeste2'])
            ))
        
        # Teste 3
        if pd.notna(row['codigoEstadoTeste3']):
            testes_data.append((
                notif_id, 3,
                row['codigoEstadoTeste3'],
                row['codigoTipoTeste3'] if pd.notna(row['codigoTipoTeste3']) else None,
                row['codigoFabricanteTeste3'] if pd.notna(row['codigoFabricanteTeste3']) else None,
                row['codigoResultadoTeste3'] if pd.notna(row['codigoResultadoTeste3']) else None,
                parse_date(row['dataColetaTeste3'])
            ))
        
        # Teste 4
        if pd.notna(row['codigoEstadoTeste4']):
            testes_data.append((
                notif_id, 4,
                row['codigoEstadoTeste4'],
                row['codigoTipoTeste4'] if pd.notna(row['codigoTipoTeste4']) else None,
                row['codigoFabricanteTeste4'] if pd.notna(row['codigoFabricanteTeste4']) else None,
                row['codigoResultadoTeste4'] if pd.notna(row['codigoResultadoTeste4']) else None,
                parse_date(row['dataColetaTeste4'])
            ))
    
    if testes_data:
        execute_batch(cursor, """
            INSERT INTO teste (notificacao_id, ordem, codigo_estado, codigo_tipo, codigo_fabricante,
                             codigo_resultado, data_coleta)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, testes_data, page_size=1000)
        conn.commit()
    
    cursor.close()
    print(f"   ✅ {len(testes_data)} registros")

def insert_vacinacao(conn, df, notif_ids):
    """19. Inserir vacinação (depende de: notificacao)"""
    print("\n1️⃣9️⃣  VACINAÇÃO (depende de: notificação)")
    cursor = conn.cursor()
    vac_data = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="   Processando"):
        if idx >= len(notif_ids):
            continue
        
        notif_id = notif_ids[idx]
        
        if pd.notna(row['codigoRecebeuVacina']):
            vac_data.append((
                notif_id,
                row['codigoRecebeuVacina'],
                row['codigoLaboratorioPrimeiraDose'] if pd.notna(row['codigoLaboratorioPrimeiraDose']) else None,
                row['codigoLaboratorioSegundaDose'] if pd.notna(row['codigoLaboratorioSegundaDose']) else None,
                row['lotePrimeiraDose'] if pd.notna(row['lotePrimeiraDose']) else None,
                row['loteSegundaDose'] if pd.notna(row['loteSegundaDose']) else None,
                row['codigoDosesVacina'] if pd.notna(row['codigoDosesVacina']) else None,
                parse_date(row['dataPrimeiraDose']),
                parse_date(row['dataSegundaDose'])
            ))
    
    if vac_data:
        execute_batch(cursor, """
            INSERT INTO vacinacao (notificacao_id, codigo_recebeu_vacina, codigo_lab_primeira_dose,
                                  codigo_lab_segunda_dose, lote_primeira_dose, lote_segunda_dose,
                                  codigo_doses_vacina, data_primeira_dose, data_segunda_dose)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, vac_data, page_size=1000)
        conn.commit()
    
    cursor.close()
    print(f"   ✅ {len(vac_data)} registros")

def insert_triagem_populacao(conn, df, notif_ids):
    """20. Inserir triagem_populacao (depende de: notificacao)"""
    print("\n2️⃣0️⃣  TRIAGEM POPULAÇÃO (depende de: notificação)")
    cursor = conn.cursor()
    triagem_data = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="   Processando"):
        if idx >= len(notif_ids):
            continue
        
        notif_id = notif_ids[idx]
        
        if pd.notna(row['codigoTriagemPopulacaoEspecifica']):
            triagem_data.append((
                notif_id,
                row['codigoTriagemPopulacaoEspecifica'],
                row['outroTriagemPopulacaoEspecifica'] if pd.notna(row['outroTriagemPopulacaoEspecifica']) else None
            ))
    
    if triagem_data:
        execute_batch(cursor, """
            INSERT INTO triagem_populacao (notificacao_id, codigo_triagem, outro_triagem)
            VALUES (%s, %s, %s)
        """, triagem_data, page_size=1000)
        conn.commit()
    
    cursor.close()
    print(f"   ✅ {len(triagem_data)} registros")

def insert_local_testagem(conn, df, notif_ids):
    """21. Inserir local_testagem (depende de: notificacao)"""
    print("\n2️⃣1️⃣  LOCAL TESTAGEM (depende de: notificação)")
    cursor = conn.cursor()
    local_data = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="   Processando"):
        if idx >= len(notif_ids):
            continue
        
        notif_id = notif_ids[idx]
        
        if pd.notna(row['codigoLocalRealizacaoTestagem']):
            local_data.append((
                notif_id,
                row['codigoLocalRealizacaoTestagem'],
                row['outroLocalRealizacaoTestagem'] if pd.notna(row['outroLocalRealizacaoTestagem']) else None
            ))
    
    if local_data:
        execute_batch(cursor, """
            INSERT INTO local_testagem (notificacao_id, codigo_local, outro_local)
            VALUES (%s, %s, %s)
        """, local_data, page_size=1000)
        conn.commit()
    
    cursor.close()
    print(f"   ✅ {len(local_data)} registros")

# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def main():
    """Função principal - Executa na ordem correta (3FN)"""
    print("="*80)
    print("CARGA DE DADOS - TERCEIRA FORMA NORMAL (3FN)")
    print("="*80)
    print("\nOrdem de insercao: Das pontas (independentes) para o centro")
    print("   Sem atributos multivalorados - cada valor e atomizado\n")
    
    # Conectar
    conn = connect_db()
    if not conn:
        return
    
    # Carregar CSV com encoding UTF-8
    print("\n📂 Carregando arquivo CSV...")
    csv_path = r"Dataset de Notificações de Síndrome Gripal - part-00000-d0823714-0f9e-4bbe-83d8-c50e1c8324c4.c000.csv.csv"
    df = pd.read_csv(csv_path, encoding='utf-8')
    print(f"   ✅ {len(df)} registros carregados\n")
    
    try:
        print("="*80)
        print("NÍVEL 1: TABELAS INDEPENDENTES (PONTAS)")
        print("="*80)
        
        # Nível 1: Tabelas independentes
        sexo_map = insert_sexo(conn)
        raca_map = insert_raca_cor(conn, df)
        prof_saude_map = insert_profissional_saude(conn)
        prof_seg_map = insert_profissional_seguranca(conn)
        sintoma_map = insert_sintomas(conn, df)
        outro_sintoma_map = insert_outro_sintoma(conn, df)
        condicao_map = insert_condicoes(conn, df)
        outra_condicao_map = insert_outras_condicoes(conn, df)
        estado_map = insert_estados(conn, df)
        municipio_map = insert_municipios(conn, df)
        busca_map = insert_busca_ativa(conn, df)
        
        print("\n" + "="*80)
        print("NÍVEL 2: TABELAS DEPENDENTES DO NÍVEL 1")
        print("="*80)
        
        # Nível 2: Depende do nível 1
        pessoa_ids = insert_pessoas(conn, df, sexo_map, raca_map, prof_saude_map, prof_seg_map)
        
        print("\n" + "="*80)
        print("NÍVEL 3: TABELAS DEPENDENTES DO NÍVEL 2")
        print("="*80)
        
        # Nível 3: Depende do nível 2
        notif_ids = insert_notificacoes(conn, df, estado_map, pessoa_ids)
        
        print("\n" + "="*80)
        print("NÍVEL 4: RELACIONAMENTOS E COMPLEMENTARES")
        print("="*80)
        
        # Nível 4: Relacionamentos e complementares
        insert_notificacao_sintomas(conn, df, notif_ids, sintoma_map, outro_sintoma_map)
        insert_notificacao_condicoes(conn, df, notif_ids, condicao_map, outra_condicao_map)
        insert_notificacao_municipio(conn, df, notif_ids, municipio_map)
        insert_notificacao_busca_ativa(conn, df, notif_ids, busca_map)
        insert_testes(conn, df, notif_ids)
        insert_vacinacao(conn, df, notif_ids)
        insert_triagem_populacao(conn, df, notif_ids)
        insert_local_testagem(conn, df, notif_ids)
        
        print("\n" + "="*80)
        print("✅ CARGA CONCLUÍDA COM SUCESSO!")
        print("="*80)
        print("\n📊 Resumo:")
        print(f"   • Pessoas: {len(pessoa_ids)}")
        print(f"   • Notificações: {len(notif_ids)}")
        print(f"   • Sintomas únicos: {len(sintoma_map)}")
        print(f"   • Condições únicas: {len(condicao_map)}")
        print(f"   • Estados: {len(estado_map)}")
        print(f"   • Municípios: {len(municipio_map)}")
        print("\n✨ Dados em 3FN - Sem atributos multivalorados!")
        
    except Exception as e:
        print(f"\n❌ Erro durante a carga: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()

if __name__ == "__main__":
    main()