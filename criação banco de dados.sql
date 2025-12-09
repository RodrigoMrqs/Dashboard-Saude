CREATE TABLE sexo (
    id SERIAL PRIMARY KEY,
    descricao VARCHAR(20) UNIQUE NOT NULL
);

CREATE TABLE raca_cor (
    id SERIAL PRIMARY KEY,
    descricao VARCHAR(30) UNIQUE NOT NULL
);

CREATE TABLE condicoes (
	id SERIAL PRIMARY KEY,
	descricao VARCHAR UNIQUE NOT NULL
);

CREATE TABLE outras_condicoes (
	id SERIAL PRIMARY KEY,
	descricao VARCHAR UNIQUE NOT NULL
);

CREATE TABLE estado (
    id SERIAL PRIMARY KEY,
    nome VARCHAR(50),
	estado_ibge VARCHAR(10),
    estado_notificacao VARCHAR(10),
	estado_notificacao_ibge VARCHAR(10),
    sigla VARCHAR(2) UNIQUE
);

CREATE TABLE busca_ativa (
	id SERIAL PRIMARY KEY,
	codigo_Estrategia_Covid smallint,
	codigo_BuscaAtiva_Assintomatico smallint,
	outro_Busca_Ativa_Assintomatico smallint,
	codigo_Contem_Comunidade_Tradicional smallint
);

CREATE TABLE municipio (
    id SERIAL PRIMARY KEY,
	municipio_Notificacao varchar(50), 
	municipio_NotificacaoIBGE int UNIQUE
);

CREATE TABLE profissional_saude (
    id SERIAL PRIMARY KEY,
    descricao VARCHAR(10) UNIQUE NOT NULL
);

CREATE TABLE profissional_seguranca (
    id SERIAL PRIMARY KEY,
    descricao VARCHAR(10) UNIQUE NOT NULL
);

CREATE TABLE sintoma (
    id SERIAL PRIMARY KEY,
    descricao VARCHAR(150) UNIQUE NOT NULL
);

CREATE TABLE outro_sintoma (
	id SERIAL PRIMARY KEY,
	descricao VARCHAR UNIQUE NOT NULL
);

CREATE TABLE pessoa (
	id SERIAL PRIMARY KEY,
    sexo_id INT REFERENCES sexo(id),
    raca_cor_id INT REFERENCES raca_cor(id),
	
    profissional_saude_id INT REFERENCES profissional_saude(id),
    profissional_seguranca_id INT REFERENCES profissional_seguranca(id)
);

CREATE TABLE notificacao (
    id SERIAL PRIMARY KEY,
    estado_id INT REFERENCES estado(id),
	pessoa_id INT REFERENCES pessoa(id),

    cbo VARCHAR(20),
	origem varchar,
	evolucaoCaso varchar,
	classificacaoFinal varchar,

    total_testes_realizados INT,

    data_notificacao DATE,
    data_inicio_sintomas DATE,
    data_encerramento DATE,
    idade INT,
	
    excluido BOOLEAN,
    validado BOOLEAN
);

CREATE TABLE notificacao_sintoma (
    notificacao_id INT REFERENCES notificacao(id) ON DELETE CASCADE,
	outro_sintoma_id INT REFERENCES outro_sintoma(id),
    sintoma_id INT REFERENCES sintoma(id),
    PRIMARY KEY (notificacao_id, sintoma_id, outro_sintoma_id)
);

CREATE TABLE notificacao_condicao (
    notificacao_id INT REFERENCES notificacao(id) ON DELETE CASCADE,
	outras_condicoes_id INT REFERENCES outras_condicoes(id),
    condicoes_id INT REFERENCES condicoes(id),
    PRIMARY KEY (notificacao_id, condicoes_id, outras_condicoes_id)
);

CREATE TABLE teste (
    id SERIAL PRIMARY KEY,
    notificacao_id INT REFERENCES notificacao(id) ON DELETE CASCADE,
    ordem INT,  -- 1,2,3,4

    codigo_estado VARCHAR(20),
    codigo_tipo VARCHAR(20),
    codigo_fabricante VARCHAR(50),
    codigo_resultado VARCHAR(20),

    data_coleta DATE
);

CREATE TABLE vacinacao (
    id SERIAL PRIMARY KEY,
    notificacao_id INT REFERENCES notificacao(id) ON DELETE CASCADE,

    codigo_recebeu_vacina VARCHAR(10),

    codigo_lab_primeira_dose VARCHAR(50),
    codigo_lab_segunda_dose VARCHAR(50),

    lote_primeira_dose VARCHAR(50),
    lote_segunda_dose VARCHAR(50),

    codigo_doses_vacina VARCHAR(10),

    data_primeira_dose DATE,
    data_segunda_dose DATE
);

CREATE TABLE triagem_populacao (
    id SERIAL PRIMARY KEY,
    notificacao_id INT REFERENCES notificacao(id) ON DELETE CASCADE,

    codigo_triagem VARCHAR(20),
    outro_triagem VARCHAR(100)
);

CREATE TABLE notificacao_municipio (
    notificacao_id INT REFERENCES notificacao(id) ON DELETE CASCADE,
	municipio_id INT REFERENCES municipio(id),
    PRIMARY KEY (notificacao_id, municipio_id)
);

CREATE TABLE notificacao_busca_ativa (
    notificacao_id INT REFERENCES notificacao(id) ON DELETE CASCADE,
	busca_ativa_id INT REFERENCES busca_ativa(id),
    PRIMARY KEY (notificacao_id, busca_ativa_id)
);

CREATE TABLE local_testagem (
    id SERIAL PRIMARY KEY,
    notificacao_id INT REFERENCES notificacao(id) ON DELETE CASCADE,

    codigo_local VARCHAR(20),
    outro_local VARCHAR(100)
);

CREATE TABLE IF NOT EXISTS log_alteracoes (
    id SERIAL PRIMARY KEY,
    tabela VARCHAR(50) NOT NULL,
    operacao VARCHAR(10) NOT NULL,  -- INSERT, UPDATE, DELETE
    registro_id INT NOT NULL,
    usuario TEXT DEFAULT CURRENT_USER,
    data_hora TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    valores_anteriores JSONB,
    valores_novos JSONB
);

CREATE OR REPLACE FUNCTION fx_log_alteracoes() 
RETURNS TRIGGER AS $$
BEGIN
    IF (TG_OP = 'INSERT') THEN
        INSERT INTO log_alteracoes(tabela, operacao, registro_id, valores_novos)
        VALUES (TG_TABLE_NAME, TG_OP, NEW.id, row_to_json(NEW));
        RETURN NEW;

    ELSIF (TG_OP = 'UPDATE') THEN
        INSERT INTO log_alteracoes(tabela, operacao, registro_id, valores_anteriores, valores_novos)
        VALUES (TG_TABLE_NAME, TG_OP, NEW.id, row_to_json(OLD), row_to_json(NEW));
        RETURN NEW;

    ELSIF (TG_OP = 'DELETE') THEN
        INSERT INTO log_alteracoes(tabela, operacao, registro_id, valores_anteriores)
        VALUES (TG_TABLE_NAME, TG_OP, OLD.id, row_to_json(OLD));
        RETURN OLD;
    END IF;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER tg_notificacao_log
AFTER INSERT OR UPDATE OR DELETE ON notificacao
FOR EACH ROW EXECUTE PROCEDURE fx_log_alteracoes();

CREATE TRIGGER tg_teste_log
AFTER INSERT OR UPDATE OR DELETE ON teste
FOR EACH ROW EXECUTE PROCEDURE fx_log_alteracoes();

CREATE TABLE IF NOT EXISTS indicadores_regionais (
    id SERIAL PRIMARY KEY,
    municipio_id INT REFERENCES municipio(id),
    data_inicio DATE NOT NULL,
    data_fim DATE NOT NULL,
    total_testes INT,
    testes_positivos INT,
    taxa_positividade NUMERIC(5,2)
);

CREATE OR REPLACE FUNCTION fx_calcular_taxa_positividade(
    p_data_inicio DATE,
    p_data_fim DATE
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO indicadores_regionais (
        municipio_id, data_inicio, data_fim, 
        total_testes, testes_positivos, taxa_positividade
    )
    SELECT 
        nm.municipio_id,
        p_data_inicio,
        p_data_fim,
        COUNT(t.id) AS total_testes,
        COUNT(CASE WHEN t.codigo_resultado = 'POSITIVO' THEN 1 END) AS testes_positivos,
        (
            COUNT(CASE WHEN t.codigo_resultado = 'POSITIVO' THEN 1 END)::NUMERIC
            / NULLIF(COUNT(t.id), 0)
        ) * 100 AS taxa_positividade
    FROM notificacao n
    JOIN notificacao_municipio nm ON nm.notificacao_id = n.id
    JOIN teste t ON t.notificacao_id = n.id
    WHERE t.data_coleta BETWEEN p_data_inicio AND p_data_fim
    GROUP BY nm.municipio_id;
END;
$$ LANGUAGE plpgsql;


CREATE OR REPLACE VIEW vw_casos_por_municipio AS
SELECT
    m.id AS municipio_id,
    m.municipio_notificacao AS municipio,
    n.data_notificacao,
    COUNT(*) AS total_notificacoes,
    COUNT(CASE WHEN n.classificacaoFinal = 'Confirmado' THEN 1 END) AS casos_confirmados,
    COUNT(CASE WHEN n.classificacaoFinal = 'Descartado' THEN 1 END) AS casos_descartados
FROM notificacao n
JOIN notificacao_municipio nm ON nm.notificacao_id = n.id
JOIN municipio m ON m.id = nm.municipio_id
GROUP BY m.id, m.municipio_notificacao, n.data_notificacao;

CREATE OR REPLACE VIEW vw_vacinacao_por_resultado AS
SELECT
    n.id AS notificacao_id,
    t.codigo_resultado,
    v.codigo_recebeu_vacina,
    v.lote_primeira_dose,
    v.lote_segunda_dose
FROM notificacao n
LEFT JOIN teste t ON t.notificacao_id = n.id
LEFT JOIN vacinacao v ON v.notificacao_id = n.id;

CREATE OR REPLACE VIEW vw_sintomas_frequentes AS
SELECT 
    s.descricao AS sintoma,
    COUNT(*) AS ocorrencias
FROM notificacao n
JOIN notificacao_sintoma ns ON ns.notificacao_id = n.id
JOIN sintoma s ON s.id = ns.sintoma_id
WHERE n.classificacaoFinal = 'Confirmado'
GROUP BY s.descricao
ORDER BY COUNT(*) DESC;

