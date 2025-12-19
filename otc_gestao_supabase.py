import uuid
from datetime import datetime, date, timedelta, timezone

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import requests


# =========================
# Config do app
# =========================
st.set_page_config(
    page_title="Gestão OTC - Painel Único (Supabase)",
    page_icon="📊",
    layout="wide",
)

TABLE_NAME = "trades"
SETTINGS_TABLE = "settings"
SETTINGS_ID = 1  # usamos o registro fixo id=1


# =========================
# Supabase REST helpers
# =========================
def _get_supabase_config():
    url = st.secrets.get("SUPABASE_URL", "").strip()
    key = st.secrets.get("SUPABASE_KEY", "").strip()
    if not url or not key:
        st.error(
            "Faltam SUPABASE_URL e/ou SUPABASE_KEY em .streamlit/secrets.toml.\n"
            "Crie o arquivo e reinicie o app."
        )
        st.stop()
    url = url.rstrip("/")
    return url, key


def _sb_headers(key: str, prefer_return: bool = False):
    h = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    if prefer_return:
        h["Prefer"] = "return=representation"
    return h


def sb_select_all():
    url, key = _get_supabase_config()
    endpoint = f"{url}/rest/v1/{TABLE_NAME}"
    params = {
        "select": "*",
        "order": "timestamp.asc",
    }
    r = requests.get(endpoint, headers=_sb_headers(key), params=params, timeout=20)
    if r.status_code >= 300:
        raise RuntimeError(f"Erro Supabase SELECT: {r.status_code} - {r.text}")
    return r.json()


def sb_insert(row: dict):
    url, key = _get_supabase_config()
    endpoint = f"{url}/rest/v1/{TABLE_NAME}"
    r = requests.post(endpoint, headers=_sb_headers(key, prefer_return=True), json=row, timeout=20)
    if r.status_code >= 300:
        raise RuntimeError(f"Erro Supabase INSERT: {r.status_code} - {r.text}")
    out = r.json()
    return out[0] if isinstance(out, list) and out else out


def sb_update_by_id(row_id: str, updates: dict):
    url, key = _get_supabase_config()
    endpoint = f"{url}/rest/v1/{TABLE_NAME}"
    params = {"id": f"eq.{row_id}"}
    r = requests.patch(endpoint, headers=_sb_headers(key, prefer_return=True), params=params, json=updates, timeout=20)
    if r.status_code >= 300:
        raise RuntimeError(f"Erro Supabase UPDATE: {r.status_code} - {r.text}")
    out = r.json()
    return out[0] if isinstance(out, list) and out else out


def sb_delete_by_id(row_id: str):
    url, key = _get_supabase_config()
    endpoint = f"{url}/rest/v1/{TABLE_NAME}"
    params = {"id": f"eq.{row_id}"}
    r = requests.delete(endpoint, headers=_sb_headers(key), params=params, timeout=20)
    if r.status_code >= 300:
        raise RuntimeError(f"Erro Supabase DELETE: {r.status_code} - {r.text}")
    return True


# =========================
# SETTINGS (Supabase) - salvar a coluna lateral
# =========================
def sb_get_settings() -> dict:
    """
    Lê settings (id=1) do Supabase. Você já criou no SQL.
    """
    url, key = _get_supabase_config()
    endpoint = f"{url}/rest/v1/{SETTINGS_TABLE}"
    params = {"select": "*", "id": f"eq.{SETTINGS_ID}", "limit": 1}
    r = requests.get(endpoint, headers=_sb_headers(key), params=params, timeout=20)
    if r.status_code >= 300:
        raise RuntimeError(f"Erro Supabase SETTINGS SELECT: {r.status_code} - {r.text}")
    data = r.json()
    if isinstance(data, list) and len(data) > 0:
        return data[0]
    return {}


def sb_save_settings(payload: dict):
    """
    Atualiza settings (id=1). Usamos PATCH porque o registro (id=1) existe.
    """
    url, key = _get_supabase_config()
    endpoint = f"{url}/rest/v1/{SETTINGS_TABLE}"
    params = {"id": f"eq.{SETTINGS_ID}"}

    payload = dict(payload)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()

    r = requests.patch(
        endpoint,
        headers=_sb_headers(key, prefer_return=True),
        params=params,
        json=payload,
        timeout=20,
    )

    # alguns setups retornam 204 em patch sem representation; aqui pedimos representation,
    # então o comum é 200. Mas vamos aceitar 2xx em geral.
    if r.status_code < 200 or r.status_code >= 300:
        raise RuntimeError(f"Erro Supabase SETTINGS UPDATE: {r.status_code} - {r.text}")

    return True


def ensure_session_defaults_from_supabase(force_reload: bool = False):
    """
    Carrega valores do Supabase para o st.session_state (para a lateral segurar).
    - force_reload=True: recarrega mesmo se já existir na sessão.
    """
    defaults_fallback = {
        "banca": 623.92,
        "payout_pct": 88.0,
        "entrada_pct": 2.0,
        "meta_pct": 10.0,
        "stop_pct": 10.0,
        "banca_base_mes": 623.92,
    }

    try:
        s = sb_get_settings() or {}
    except Exception as e:
        # se der erro, só usa fallback e continua o app
        s = {}
        st.sidebar.warning(f"Não consegui carregar configurações do Supabase: {e}")

    # pega do banco ou usa fallback
    banca_db = float(s.get("banca", defaults_fallback["banca"]))
    payout_db = float(s.get("payout_pct", defaults_fallback["payout_pct"]))
    entrada_db = float(s.get("entrada_pct", defaults_fallback["entrada_pct"]))
    meta_db = float(s.get("meta_pct", defaults_fallback["meta_pct"]))
    stop_db = float(s.get("stop_pct", defaults_fallback["stop_pct"]))
    banca_base_db = float(s.get("banca_base_mes", banca_db))

    desired = {
        "banca": banca_db,
        "payout_pct": payout_db,
        "entrada_pct": entrada_db,
        "meta_pct": meta_db,
        "stop_pct": stop_db,
        "banca_base_mes": banca_base_db,
    }

    for k, v in desired.items():
        if force_reload or (k not in st.session_state):
            st.session_state[k] = float(v)

    # também inicializa os “espelhos” do slider/input sem gerar warning
    if force_reload or ("entrada_pct_slider" not in st.session_state):
        st.session_state.entrada_pct_slider = float(st.session_state["entrada_pct"])
    if force_reload or ("entrada_pct_input" not in st.session_state):
        st.session_state.entrada_pct_input = float(st.session_state["entrada_pct"])


# =========================
# Utilidades
# =========================
def brl(v: float) -> str:
    s = f"{v:,.2f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {s}"


def pct(v: float) -> str:
    return f"{v:.0f}%" if abs(v - round(v)) < 1e-9 else f"{v:.2f}%"


def calc_profit_normal(entrada: float, payout_pct: float, result: str) -> float:
    result = (result or "").upper().strip()
    if result == "WIN":
        return round(entrada * (payout_pct / 100.0), 2)
    return round(-entrada, 2)


def calc_profit_g1(entrada_g1: float, entrada1_loss: float, payout_pct: float, result: str) -> float:
    """
    Regra pedida:
    - Se marcou G1, o app assume sempre que a 1ª foi LOSS (entrada1_loss)
    - O resultado selecionado refere-se ao G1 (a 2ª tentativa)
    """
    result = (result or "").upper().strip()
    primeiro = -float(entrada1_loss)  # SEMPRE LOSS
    if result == "WIN":
        segundo = float(entrada_g1) * (payout_pct / 100.0)
    else:
        segundo = -float(entrada_g1)
    return round(primeiro + segundo, 2)


def now_utc_iso():
    return datetime.now(timezone.utc).isoformat()


def today_range_local():
    now = datetime.now(timezone.utc)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = now.replace(hour=23, minute=59, second=59, microsecond=999999)
    return start, end


def week_start(d: date) -> date:
    return d - timedelta(days=d.weekday())


def month_start(d: date) -> date:
    return date(d.year, d.month, 1)


@st.cache_data(ttl=5)
def load_trades_cached():
    raw = sb_select_all()
    df = pd.DataFrame(raw if raw else [])

    needed = ["id", "timestamp", "stake", "stake_1", "payout", "result", "profit", "is_g1"]
    for c in needed:
        if c not in df.columns:
            df[c] = None

    if len(df) > 0:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["stake"] = pd.to_numeric(df["stake"], errors="coerce").fillna(0.0)
        df["stake_1"] = pd.to_numeric(df["stake_1"], errors="coerce").fillna(0.0)
        df["payout"] = pd.to_numeric(df["payout"], errors="coerce").fillna(0.0)
        df["profit"] = pd.to_numeric(df["profit"], errors="coerce").fillna(0.0)
        df["is_g1"] = df["is_g1"].fillna(False).astype(bool)
        df["result"] = (
            df["result"]
            .astype(str)
            .str.upper()
            .str.strip()
            .replace({"WIN ": "WIN", "LOSS ": "LOSS"})
        )

    df["id"] = df["id"].astype(str)
    return df


def refresh_data():
    load_trades_cached.clear()
    st.rerun()


# =========================
# 1) Carrega settings do Supabase ANTES da Sidebar
# =========================
ensure_session_defaults_from_supabase(force_reload=False)


# =========================
# Sidebar
# =========================
st.sidebar.title("⚙️ Configurações")

compact_mode = st.sidebar.toggle("Modo compacto (gráficos menores)", value=True, key="compact_mode")

banca = st.sidebar.number_input(
    "Banca atual (R$)",
    min_value=0.0,
    value=float(st.session_state["banca"]),
    step=1.0,
    format="%.2f",
    key="banca",
)

payout_pct = st.sidebar.number_input(
    "Payout (%)",
    min_value=0.0,
    max_value=99.0,
    value=float(st.session_state["payout_pct"]),
    step=1.0,
    format="%.2f",
    key="payout_pct",
)

st.sidebar.subheader("Entrada (% da banca)")

def _sync_from_slider():
    st.session_state.entrada_pct = float(st.session_state.entrada_pct_slider)
    st.session_state.entrada_pct_input = float(st.session_state.entrada_pct_slider)

def _sync_from_input():
    st.session_state.entrada_pct = float(st.session_state.entrada_pct_input)
    st.session_state.entrada_pct_slider = float(st.session_state.entrada_pct_input)

st.sidebar.slider(
    "Ajuste rápido",
    min_value=0.10,
    max_value=20.0,
    value=float(st.session_state["entrada_pct_slider"]),
    step=0.10,
    key="entrada_pct_slider",
    on_change=_sync_from_slider,
)

st.sidebar.number_input(
    "Valor exato",
    min_value=0.10,
    max_value=20.0,
    value=float(st.session_state["entrada_pct_input"]),
    step=0.10,
    format="%.2f",
    key="entrada_pct_input",
    on_change=_sync_from_input,
)

entrada_pct = float(st.session_state["entrada_pct"])

meta_pct = st.sidebar.number_input(
    "Meta do dia (%)",
    min_value=0.0,
    max_value=100.0,
    value=float(st.session_state["meta_pct"]),
    step=0.5,
    format="%.2f",
    key="meta_pct",
)

stop_pct = st.sidebar.number_input(
    "Stop do dia (%)",
    min_value=0.0,
    max_value=100.0,
    value=float(st.session_state["stop_pct"]),
    step=0.5,
    format="%.2f",
    key="stop_pct",
)

st.sidebar.divider()

banca_base_mes = st.sidebar.number_input(
    "Banca base do mês (para gráfico do mês)",
    min_value=0.0,
    value=float(st.session_state["banca_base_mes"]),
    step=1.0,
    format="%.2f",
    key="banca_base_mes",
)

st.sidebar.divider()

# ✅ BOTÕES DA SIDEBAR (o que faltava!)
if st.sidebar.button("💾 Salvar configurações (Supabase)", type="primary"):
    try:
        sb_save_settings(
            {
                "banca": float(st.session_state["banca"]),
                "payout_pct": float(st.session_state["payout_pct"]),
                "entrada_pct": float(st.session_state["entrada_pct"]),
                "meta_pct": float(st.session_state["meta_pct"]),
                "stop_pct": float(st.session_state["stop_pct"]),
                "banca_base_mes": float(st.session_state["banca_base_mes"]),
            }
        )
        st.sidebar.success("Configurações salvas! Agora não volta mais no F5.")
    except Exception as e:
        st.sidebar.error(f"Falha ao salvar configurações: {e}")

col_reload, col_sync = st.sidebar.columns(2)

with col_reload:
    if st.sidebar.button("⬇️ Recarregar do Supabase", type="secondary"):
        ensure_session_defaults_from_supabase(force_reload=True)
        st.sidebar.success("Recarregado do Supabase.")
        st.rerun()

with col_sync:
    if st.sidebar.button("🔄 Atualizar operações", type="secondary"):
        refresh_data()


# =========================
# Carregar dados
# =========================
try:
    df_all = load_trades_cached()
except Exception as e:
    st.error(f"Falha ao carregar dados do Supabase: {e}")
    st.stop()


# =========================
# Cálculos principais
# =========================
meta_val = round(banca * (meta_pct / 100.0), 2)
stop_val = round(banca * (stop_pct / 100.0), 2)
entrada_sugerida = round(banca * (entrada_pct / 100.0), 2)
lucro_por_win = round(entrada_sugerida * (payout_pct / 100.0), 2)

start_today, end_today = today_range_local()
df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], utc=True, errors="coerce")

df_today = df_all[(df_all["timestamp"] >= start_today) & (df_all["timestamp"] <= end_today)].copy()
resultado_dia = float(df_today["profit"].sum()) if len(df_today) else 0.0

status_txt = "EM OPERAÇÃO"
status_color = "🟣"
if resultado_dia >= meta_val and meta_val > 0:
    status_txt = "META DO DIA BATIDA (AVISO)"
    status_color = "🟢"
elif abs(resultado_dia) >= stop_val and stop_val > 0 and resultado_dia < 0:
    status_txt = "STOP DO DIA ATINGIDO (AVISO)"
    status_color = "🔴"

today_utc = datetime.now(timezone.utc).date()
ws = week_start(today_utc)
we = ws + timedelta(days=6)

ws_dt = datetime.combine(ws, datetime.min.time()).replace(tzinfo=timezone.utc)
we_dt = datetime.combine(we, datetime.max.time()).replace(tzinfo=timezone.utc)

df_week = df_all[(df_all["timestamp"] >= ws_dt) & (df_all["timestamp"] <= we_dt)].copy()
resultado_semana = float(df_week["profit"].sum()) if len(df_week) else 0.0
trades_semana = int(len(df_week))

ms = month_start(today_utc)
ms_dt = datetime.combine(ms, datetime.min.time()).replace(tzinfo=timezone.utc)

df_month = df_all[df_all["timestamp"] >= ms_dt].copy()
resultado_mes = float(df_month["profit"].sum()) if len(df_month) else 0.0
trades_mes = int(len(df_month))


# =========================
# Layout topo
# =========================
st.title("📊 Gestão OTC - Painel Único (Supabase)")
st.subheader(f"{status_color} {status_txt}")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Banca", brl(banca))
c2.metric("Resultado do dia", brl(resultado_dia))
c3.metric(f"Meta do dia ({pct(meta_pct)})", brl(meta_val))
c4.metric(f"Stop do dia ({pct(stop_pct)})", brl(stop_val))
c5.metric(f"Entrada sugerida ({pct(entrada_pct)})", brl(entrada_sugerida))
c6.metric("Payout", pct(payout_pct))

s1, s2, s3, s4 = st.columns(4)
s1.metric("Semana (resultado)", brl(resultado_semana))
s2.metric("Semana (trades)", str(trades_semana))
s3.metric("Mês (resultado)", brl(resultado_mes))
s4.metric("Mês (trades)", str(trades_mes))

st.caption(
    f"Se entrar com {brl(entrada_sugerida)}: WIN = {brl(lucro_por_win)} | LOSS = {brl(-entrada_sugerida)}"
)

st.divider()


# =========================
# Registrar operação
# =========================
st.subheader("📝 Registrar operação")

colA, colB, colC, colD, colE = st.columns([1.6, 1.2, 1.4, 1.4, 1.0])

with colA:
    resultado_sel = st.selectbox("Resultado (da operação final)", ["WIN", "LOSS"], index=0)

with colB:
    foi_g1 = st.checkbox("Foi G1? (reentrada técnica)", value=False)

with colC:
    entrada_g1 = st.number_input(
        "Entrada (R$)",
        min_value=0.0,
        value=float(st.session_state.get("entrada_g1", entrada_sugerida)),
        step=0.5,
        format="%.2f",
        key="entrada_g1",
        help="Valor da entrada que você executou (na tentativa final).",
    )

with colD:
    entrada_1_loss = st.number_input(
        "Entrada 1 (LOSS) (R$)",
        min_value=0.0,
        value=float(st.session_state.get("entrada_1_loss", entrada_g1)),
        step=0.5,
        format="%.2f",
        key="entrada_1_loss",
        help="No modo G1, o app assume que a primeira entrada foi LOSS.",
        disabled=not foi_g1,
    )

with colE:
    st.write("")
    st.write("")
    st.info("G1 marcado" if foi_g1 else "G1 disponível (se precisar)")

if st.button("Salvar operação", type="primary"):
    try:
        new_id = str(uuid.uuid4())
        ts = now_utc_iso()

        entrada_g1_val = float(entrada_g1)
        entrada_1_val = float(entrada_1_loss) if foi_g1 else 0.0

        if foi_g1:
            profit_val = calc_profit_g1(entrada_g1_val, entrada_1_val, float(payout_pct), resultado_sel)
        else:
            profit_val = calc_profit_normal(entrada_g1_val, float(payout_pct), resultado_sel)

        row = {
            "id": new_id,
            "timestamp": ts,
            "stake": round(entrada_g1_val, 2),
            "stake_1": round(entrada_1_val, 2),
            "payout": float(payout_pct),
            "result": resultado_sel,
            "profit": float(profit_val),
            "is_g1": bool(foi_g1),
        }

        sb_insert(row)
        st.success("Operação salva e sincronizada.")
        refresh_data()

    except Exception as e:
        st.error(f"Falha ao salvar no Supabase: {e}")

st.divider()


# =========================
# Editar / Excluir operações
# =========================
st.subheader("📋 Operações de hoje (editar / excluir)")

df_today = df_all[(df_all["timestamp"] >= start_today) & (df_all["timestamp"] <= end_today)].copy()

if len(df_today) == 0:
    st.info("Nenhuma operação hoje.")
else:
    df_show = df_today.copy()
    df_show["Hora"] = df_show["timestamp"].dt.strftime("%H:%M:%S")
    df_show["Entrada"] = df_show["stake"].apply(brl)
    df_show["Entrada 1"] = df_show["stake_1"].apply(brl)
    df_show["Payout"] = df_show["payout"].apply(lambda x: pct(float(x)))
    df_show["Profit"] = df_show["profit"].apply(brl)
    df_show["G1"] = df_show["is_g1"].apply(lambda x: "SIM" if x else "NÃO")
    df_show["Resultado"] = df_show["result"].astype(str)

    df_show = df_show[["Hora", "Entrada", "Entrada 1", "Payout", "Resultado", "Profit", "G1", "id"]].rename(
        columns={"id": "ID"}
    )

    st.dataframe(df_show, use_container_width=True, hide_index=True)

    ids = df_show["ID"].tolist()
    sel_id = st.selectbox("Selecione um registro (ID) para editar ou excluir", ids)

    reg = df_all[df_all["id"] == str(sel_id)].copy()
    if len(reg) == 0:
        st.warning("Registro não encontrado.")
    else:
        reg = reg.iloc[0]

        e1, e2, e3, e4, e5 = st.columns([1.2, 1.2, 1.2, 1.2, 1.2])

        with e1:
            edit_result = st.selectbox(
                "Resultado (editar)",
                ["WIN", "LOSS"],
                index=0 if str(reg["result"]).upper() == "WIN" else 1,
                key="edit_result",
            )

        with e2:
            edit_g1 = st.checkbox("G1 (editar)", value=bool(reg["is_g1"]), key="edit_g1")

        with e3:
            edit_entrada = st.number_input(
                "Entrada (editar) (R$)",
                min_value=0.0,
                value=float(reg["stake"]),
                step=0.5,
                format="%.2f",
                key="edit_entrada",
            )

        with e4:
            edit_payout = st.number_input(
                "Payout (editar) (%)",
                min_value=0.0,
                max_value=99.0,
                value=float(reg["payout"]),
                step=1.0,
                format="%.2f",
                key="edit_payout",
            )

        with e5:
            st.write("")
            st.write("")
            btn_save_edit = st.button("Salvar edição", type="secondary")

        edit_entrada1 = st.number_input(
            "Entrada 1 (LOSS) (editar) (R$)",
            min_value=0.0,
            value=float(reg["stake_1"]),
            step=0.5,
            format="%.2f",
            key="edit_entrada1",
            disabled=not bool(edit_g1),
            help="No modo G1, a Entrada 1 é sempre LOSS.",
        )

        bdel1, bdel2 = st.columns([1.0, 3.0])
        with bdel1:
            btn_delete = st.button("Excluir registro", type="secondary")
        with bdel2:
            st.caption("No modo G1, a Entrada 1 é sempre LOSS e o profit recalcula somando (LOSS da 1ª + resultado do G1).")

        if btn_save_edit:
            try:
                if bool(edit_g1):
                    profit_new = calc_profit_g1(float(edit_entrada), float(edit_entrada1), float(edit_payout), edit_result)
                    stake_1_val = round(float(edit_entrada1), 2)
                else:
                    profit_new = calc_profit_normal(float(edit_entrada), float(edit_payout), edit_result)
                    stake_1_val = 0.0

                updates = {
                    "stake": round(float(edit_entrada), 2),
                    "stake_1": stake_1_val,
                    "payout": float(edit_payout),
                    "result": str(edit_result).upper(),
                    "profit": float(profit_new),
                    "is_g1": bool(edit_g1),
                }

                sb_update_by_id(str(sel_id), updates)
                st.success("Edição salva e sincronizada.")
                refresh_data()

            except Exception as e:
                st.error(f"Falha ao editar no Supabase: {e}")

        if btn_delete:
            try:
                sb_delete_by_id(str(sel_id))
                st.success("Registro excluído.")
                refresh_data()
            except Exception as e:
                st.error(f"Falha ao excluir no Supabase: {e}")

st.divider()


# =========================
# Gráficos (compactos)
# =========================
def fig_size():
    if compact_mode:
        return (6.2, 2.1)
    return (7.8, 2.9)


def make_fig():
    w, h = fig_size()
    fig, ax = plt.subplots(figsize=(w, h), dpi=120)
    return fig, ax


def plot_day(df: pd.DataFrame):
    fig, ax = make_fig()
    if len(df) == 0:
        ax.text(0.5, 0.5, "Sem operações hoje", ha="center", va="center")
        ax.axis("off")
        return fig

    df = df.sort_values("timestamp").copy()
    df["acc"] = df["profit"].cumsum()

    ax.plot(range(1, len(df) + 1), df["acc"].values, marker="o")
    ax.axhline(meta_val, linestyle="--", label=f"Meta {pct(meta_pct)}")
    ax.axhline(-stop_val, linestyle="--", label=f"Stop {pct(stop_pct)}")

    ax.set_title("Dia (resultado acumulado)")
    ax.set_xlabel("Trades")
    ax.set_ylabel("R$")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_week(df: pd.DataFrame):
    fig, ax = make_fig()
    if len(df) == 0:
        ax.text(0.5, 0.5, "Sem operações na semana", ha="center", va="center")
        ax.axis("off")
        return fig

    df = df.copy()
    df["dia"] = df["timestamp"].dt.date
    daily = df.groupby("dia")["profit"].sum().reset_index()

    ax.bar(daily["dia"].astype(str), daily["profit"].values)
    ax.set_title("Semana (resultado por dia)")
    ax.set_xlabel("Dia")
    ax.set_ylabel("R$")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    fig.tight_layout()
    return fig


def plot_month_bank(df: pd.DataFrame):
    fig, ax = make_fig()
    if len(df) == 0:
        ax.text(0.5, 0.5, "Sem operações no mês", ha="center", va="center")
        ax.axis("off")
        return fig

    df = df.copy()
    df["dia"] = df["timestamp"].dt.date
    daily = df.groupby("dia")["profit"].sum().sort_index()
    bank = daily.cumsum() + float(banca_base_mes)

    ax.plot(bank.index.astype(str), bank.values, marker="o")
    ax.set_title("Mês (evolução da banca no mês)")
    ax.set_xlabel("Dia")
    ax.set_ylabel("Banca (R$)")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    fig.tight_layout()
    return fig


st.subheader("📈 Gráficos")

st.markdown("### 📊 Dia")
st.pyplot(plot_day(df_today), use_container_width=False)

st.markdown("### 📈 Semana")
st.pyplot(plot_week(df_week), use_container_width=False)

st.markdown("### 🗓️ Mês")
st.pyplot(plot_month_bank(df_month), use_container_width=False)

st.caption("Dados sincronizados no Supabase. Use 'Atualizar operações' para forçar refresh.")
