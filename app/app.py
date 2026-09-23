import os
import logging
import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc, dash_table, Input, Output, State, callback, no_update
import dash_bootstrap_components as dbc
from databricks.sdk import WorkspaceClient
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

# ════════════════════════════════════════════════════════════════
# Logging
# ════════════════════════════════════════════════════════════════
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("portfolio-manager")

# ════════════════════════════════════════════════════════════════
# Lakebase connection — OAuth token rotation via ConnectionPool
# ════════════════════════════════════════════════════════════════
w = WorkspaceClient()
ENDPOINT_NAME = os.environ.get(
    "ENDPOINT_NAME",
    "projects/fe-bar/branches/production/endpoints/primary",
)


class OAuthConnection(psycopg.Connection):
    """Injects a fresh OAuth token each time the pool opens a connection."""

    @classmethod
    def connect(cls, conninfo="", **kwargs):
        cred = w.postgres.generate_database_credential(endpoint=ENDPOINT_NAME)
        kwargs["password"] = cred.token
        return super().connect(conninfo, **kwargs)


_user = os.environ.get("PGUSER", os.environ.get("DATABRICKS_CLIENT_ID", ""))
_host = os.environ.get("PGHOST", "ep-super-tree-d283ymp5.database.us-east-1.cloud.databricks.com")
_port = os.environ.get("PGPORT", "5432")
_db = os.environ.get("PGDATABASE", "databricks_postgres")
_ssl = os.environ.get("PGSSLMODE", "require")

pool = ConnectionPool(
    conninfo=f"dbname={_db} user={_user} host={_host} port={_port} sslmode={_ssl}",
    connection_class=OAuthConnection,
    min_size=1,
    max_size=5,
    open=True,
)


def query(sql, params=None):
    """Execute a read query and return a DataFrame."""
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ════════════════════════════════════════════════════════════════
# Discover synced-table schema
# ════════════════════════════════════════════════════════════════
def _discover_schema():
    try:
        df = query(
            "SELECT table_schema FROM information_schema.tables "
            "WHERE table_name = 'client_worklist_synced' LIMIT 1"
        )
        if not df.empty:
            return df.iloc[0]["table_schema"]
    except Exception as exc:
        log.warning("Schema discovery failed: %s", exc)
    return "public"


SCHEMA = _discover_schema()
if SCHEMA == "public":
    SCHEMA = "gold"  # synced tables live in the gold schema
log.info("Postgres schema: %s", SCHEMA)
CW = f'"{SCHEMA}"."client_worklist_synced"'
AH = f'"{SCHEMA}"."account_health_synced"'


def load_clients():
    return query(f"SELECT * FROM {CW} ORDER BY priority_score DESC")


def load_accounts():
    return query(f"SELECT * FROM {AH}")


# ════════════════════════════════════════════════════════════════
# Dash app
# ════════════════════════════════════════════════════════════════
app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.FLATLY,
        "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css",
    ],
    suppress_callback_exceptions=True,
    title="Portfolio Manager",
)

COLORS = {
    "primary": "#2c3e50",
    "accent": "#3498db",
    "danger": "#e74c3c",
    "success": "#27ae60",
    "warning": "#f39c12",
}


# ──── Helpers ────────────────────────────────────────────────────
def kpi_card(title, value, icon, color):
    return dbc.Card(
        dbc.CardBody([
            html.Div(
                html.I(className=f"bi {icon}", style={"fontSize": "1.8rem", "color": color}),
                style={"float": "right"},
            ),
            html.P(title, className="text-muted mb-1",
                   style={"fontSize": "0.78rem", "textTransform": "uppercase"}),
            html.H4(value, className="mb-0 fw-bold"),
        ]),
        className="shadow-sm h-100",
        style={"borderLeft": f"4px solid {color}"},
    )


def safe_cols(df, wanted):
    return [c for c in wanted if c in df.columns]


# ──── Navbar ─────────────────────────────────────────────────────
navbar = dbc.Navbar(
    dbc.Container([
        dbc.NavbarBrand([
            html.I(className="bi bi-briefcase-fill me-2"),
            "Portfolio Manager",
        ], className="fw-bold fs-4"),
        dbc.Button(
            [html.I(className="bi bi-arrow-clockwise me-1"), "Refresh"],
            id="btn-refresh", color="light", size="sm",
        ),
    ], fluid=True),
    color="primary", dark=True, className="mb-3",
)


# ──── Filters ────────────────────────────────────────────────────
filters = dbc.Row([
    dbc.Col(dbc.Select(id="f-advisor", placeholder="All Advisors"), md=3),
    dbc.Col(dbc.Select(id="f-risk", placeholder="All Risk Profiles"), md=2),
    dbc.Col(dbc.Select(id="f-state", placeholder="All States"), md=2),
    dbc.Col(
        dbc.Checklist(
            id="f-attn",
            options=[{"label": " Needs Attention Only", "value": "y"}],
            value=[], inline=True, className="mt-2",
        ), md=3,
    ),
], className="mb-3 g-2")


# ──── Layout ─────────────────────────────────────────────────────
app.layout = html.Div([
    navbar,
    dbc.Container([
        dbc.Row(id="kpi-row", className="mb-3 g-3"),
        filters,
        dbc.Tabs([
            dbc.Tab(label="Client Worklist", tab_id="t-wl"),
            dbc.Tab(label="Account Health", tab_id="t-ah"),
            dbc.Tab(label="At-Risk Accounts", tab_id="t-risk"),
            dbc.Tab(label="Portfolio Analytics", tab_id="t-analytics"),
        ], id="tabs", active_tab="t-wl"),
        html.Div(id="tab-body", className="mt-3"),
    ], fluid=True),
    dcc.Store(id="s-clients"),
    dcc.Store(id="s-accounts"),
    dcc.Store(id="s-sel-client"),
])


# ════════════════════════════════════════════════════════════════
# Callbacks
# ════════════════════════════════════════════════════════════════

@callback(
    Output("s-clients", "data"),
    Output("s-accounts", "data"),
    Output("f-advisor", "options"),
    Output("f-risk", "options"),
    Output("f-state", "options"),
    Input("btn-refresh", "n_clicks"),
)
def cb_load(_):
    clients = load_clients()
    accounts = load_accounts()

    def opts(series, label):
        vals = sorted(series.dropna().unique())
        return [{"label": label, "value": ""}] + [{"label": v, "value": v} for v in vals]

    return (
        clients.to_json(date_format="iso", orient="split"),
        accounts.to_json(date_format="iso", orient="split"),
        opts(clients["advisor_name"], "All Advisors") if "advisor_name" in clients.columns else [],
        opts(clients["risk_profile"], "All Risk Profiles") if "risk_profile" in clients.columns else [],
        opts(clients["state"], "All States") if "state" in clients.columns else [],
    )


def _filter(cj, adv, risk, state, attn):
    if not cj:
        return pd.DataFrame()
    df = pd.read_json(cj, orient="split")
    if adv:
        df = df[df["advisor_name"] == adv]
    if risk:
        df = df[df["risk_profile"] == risk]
    if state:
        df = df[df["state"] == state]
    if "y" in (attn or []):
        df = df[df["needs_attention"] == True]
    return df


# ── KPIs ─────────────────────────────────────────────────────────
@callback(
    Output("kpi-row", "children"),
    Input("s-clients", "data"),
    Input("f-advisor", "value"),
    Input("f-risk", "value"),
    Input("f-state", "value"),
    Input("f-attn", "value"),
)
def cb_kpis(cj, adv, risk, st, attn):
    df = _filter(cj, adv, risk, st, attn)
    if df.empty:
        return [dbc.Col(html.P("No data.", className="text-muted"))]
    aum = df["portfolio_value"].sum()
    aum_s = f"${aum / 1e6:,.1f}M" if aum >= 1e6 else f"${aum:,.0f}"
    return [
        dbc.Col(kpi_card("Total AUM", aum_s, "bi-bank", COLORS["accent"]), md=3),
        dbc.Col(kpi_card("Clients", f"{len(df):,}", "bi-people-fill", COLORS["primary"]), md=3),
        dbc.Col(kpi_card("Needs Attention", f"{int(df['needs_attention'].sum()):,}",
                         "bi-exclamation-triangle-fill", COLORS["danger"]), md=3),
        dbc.Col(kpi_card("Avg Priority", f"{df['priority_score'].mean():.1f}",
                         "bi-speedometer2", COLORS["warning"]), md=3),
    ]


# ── Tab router ───────────────────────────────────────────────────
@callback(
    Output("tab-body", "children"),
    Input("tabs", "active_tab"),
    Input("s-clients", "data"),
    Input("s-accounts", "data"),
    Input("f-advisor", "value"),
    Input("f-risk", "value"),
    Input("f-state", "value"),
    Input("f-attn", "value"),
    Input("s-sel-client", "data"),
)
def cb_tab(tab, cj, aj, adv, risk, st, attn, sel):
    clients = _filter(cj, adv, risk, st, attn)
    accounts = pd.read_json(aj, orient="split") if aj else pd.DataFrame()
    all_clients = pd.read_json(cj, orient="split") if cj else pd.DataFrame()

    if tab == "t-wl":
        return _render_worklist(clients)
    if tab == "t-ah":
        return _render_accounts(sel, accounts, all_clients)
    if tab == "t-risk":
        return _render_risk(clients)
    if tab == "t-analytics":
        return _render_analytics(clients, accounts)
    return html.P("Select a tab.")


# ── Worklist ─────────────────────────────────────────────────────
def _render_worklist(df):
    if df.empty:
        return html.P("No clients match filters.", className="text-muted")
    cols = safe_cols(df, [
        "client_id", "first_name", "last_name", "risk_profile", "state",
        "portfolio_value", "priority_score", "needs_attention", "reason",
        "flag_drift", "flag_cash_drag", "flag_review_overdue",
        "flag_tax_loss_harvest", "flag_rmd", "flag_concentrated_drop",
        "advisor_name",
    ])
    return dash_table.DataTable(
        id="wl-table",
        columns=[
            {"name": c.replace("_", " ").title(), "id": c,
             "type": "numeric" if df[c].dtype in ("float64", "int64", "int32") else "text"}
            for c in cols
        ],
        data=df[cols].to_dict("records"),
        sort_action="native",
        filter_action="native",
        page_size=20,
        row_selectable="single",
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "8px", "fontSize": "0.85rem"},
        style_header={"backgroundColor": COLORS["primary"], "color": "white", "fontWeight": "bold"},
        style_data_conditional=[
            {"if": {"filter_query": "{needs_attention} = true"}, "backgroundColor": "#fde8e8"},
            {"if": {"filter_query": "{priority_score} >= 50"}, "fontWeight": "bold"},
        ],
    )


@callback(
    Output("s-sel-client", "data"),
    Input("wl-table", "selected_rows"),
    State("wl-table", "data"),
    prevent_initial_call=True,
)
def cb_select(rows, data):
    if not rows or not data:
        return no_update
    return data[rows[0]]["client_id"]


# ── Account Health ───────────────────────────────────────────────
def _render_accounts(client_id, accounts, all_clients):
    if not client_id:
        return dbc.Alert(
            [html.I(className="bi bi-info-circle me-2"),
             "Select a client from the Worklist tab to view account details."],
            color="info",
        )
    crow = all_clients[all_clients["client_id"] == client_id]
    if crow.empty:
        return html.P("Client not found.")
    c = crow.iloc[0]

    flags = []
    for fc in ["flag_drift", "flag_cash_drag", "flag_review_overdue",
               "flag_tax_loss_harvest", "flag_rmd", "flag_concentrated_drop"]:
        if fc in c and c[fc]:
            flags.append(dbc.Badge(
                fc.replace("flag_", "").replace("_", " ").title(),
                color="danger", className="me-1",
            ))

    card = dbc.Card([
        dbc.CardHeader(html.H5(
            f"{c.get('first_name', '')} {c.get('last_name', '')}  ({client_id})",
            className="mb-0",
        )),
        dbc.CardBody(dbc.Row([
            dbc.Col([
                html.P([html.Strong("Risk: "), str(c.get("risk_profile", ""))]),
                html.P([html.Strong("State: "), str(c.get("state", ""))]),
                html.P([html.Strong("Advisor: "), str(c.get("advisor_name", ""))]),
            ], md=4),
            dbc.Col([
                html.P([html.Strong("Portfolio: "), f"${c.get('portfolio_value', 0):,.2f}"]),
                html.P([html.Strong("Priority: "), str(c.get("priority_score", 0))]),
                html.P([html.Strong("Accounts: "), str(c.get("num_accounts", 0))]),
            ], md=4),
            dbc.Col([
                html.P([html.Strong("Flags: ")] + (flags or [html.Span("None", className="text-muted")])),
                html.P([html.Strong("Reason: "), str(c.get("reason", ""))]),
            ], md=4),
        ])),
    ], className="mb-4 shadow-sm")

    accts = accounts[accounts["client_id"] == client_id] if not accounts.empty else pd.DataFrame()
    if accts.empty:
        return html.Div([card, html.P("No accounts found.")])

    acol = safe_cols(accts, [
        "account_id", "account_type", "model_portfolio", "account_value",
        "max_drift", "cash_weight", "taxable_unrealized_loss", "has_concentrated_drop",
    ])
    tbl = dash_table.DataTable(
        columns=[{"name": c.replace("_", " ").title(), "id": c} for c in acol],
        data=accts[acol].to_dict("records"),
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "8px", "fontSize": "0.85rem"},
        style_header={"backgroundColor": COLORS["primary"], "color": "white", "fontWeight": "bold"},
    )

    charts = []
    if len(accts) > 1 and "account_value" in accts.columns:
        fig = px.bar(accts, x="account_id", y="account_value",
                     color="account_type" if "account_type" in accts.columns else None,
                     title="Account Values")
        fig.update_layout(template="plotly_white", height=320)
        charts.append(dbc.Col(dcc.Graph(figure=fig), md=6))
    if len(accts) > 1 and "max_drift" in accts.columns:
        fig2 = px.bar(accts, x="account_id", y="max_drift",
                      color="model_portfolio" if "model_portfolio" in accts.columns else None,
                      title="Max Drift by Account")
        fig2.update_layout(template="plotly_white", height=320)
        charts.append(dbc.Col(dcc.Graph(figure=fig2), md=6))

    return html.Div([
        card,
        html.H5("Accounts", className="mb-3"),
        tbl,
        dbc.Row(charts, className="mt-3") if charts else html.Div(),
    ])


# ── At-Risk ──────────────────────────────────────────────────────
def _render_risk(df):
    df = df[df["needs_attention"] == True] if not df.empty else df
    if df.empty:
        return html.P("No at-risk clients.", className="text-muted")

    flag_cols = safe_cols(df, [
        "flag_drift", "flag_cash_drag", "flag_review_overdue",
        "flag_tax_loss_harvest", "flag_rmd", "flag_concentrated_drop",
    ])
    flag_counts = {
        c.replace("flag_", "").replace("_", " ").title(): int(df[c].sum())
        for c in flag_cols
    }
    fdf = (pd.DataFrame(list(flag_counts.items()), columns=["Flag", "Count"])
           .sort_values("Count", ascending=True))

    fig1 = px.bar(fdf, x="Count", y="Flag", orientation="h",
                  title="Flag Distribution", color="Count", color_continuous_scale="Reds")
    fig1.update_layout(template="plotly_white", height=340, showlegend=False)

    fig2 = px.histogram(df, x="priority_score", nbins=15,
                        title="Priority Score Distribution",
                        color_discrete_sequence=[COLORS["danger"]])
    fig2.update_layout(template="plotly_white", height=340)

    cols = safe_cols(df, [
        "client_id", "first_name", "last_name", "priority_score",
        "reason", "advisor_name", "portfolio_value",
    ])
    tbl = dash_table.DataTable(
        columns=[{"name": c.replace("_", " ").title(), "id": c} for c in cols],
        data=df[cols].sort_values("priority_score", ascending=False).to_dict("records"),
        sort_action="native", page_size=15,
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "8px", "fontSize": "0.85rem"},
        style_header={"backgroundColor": COLORS["danger"], "color": "white", "fontWeight": "bold"},
    )

    return html.Div([
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig1), md=6),
            dbc.Col(dcc.Graph(figure=fig2), md=6),
        ], className="mb-3"),
        html.H5(f"At-Risk Clients ({len(df)})", className="mb-3"),
        tbl,
    ])


# ── Analytics ────────────────────────────────────────────────────
def _render_analytics(clients, accounts):
    if clients.empty:
        return html.P("No data.", className="text-muted")

    figs = []

    # Risk-profile pie
    rc = clients["risk_profile"].value_counts().reset_index()
    rc.columns = ["Risk Profile", "Count"]
    f1 = px.pie(rc, names="Risk Profile", values="Count",
                title="Clients by Risk Profile", hole=0.4)
    f1.update_layout(template="plotly_white", height=360)
    figs.append(dbc.Col(dcc.Graph(figure=f1), md=6))

    # Portfolio-value histogram
    f2 = px.histogram(clients, x="portfolio_value", nbins=20,
                      title="Portfolio Value Distribution",
                      color_discrete_sequence=[COLORS["accent"]])
    f2.update_layout(template="plotly_white", height=360,
                     xaxis_title="Portfolio Value ($)", yaxis_title="Count")
    figs.append(dbc.Col(dcc.Graph(figure=f2), md=6))

    if not accounts.empty:
        accts = accounts[accounts["client_id"].isin(clients["client_id"])]

        if "max_drift" in accts.columns:
            f3 = px.histogram(accts, x="max_drift", nbins=20,
                              title="Drift Distribution",
                              color_discrete_sequence=[COLORS["warning"]])
            f3.update_layout(template="plotly_white", height=360)
            figs.append(dbc.Col(dcc.Graph(figure=f3), md=6))

        if "cash_weight" in accts.columns:
            f4 = px.histogram(accts, x="cash_weight", nbins=20,
                              title="Cash Weight Distribution",
                              color_discrete_sequence=[COLORS["success"]])
            f4.update_layout(template="plotly_white", height=360)
            figs.append(dbc.Col(dcc.Graph(figure=f4), md=6))

        if "model_portfolio" in accts.columns and "account_value" in accts.columns:
            mp = accts.groupby("model_portfolio")["account_value"].sum().reset_index()
            mp.columns = ["Model", "AUM"]
            f5 = px.bar(mp.sort_values("AUM"), x="AUM", y="Model",
                        orientation="h", title="AUM by Model Portfolio", color="Model")
            f5.update_layout(template="plotly_white", height=360, showlegend=False)
            figs.append(dbc.Col(dcc.Graph(figure=f5), md=6))

        if "account_type" in accts.columns:
            tc = accts["account_type"].value_counts().reset_index()
            tc.columns = ["Type", "Count"]
            f6 = px.pie(tc, names="Type", values="Count",
                        title="Accounts by Type", hole=0.4)
            f6.update_layout(template="plotly_white", height=360)
            figs.append(dbc.Col(dcc.Graph(figure=f6), md=6))

    return dbc.Row(figs, className="g-3")


# ════════════════════════════════════════════════════════════════
# Run
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
