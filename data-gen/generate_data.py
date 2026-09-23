#!/usr/bin/env python3
"""
Synthetic data generator for the Wealth-Advisor Worklist demo.

Produces raw parquet files modeled on nightly custodian drops (Schwab/Fidelity/
Pershing-style flat files), ready to upload to S3 and ingest with Autoloader.

Output layout (under ./data):
  data/reference/advisors.parquet
  data/reference/clients.parquet
  data/reference/accounts.parquet
  data/reference/securities.parquet
  data/reference/model_allocations.parquet
  data/prices/prices_YYYY-MM-DD.parquet             (one file per business day)
  data/positions/positions_YYYY-MM-DD.parquet       (one file per business day)
  data/transactions/transactions_YYYY-MM-DD.parquet (one file per business day)

The generator deliberately SEEDS the "needs a call today" conditions so the
downstream worklist is populated but selective:
  - allocation drift vs. model portfolio
  - concentrated position in a stock that just cratered
  - taxable lots sitting at an unrealized loss (tax-loss harvesting)
  - cash drag (idle cash well above target)
  - RMD-age clients holding a Traditional IRA
  - client review overdue (> 12 months)
"""

from pathlib import Path
import numpy as np
import pandas as pd

SEED = 7
rng = np.random.default_rng(SEED)

TODAY = pd.Timestamp("2026-08-28")
N_BIZ_DAYS = 20
BIZ_DAYS = pd.bdate_range(end=TODAY, periods=N_BIZ_DAYS)

N_ADVISORS = 10
CLIENTS_PER_ADVISOR = 20  # -> 200 clients

OUT = Path(__file__).parent / "data"
CRASH_TICKERS = ["NVDA", "BA"]  # engineered sell-offs late in the window

# ---------------------------------------------------------------------------
# Reference: securities  (symbol, name, asset_class, sector, base_price, ann_vol)
# ---------------------------------------------------------------------------
SECURITIES = [
    # --- Equity: individual stocks ---
    ("AAPL", "Apple Inc.", "Equity", "Information Technology", 230, 0.26),
    ("MSFT", "Microsoft Corp.", "Equity", "Information Technology", 430, 0.24),
    ("NVDA", "NVIDIA Corp.", "Equity", "Information Technology", 120, 0.45),
    ("GOOGL", "Alphabet Inc.", "Equity", "Communication Services", 175, 0.27),
    ("META", "Meta Platforms", "Equity", "Communication Services", 520, 0.33),
    ("AMZN", "Amazon.com", "Equity", "Consumer Discretionary", 185, 0.30),
    ("TSLA", "Tesla Inc.", "Equity", "Consumer Discretionary", 250, 0.50),
    ("CRM", "Salesforce", "Equity", "Information Technology", 270, 0.30),
    ("ORCL", "Oracle Corp.", "Equity", "Information Technology", 170, 0.28),
    ("ADBE", "Adobe Inc.", "Equity", "Information Technology", 500, 0.30),
    ("JPM", "JPMorgan Chase", "Equity", "Financials", 210, 0.24),
    ("BAC", "Bank of America", "Equity", "Financials", 40, 0.28),
    ("WFC", "Wells Fargo", "Equity", "Financials", 60, 0.28),
    ("GS", "Goldman Sachs", "Equity", "Financials", 480, 0.27),
    ("MS", "Morgan Stanley", "Equity", "Financials", 100, 0.27),
    ("V", "Visa Inc.", "Equity", "Financials", 280, 0.22),
    ("MA", "Mastercard", "Equity", "Financials", 470, 0.23),
    ("JNJ", "Johnson & Johnson", "Equity", "Health Care", 155, 0.17),
    ("UNH", "UnitedHealth Group", "Equity", "Health Care", 500, 0.28),
    ("PFE", "Pfizer Inc.", "Equity", "Health Care", 28, 0.24),
    ("MRK", "Merck & Co.", "Equity", "Health Care", 110, 0.21),
    ("ABBV", "AbbVie Inc.", "Equity", "Health Care", 175, 0.22),
    ("LLY", "Eli Lilly", "Equity", "Health Care", 800, 0.30),
    ("HD", "Home Depot", "Equity", "Consumer Discretionary", 360, 0.24),
    ("MCD", "McDonald's", "Equity", "Consumer Discretionary", 290, 0.18),
    ("NKE", "Nike Inc.", "Equity", "Consumer Discretionary", 80, 0.28),
    ("SBUX", "Starbucks", "Equity", "Consumer Discretionary", 95, 0.26),
    ("PG", "Procter & Gamble", "Equity", "Consumer Staples", 165, 0.15),
    ("KO", "Coca-Cola", "Equity", "Consumer Staples", 68, 0.15),
    ("PEP", "PepsiCo", "Equity", "Consumer Staples", 170, 0.16),
    ("WMT", "Walmart", "Equity", "Consumer Staples", 75, 0.19),
    ("COST", "Costco", "Equity", "Consumer Staples", 900, 0.22),
    ("XOM", "Exxon Mobil", "Equity", "Energy", 115, 0.25),
    ("CVX", "Chevron", "Equity", "Energy", 155, 0.24),
    ("COP", "ConocoPhillips", "Equity", "Energy", 105, 0.28),
    ("BA", "Boeing Co.", "Equity", "Industrials", 180, 0.35),
    ("CAT", "Caterpillar", "Equity", "Industrials", 350, 0.26),
    ("GE", "GE Aerospace", "Equity", "Industrials", 180, 0.30),
    ("HON", "Honeywell", "Equity", "Industrials", 210, 0.22),
    ("UPS", "United Parcel Service", "Equity", "Industrials", 130, 0.26),
    ("DIS", "Walt Disney", "Equity", "Communication Services", 95, 0.28),
    ("NFLX", "Netflix", "Equity", "Communication Services", 700, 0.34),
    ("VZ", "Verizon", "Equity", "Communication Services", 42, 0.18),
    ("NEE", "NextEra Energy", "Equity", "Utilities", 80, 0.22),
    ("DUK", "Duke Energy", "Equity", "Utilities", 110, 0.18),
    ("AMT", "American Tower", "Equity", "Real Estate", 200, 0.24),
    ("PLD", "Prologis", "Equity", "Real Estate", 115, 0.26),
    # --- Equity: ETFs ---
    ("SPY", "SPDR S&P 500 ETF", "Equity", "US Equity ETF", 550, 0.16),
    ("VOO", "Vanguard S&P 500 ETF", "Equity", "US Equity ETF", 500, 0.16),
    ("VTI", "Vanguard Total Market ETF", "Equity", "US Equity ETF", 275, 0.16),
    ("QQQ", "Invesco QQQ Trust", "Equity", "US Equity ETF", 480, 0.22),
    ("IWM", "iShares Russell 2000 ETF", "Equity", "US Equity ETF", 220, 0.22),
    ("VEA", "Vanguard Developed Mkts ETF", "Equity", "Intl Equity ETF", 52, 0.17),
    ("VWO", "Vanguard Emerging Mkts ETF", "Equity", "Intl Equity ETF", 45, 0.20),
    # --- Fixed Income ETFs ---
    ("AGG", "iShares Core US Aggregate Bond", "Fixed Income", "Bond ETF", 100, 0.06),
    ("BND", "Vanguard Total Bond Market", "Fixed Income", "Bond ETF", 74, 0.06),
    ("TLT", "iShares 20+ Yr Treasury", "Fixed Income", "Bond ETF", 95, 0.14),
    ("LQD", "iShares Inv Grade Corp Bond", "Fixed Income", "Bond ETF", 110, 0.08),
    ("HYG", "iShares High Yield Corp Bond", "Fixed Income", "Bond ETF", 79, 0.09),
    ("MUB", "iShares National Muni Bond", "Fixed Income", "Bond ETF", 108, 0.05),
    ("SHY", "iShares 1-3 Yr Treasury", "Fixed Income", "Bond ETF", 82, 0.03),
    # --- Alternatives ---
    ("GLD", "SPDR Gold Shares", "Alternatives", "Commodity", 210, 0.14),
    ("VNQ", "Vanguard Real Estate ETF", "Alternatives", "Real Estate ETF", 92, 0.20),
    ("DBC", "Invesco DB Commodity", "Alternatives", "Commodity", 23, 0.18),
    # --- Cash ---
    ("CASH", "Cash & Sweep", "Cash", "Cash", 1.0, 0.0),
]

MODEL_ALLOC = {
    "Conservative": {"Equity": 0.30, "Fixed Income": 0.60, "Alternatives": 0.05, "Cash": 0.05},
    "Balanced":     {"Equity": 0.55, "Fixed Income": 0.35, "Alternatives": 0.05, "Cash": 0.05},
    "Growth":       {"Equity": 0.75, "Fixed Income": 0.18, "Alternatives": 0.04, "Cash": 0.03},
    "Aggressive":   {"Equity": 0.90, "Fixed Income": 0.05, "Alternatives": 0.03, "Cash": 0.02},
}
ASSET_CLASSES = ["Equity", "Fixed Income", "Alternatives", "Cash"]
ANN_DRIFT = {"Equity": 0.09, "Fixed Income": 0.02, "Alternatives": 0.04, "Cash": 0.0}

FIRST_NAMES = ["James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael",
               "Linda", "David", "Elizabeth", "William", "Barbara", "Richard", "Susan",
               "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen", "Daniel",
               "Nancy", "Matthew", "Lisa", "Anthony", "Betty", "Mark", "Sandra",
               "Steven", "Ashley", "Paul", "Kimberly", "Andrew", "Emily", "Kenneth", "Donna"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
              "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
              "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
              "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
              "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King"]
STATES = ["CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "MI", "NJ", "WA", "MA", "CO", "AZ"]
ADVISOR_NAMES = ["Olivia Bennett", "Ethan Carter", "Sophia Nguyen", "Liam Patel",
                 "Ava Rossi", "Noah Kim", "Isabella Flores", "Mason Reed",
                 "Mia Hughes", "Lucas Bauer"]


def write_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=False)


# ---------------------------------------------------------------------------
# 1. Securities + 2. Model allocations
# ---------------------------------------------------------------------------
securities = pd.DataFrame(SECURITIES,
                          columns=["symbol", "name", "asset_class", "sector", "base_price", "ann_vol"])

model_alloc = pd.DataFrame(
    [(m, ac, w) for m, d in MODEL_ALLOC.items() for ac, w in d.items()],
    columns=["model_portfolio", "asset_class", "target_weight"],
)

# ---------------------------------------------------------------------------
# 3. Prices — GBM path per symbol over the window, with engineered crashes
# ---------------------------------------------------------------------------
n = len(BIZ_DAYS)
crash_idx = n - 4  # sell-off begins ~4 business days before "today"
price_paths = {}  # symbol -> np.array of daily closes
for _, s in securities.iterrows():
    sym, base, vol, ac = s.symbol, s.base_price, s.ann_vol, s.asset_class
    if sym == "CASH":
        price_paths[sym] = np.ones(n)
        continue
    mu = ANN_DRIFT[ac]
    daily = rng.normal((mu - 0.5 * vol**2) / 252, vol / np.sqrt(252), size=n)
    if sym in CRASH_TICKERS:
        daily[crash_idx] += -0.15
        daily[crash_idx + 1] += -0.11
    path = base * np.exp(np.cumsum(daily))
    price_paths[sym] = np.round(path, 2)

price_rows = []
for i, d in enumerate(BIZ_DAYS):
    for sym in price_paths:
        price_rows.append((d, sym, float(price_paths[sym][i])))
prices = pd.DataFrame(price_rows, columns=["as_of_date", "symbol", "close_price"])
latest_price = {sym: float(price_paths[sym][-1]) for sym in price_paths}

# ---------------------------------------------------------------------------
# 4. Advisors
# ---------------------------------------------------------------------------
advisors = pd.DataFrame({
    "advisor_id": [f"ADV{i:03d}" for i in range(1, N_ADVISORS + 1)],
    "advisor_name": ADVISOR_NAMES,
    "advisor_email": [f"{nm.split()[0].lower()}.{nm.split()[1].lower()}@meridianwealth.example.com"
                      for nm in ADVISOR_NAMES],
})

# ---------------------------------------------------------------------------
# 5. Clients
# ---------------------------------------------------------------------------
n_clients = N_ADVISORS * CLIENTS_PER_ADVISOR
client_rows = []
for i in range(n_clients):
    cid = f"CL{i+1:04d}"
    adv = advisors.advisor_id[i % N_ADVISORS]
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    age = int(np.clip(rng.normal(58, 13), 28, 90))
    dob = TODAY - pd.Timedelta(days=age * 365 + int(rng.integers(0, 365)))
    # older clients skew conservative
    if age >= 70:
        risk = rng.choice(["Conservative", "Balanced", "Growth"], p=[0.5, 0.35, 0.15])
    else:
        risk = rng.choice(["Conservative", "Balanced", "Growth", "Aggressive"],
                          p=[0.15, 0.4, 0.3, 0.15])
    review_overdue = rng.random() < 0.15
    last_review = TODAY - pd.Timedelta(days=int(rng.integers(400, 780) if review_overdue
                                                 else rng.integers(20, 350)))
    client_rows.append((cid, adv, first, last,
                        f"{first.lower()}.{last.lower()}{i}@example.com",
                        f"+1-{rng.integers(200,999)}-{rng.integers(200,999)}-{rng.integers(1000,9999)}",
                        rng.choice(STATES), dob, age, risk, last_review))
clients = pd.DataFrame(client_rows, columns=[
    "client_id", "advisor_id", "first_name", "last_name", "email", "phone",
    "state", "date_of_birth", "age", "risk_profile", "last_review_date"])

# ---------------------------------------------------------------------------
# 6. Accounts  (1-3 per client; RMD-age clients guaranteed a Traditional IRA)
# ---------------------------------------------------------------------------
CUSTODIANS = ["Schwab", "Fidelity", "Pershing"]
account_rows = []
acct_counter = 1
for c in clients.itertuples():
    n_acct = int(rng.integers(1, 4))
    types = list(rng.choice(["Taxable", "Traditional IRA", "Roth IRA", "Joint Taxable"],
                            size=n_acct, replace=False)) if n_acct <= 4 else []
    if c.age >= 73 and "Traditional IRA" not in types:
        types[0] = "Traditional IRA"
    for t in types:
        account_rows.append((
            f"ACC{acct_counter:05d}", c.client_id, t,
            c.risk_profile,  # model_portfolio inherits client's risk profile
            rng.choice(CUSTODIANS),
            TODAY - pd.Timedelta(days=int(rng.integers(200, 3500))),
        ))
        acct_counter += 1
accounts = pd.DataFrame(account_rows, columns=[
    "account_id", "client_id", "account_type", "model_portfolio", "custodian", "open_date"])

# ---------------------------------------------------------------------------
# 7. Holdings (base, static quantities) -> seeded conditions
# ---------------------------------------------------------------------------
eq_stocks = securities[(securities.asset_class == "Equity") & (~securities.symbol.str.contains("ETF", na=False))]
equity_syms = securities[securities.asset_class == "Equity"].symbol.tolist()
equity_etfs = ["SPY", "VOO", "VTI", "QQQ", "IWM", "VEA", "VWO"]
equity_single = [s for s in equity_syms if s not in equity_etfs]
fi_syms = securities[securities.asset_class == "Fixed Income"].symbol.tolist()
alt_syms = securities[securities.asset_class == "Alternatives"].symbol.tolist()

n_acct_total = len(accounts)
drift_accts = set(rng.choice(accounts.account_id, size=int(0.15 * n_acct_total), replace=False))
conc_accts = set(rng.choice(accounts.account_id, size=int(0.08 * n_acct_total), replace=False))
cash_accts = set(rng.choice(accounts.account_id, size=int(0.10 * n_acct_total), replace=False))

holding_rows = []
for a in accounts.itertuples():
    base = np.array([MODEL_ALLOC[a.model_portfolio][ac] for ac in ASSET_CLASSES], dtype=float)
    w = np.clip(base + rng.normal(0, 0.04, 4), 0.005, None)
    if a.account_id in drift_accts:
        if rng.random() < 0.5:  # equity-heavy drift
            w[0] += 0.25; w[1] = max(w[1] - 0.20, 0.01)
        else:                    # cash-heavy drift
            w[3] += 0.22
    if a.account_id in cash_accts:
        w[3] = rng.uniform(0.15, 0.30)
    w = w / w.sum()
    account_value = float(np.clip(rng.lognormal(mean=12.6, sigma=0.7), 40_000, 5_000_000))
    class_dollars = dict(zip(ASSET_CLASSES, w * account_value))

    def add_holding(sym, dollars, taxable, force_loss=False):
        if dollars < 50:
            return
        px = latest_price[sym]
        qty = round(dollars / px, 4)
        if force_loss:
            factor = rng.uniform(1.20, 1.60)          # cost above market -> loss
        else:
            factor = rng.uniform(0.45, 1.10)          # mostly gains
        avg_cost = round(px * factor, 4)
        acquired = TODAY - pd.Timedelta(days=int(rng.integers(60, 2200)))
        holding_rows.append((a.account_id, sym, qty, avg_cost,
                             round(qty * avg_cost, 2), acquired))

    taxable = a.account_type in ("Taxable", "Joint Taxable")

    # --- Equity sleeve ---
    eq_dollars = class_dollars["Equity"]
    if a.account_id in conc_accts and eq_dollars > 0:
        crash_sym = rng.choice(CRASH_TICKERS)
        add_holding(crash_sym, eq_dollars * rng.uniform(0.55, 0.80), taxable)
        others = rng.choice([s for s in equity_syms if s != crash_sym],
                            size=int(rng.integers(2, 5)), replace=False)
        rem = eq_dollars * (1 - 0.67)
        for s in others:
            add_holding(s, rem / len(others), taxable)
    elif eq_dollars > 0:
        k = int(rng.integers(3, 7))
        picks = list(rng.choice(equity_etfs, size=min(2, k), replace=False))
        picks += list(rng.choice(equity_single, size=k - len(picks), replace=False))
        splits = rng.dirichlet(np.ones(len(picks)))
        loss_idx = rng.integers(0, len(picks)) if (taxable and rng.random() < 0.35) else -1
        for j, s in enumerate(picks):
            add_holding(s, eq_dollars * splits[j], taxable, force_loss=(j == loss_idx))

    # --- Fixed income sleeve ---
    fi_dollars = class_dollars["Fixed Income"]
    if fi_dollars > 0:
        picks = list(rng.choice(fi_syms, size=int(rng.integers(1, 4)), replace=False))
        splits = rng.dirichlet(np.ones(len(picks)))
        for j, s in enumerate(picks):
            add_holding(s, fi_dollars * splits[j], taxable)

    # --- Alternatives sleeve ---
    if class_dollars["Alternatives"] > 0:
        add_holding(rng.choice(alt_syms), class_dollars["Alternatives"], taxable)

    # --- Cash sleeve ---
    cash_dollars = class_dollars["Cash"]
    if cash_dollars > 0:
        holding_rows.append((a.account_id, "CASH", round(cash_dollars, 2), 1.0,
                             round(cash_dollars, 2), a.open_date))

holdings = pd.DataFrame(holding_rows, columns=[
    "account_id", "symbol", "quantity", "avg_cost", "cost_basis_total", "acquired_date"])

# ---------------------------------------------------------------------------
# 8. Positions — value holdings at each day's close (daily custodian snapshot)
# ---------------------------------------------------------------------------
pos_frames = []
for i, d in enumerate(BIZ_DAYS):
    day_px = {sym: float(price_paths[sym][i]) for sym in price_paths}
    f = holdings.copy()
    f["as_of_date"] = d
    f["close_price"] = f["symbol"].map(day_px)
    f["market_value"] = (f["quantity"] * f["close_price"]).round(2)
    pos_frames.append(f[["as_of_date", "account_id", "symbol", "quantity", "close_price",
                         "market_value", "avg_cost", "cost_basis_total", "acquired_date"]])
positions_all = pd.concat(pos_frames, ignore_index=True)

# ---------------------------------------------------------------------------
# 9. Transactions — light daily buy/sell/contribution flow
# ---------------------------------------------------------------------------
txn_rows = []
txn_id = 1
acct_ids = accounts.account_id.tolist()
for i, d in enumerate(BIZ_DAYS):
    for _ in range(int(rng.integers(2, 6))):
        acct = rng.choice(acct_ids)
        side = rng.choice(["BUY", "SELL", "CONTRIBUTION", "WITHDRAWAL", "DIVIDEND"],
                          p=[0.35, 0.25, 0.15, 0.10, 0.15])
        if side in ("CONTRIBUTION", "WITHDRAWAL"):
            sym, px = "CASH", 1.0
            qty = round(float(rng.uniform(500, 25000)), 2)
        else:
            sym = rng.choice(equity_syms + fi_syms)
            px = float(price_paths[sym][i])
            qty = round(float(rng.uniform(5, 300)), 4)
        txn_rows.append((f"TXN{txn_id:06d}", d, acct, sym, side, qty, round(px, 2),
                         round(qty * px, 2)))
        txn_id += 1
transactions = pd.DataFrame(txn_rows, columns=[
    "transaction_id", "trade_date", "account_id", "symbol", "side", "quantity",
    "price", "amount"])

# ---------------------------------------------------------------------------
# Write everything
# ---------------------------------------------------------------------------
write_parquet(advisors, OUT / "reference" / "advisors.parquet")
write_parquet(clients, OUT / "reference" / "clients.parquet")
write_parquet(accounts, OUT / "reference" / "accounts.parquet")
write_parquet(securities.drop(columns=["base_price", "ann_vol"]),
              OUT / "reference" / "securities.parquet")
write_parquet(model_alloc, OUT / "reference" / "model_allocations.parquet")

for d in BIZ_DAYS:
    ds = d.strftime("%Y-%m-%d")
    write_parquet(prices[prices.as_of_date == d], OUT / "prices" / f"prices_{ds}.parquet")
    write_parquet(positions_all[positions_all.as_of_date == d],
                  OUT / "positions" / f"positions_{ds}.parquet")
    tx = transactions[transactions.trade_date == d]
    if len(tx):
        write_parquet(tx, OUT / "transactions" / f"transactions_{ds}.parquet")

# ---------------------------------------------------------------------------
# Sanity check — how many clients will surface in the worklist, by signal
# ---------------------------------------------------------------------------
latest = positions_all[positions_all.as_of_date == BIZ_DAYS[-1]].merge(
    securities[["symbol", "asset_class"]], on="symbol")
latest = latest.merge(accounts[["account_id", "client_id", "account_type", "model_portfolio"]],
                      on="account_id")
acct_val = latest.groupby("account_id")["market_value"].sum().rename("acct_val")
latest = latest.merge(acct_val, on="account_id")

# drift
acw = (latest.groupby(["account_id", "model_portfolio", "asset_class"])["market_value"].sum()
       / latest.groupby("account_id")["market_value"].transform("sum").groupby(
           [latest.account_id]).first())
acw = latest.groupby(["account_id", "client_id", "model_portfolio", "asset_class"]).agg(
    mv=("market_value", "sum"), av=("acct_val", "first")).reset_index()
acw["weight"] = acw.mv / acw.av
acw = acw.merge(model_alloc, on=["model_portfolio", "asset_class"])
acw["drift"] = (acw.weight - acw.target_weight).abs()
drift_clients = acw[acw.drift > 0.10].client_id.nunique()

# concentration + recent crash
ret5 = {sym: (price_paths[sym][-1] / price_paths[sym][-6] - 1) for sym in price_paths}
latest["pos_weight"] = latest.market_value / latest.acct_val
latest["ret5"] = latest.symbol.map(ret5)
conc_clients = latest[(latest.pos_weight > 0.20) & (latest.ret5 < -0.12)].client_id.nunique()

# tax-loss harvesting (taxable only)
latest["unrealized"] = latest.market_value - latest.cost_basis_total
tlh_clients = latest[(latest.account_type.isin(["Taxable", "Joint Taxable"]))
                     & (latest.unrealized < -2000)].client_id.nunique()

# cash drag
cashw = latest[latest.asset_class == "Cash"].copy()
cashw["cash_w"] = cashw.market_value / cashw.acct_val
cash_clients = cashw[cashw.cash_w > 0.12].client_id.nunique()

# RMD
ira = accounts[accounts.account_type == "Traditional IRA"].client_id.unique()
rmd_clients = clients[(clients.age >= 73) & (clients.client_id.isin(ira))].client_id.nunique()

# review overdue
review_clients = int((clients.last_review_date < (TODAY - pd.Timedelta(days=365))).sum())

flagged = set(acw[acw.drift > 0.10].client_id) \
    | set(latest[(latest.pos_weight > 0.20) & (latest.ret5 < -0.12)].client_id) \
    | set(latest[(latest.account_type.isin(["Taxable", "Joint Taxable"])) & (latest.unrealized < -2000)].client_id) \
    | set(cashw[cashw.cash_w > 0.12].client_id) \
    | set(clients[(clients.age >= 73) & (clients.client_id.isin(ira))].client_id) \
    | set(clients[clients.last_review_date < (TODAY - pd.Timedelta(days=365))].client_id)

print("=" * 62)
print("  WEALTH-ADVISOR WORKLIST — synthetic data generated")
print("=" * 62)
print(f"  window            : {BIZ_DAYS[0].date()} -> {BIZ_DAYS[-1].date()}  ({n} business days)")
print(f"  advisors          : {len(advisors)}")
print(f"  clients           : {len(clients)}")
print(f"  accounts          : {len(accounts)}")
print(f"  securities        : {len(securities)}")
print(f"  holdings (base)   : {len(holdings)}")
print(f"  positions (daily) : {len(positions_all):,}  ({len(holdings)} x {n} days)")
print(f"  transactions      : {len(transactions)}")
print("-" * 62)
print("  Clients surfacing in the worklist, by signal:")
print(f"    allocation drift (>10%)          : {drift_clients}")
print(f"    concentrated + crashed position  : {conc_clients}")
print(f"    tax-loss harvesting (loss>$2k)   : {tlh_clients}")
print(f"    cash drag (>12%)                 : {cash_clients}")
print(f"    RMD age (>=73 w/ Traditional IRA): {rmd_clients}")
print(f"    review overdue (>12 months)      : {review_clients}")
print(f"  --> distinct clients needing a call: {len(flagged)} of {len(clients)}")
print("=" * 62)
print(f"  output: {OUT}")
