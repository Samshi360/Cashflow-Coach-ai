import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ----------------------------
# Helpers: parsing & cleaning
# ----------------------------
REQUIRED_COLS = {"date", "description", "amount"}

DEFAULT_CATEGORY_RULES = [
    ("Income", r"(payroll|salary|paycheck|income|bonus|transfer in)"),
    ("Rent/Housing", r"(rent|lease|mortgage|landlord)"),
    ("Utilities", r"(hydro|electric|water|utility|gas bill|internet|phone|bell|rogers|telus)"),
    ("Groceries", r"(grocery|superstore|walmart|loblaws|no frills|costco|metro|freshco|sobeys)"),
    ("Dining", r"(restaurant|cafe|coffee|tim hortons|starbucks|uber eats|doordash|skip)"),
    ("Transport", r"(uber|lyft|gas station|petro|esso|shell|transit|ttc|go train|parking)"),
    ("Subscriptions", r"(netflix|spotify|apple|google|prime|subscription)"),
    ("Insurance", r"(insurance|premium)"),
    ("Debt/Loans", r"(loan|credit card payment|interest|minimum payment)"),
    ("Shopping", r"(amazon|shopping|store|mall|ikea)"),
    ("Healthcare", r"(pharmacy|drug|clinic|dental|health)"),
]

FIXED_LIKE_CATEGORIES = {"Rent/Housing", "Utilities", "Subscriptions", "Insurance", "Debt/Loans"}


def safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def normalize_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    df["date"] = safe_to_datetime(df["date"])
    df = df.dropna(subset=["date"])

    df["description"] = df["description"].astype(str).fillna("").str.strip()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["amount"])

    if "category" not in df.columns:
        df["category"] = ""

    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    return df.sort_values("date")


def auto_categorize(df: pd.DataFrame, rules=DEFAULT_CATEGORY_RULES) -> pd.DataFrame:
    df = df.copy()
    mask_missing = df["category"].astype(str).str.strip().eq("")
    desc = df.loc[mask_missing, "description"].str.lower()

    assigned = pd.Series(["Other"] * mask_missing.sum(), index=desc.index)
    for cat, pattern in rules:
        hits = desc.str.contains(pattern, regex=True, na=False)
        assigned.loc[hits] = cat

    income_like = (df.loc[mask_missing, "amount"] > 0) & (assigned == "Other")
    assigned.loc[income_like] = "Income"

    df.loc[mask_missing, "category"] = assigned
    return df


# ----------------------------
# Analytics
# ----------------------------
def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    by_month = df.groupby("month")["amount"].sum().rename("net_cashflow").to_frame()
    by_month["income"] = df[df["amount"] > 0].groupby("month")["amount"].sum()
    by_month["expenses"] = -df[df["amount"] < 0].groupby("month")["amount"].sum()
    by_month = by_month.fillna(0.0)
    by_month["savings_rate"] = np.where(
        by_month["income"] > 0,
        by_month["net_cashflow"] / by_month["income"],
        np.nan,
    )
    return by_month.reset_index().sort_values("month")


def category_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    exp = df[df["amount"] < 0].copy()
    exp["spend"] = -exp["amount"]
    out = exp.groupby("category")["spend"].sum().sort_values(ascending=False).reset_index()
    out["share"] = out["spend"] / out["spend"].sum() if out["spend"].sum() > 0 else 0
    return out


def risk_scores(df: pd.DataFrame, cash_on_hand: float = 0.0) -> dict:
    ms = monthly_summary(df)
    if ms.empty:
        return {
            "liquidity": 0, "rigidity": 0, "income_volatility": 0, "overspend_streak": 0,
            "avg_monthly_income": 0, "avg_monthly_expenses": 0, "fixed_share": 0,
            "max_deficit_streak_months": 0, "avg_savings_rate": 0
        }

    avg_exp = ms["expenses"].replace(0, np.nan).mean()
    avg_inc = ms["income"].replace(0, np.nan).mean()

    avg_exp = float(avg_exp) if not np.isnan(avg_exp) else 0.0
    avg_inc = float(avg_inc) if not np.isnan(avg_inc) else 0.0

    if avg_exp > 0:
        buffer_months = cash_on_hand / avg_exp
        liquidity = np.clip(100 * (1 - (buffer_months - 0.5) / (3.0 - 0.5)), 0, 100)
    else:
        liquidity = 20

    exp = df[df["amount"] < 0].copy()
    exp["spend"] = -exp["amount"]
    total_spend = exp["spend"].sum()
    fixed_spend = exp[exp["category"].isin(FIXED_LIKE_CATEGORIES)]["spend"].sum()
    fixed_share = (fixed_spend / total_spend) if total_spend > 0 else 0.0
    rigidity = np.clip(100 * (fixed_share - 0.25) / (0.75 - 0.25), 0, 100)

    inc_series = ms["income"].values
    inc_nonzero = inc_series[inc_series > 0]
    if len(inc_nonzero) >= 2 and np.mean(inc_nonzero) > 0:
        cv = float(np.std(inc_nonzero) / np.mean(inc_nonzero))
        income_vol = np.clip(100 * cv / 0.5, 0, 100)
    else:
        income_vol = 30

    streak = 0
    max_streak = 0
    for x in ms["net_cashflow"].values:
        if x < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    overspend = np.clip(25 * max_streak, 0, 100)

    sr = ms["savings_rate"].replace([np.inf, -np.inf], np.nan).dropna()
    avg_savings_rate = float(sr.mean()) if len(sr) else 0.0

    return {
        "liquidity": int(round(liquidity)),
        "rigidity": int(round(rigidity)),
        "income_volatility": int(round(income_vol)),
        "overspend_streak": int(round(overspend)),
        "avg_monthly_income": round(avg_inc, 2),
        "avg_monthly_expenses": round(avg_exp, 2),
        "fixed_share": round(fixed_share, 2),
        "max_deficit_streak_months": int(max_streak),
        "avg_savings_rate": round(avg_savings_rate, 3),
    }


def runway_projection(df: pd.DataFrame, cash_on_hand: float) -> dict:
    ms = monthly_summary(df)
    if ms.empty:
        return {"30d": None, "60d": None, "90d": None, "avg_monthly_net": None}

    avg_monthly_net = float(ms["net_cashflow"].mean())

    def proj(days: int):
        months = days / 30.0
        return cash_on_hand + avg_monthly_net * months

    return {
        "30d": round(proj(30), 2),
        "60d": round(proj(60), 2),
        "90d": round(proj(90), 2),
        "avg_monthly_net": round(avg_monthly_net, 2),
    }


# ----------------------------
# Feature A: What-if Scenario Simulator
# ----------------------------
def apply_what_if_scenario(
    df: pd.DataFrame,
    income_change_pct: float,
    dining_cut_pct: float,
    shopping_cut_pct: float,
    rent_change_monthly: float
) -> pd.DataFrame:
    dfx = df.copy()

    income_mask = dfx["amount"] > 0
    if income_change_pct != 0:
        dfx.loc[income_mask, "amount"] = dfx.loc[income_mask, "amount"] * (1 + income_change_pct / 100.0)

    if dining_cut_pct > 0:
        dining_mask = (dfx["category"] == "Dining") & (dfx["amount"] < 0)
        dfx.loc[dining_mask, "amount"] = dfx.loc[dining_mask, "amount"] * (1 - dining_cut_pct / 100.0)

    if shopping_cut_pct > 0:
        shop_mask = (dfx["category"] == "Shopping") & (dfx["amount"] < 0)
        dfx.loc[shop_mask, "amount"] = dfx.loc[shop_mask, "amount"] * (1 - shopping_cut_pct / 100.0)

    if rent_change_monthly != 0:
        rent_mask = (dfx["category"] == "Rent/Housing") & (dfx["amount"] < 0)
        rent_months = dfx.loc[rent_mask, "month"].unique()
        for m in rent_months:
            month_mask = rent_mask & (dfx["month"] == m)
            n = int(month_mask.sum())
            if n > 0:
                per_txn_adjustment = rent_change_monthly / n
                dfx.loc[month_mask, "amount"] = dfx.loc[month_mask, "amount"] - per_txn_adjustment

    return dfx


# ----------------------------
# Feature B: Financial Health Score
# ----------------------------
def financial_health_score(rs: dict) -> dict:
    liquidity = rs.get("liquidity", 0)
    rigidity = rs.get("rigidity", 0)
    income_vol = rs.get("income_volatility", 0)
    overspend = rs.get("overspend_streak", 0)

    avg_sr = rs.get("avg_savings_rate", 0.0)
    savings_boost = np.clip((avg_sr - 0.00) / 0.10 * 10, 0, 10)

    weighted_risk = 0.35 * liquidity + 0.25 * rigidity + 0.20 * income_vol + 0.20 * overspend
    score = int(np.clip(round(100 - weighted_risk + savings_boost), 0, 100))

    if score >= 80:
        label = "Healthy"
    elif score >= 60:
        label = "Good"
    elif score >= 40:
        label = "Watchlist"
    else:
        label = "High Risk"

    return {"score": score, "label": label, "savings_boost": round(float(savings_boost), 1)}


# ----------------------------
# Feature C: Behavioral Pattern Flags
# ----------------------------
def behavioral_flags(df_base: pd.DataFrame, df_scn: pd.DataFrame) -> list:
    flags = []
    ms_base = monthly_summary(df_base)

    if not ms_base.empty:
        deficits = (ms_base["net_cashflow"] < 0).sum()
        if deficits >= 2:
            flags.append(f"Repeated deficit months detected ({int(deficits)} months).")

        if len(ms_base) >= 2:
            last2 = ms_base.tail(2)
            inc_change = last2["income"].iloc[-1] - last2["income"].iloc[0]

            exp = df_base[df_base["amount"] < 0].copy()
            exp["spend"] = -exp["amount"]
            dining_by_month = exp[exp["category"] == "Dining"].groupby("month")["spend"].sum()

            if len(dining_by_month) >= 2:
                d_last2 = dining_by_month.reindex(last2["month"]).fillna(0)
                dining_change = d_last2.iloc[-1] - d_last2.iloc[0]
                if inc_change < 0 and dining_change > 0:
                    flags.append("Dining spend rose while income fell (behavioral mismatch).")

        exp = df_base[df_base["amount"] < 0].copy()
        exp["spend"] = -exp["amount"]
        total_spend = exp["spend"].sum()
        fixed_spend = exp[exp["category"].isin(FIXED_LIKE_CATEGORIES)]["spend"].sum()
        if total_spend > 0 and (fixed_spend / total_spend) > 0.6:
            flags.append("Fixed costs consume a large share of spending (limited flexibility).")

        shop = df_base[(df_base["category"] == "Shopping") & (df_base["amount"] < 0)].copy()
        if len(shop) >= 6:
            shop["day"] = shop["date"].dt.day
            mid = shop[(shop["day"] >= 10) & (shop["day"] <= 20)]["amount"].abs().sum()
            early_late = shop[(shop["day"] < 10) | (shop["day"] > 20)]["amount"].abs().sum()
            if mid > early_late:
                flags.append("Shopping spend clusters mid-month (possible impulse window).")

    if not df_scn.equals(df_base):
        flags.append("Scenario applied: charts and scores reflect your what-if assumptions.")

    return flags[:6]


# ----------------------------
# Feature D: Priority Actions Box
# ----------------------------
def priority_actions(rs: dict, cb: pd.DataFrame) -> list:
    actions = []

    risk_pairs = [
        ("Liquidity risk", rs.get("liquidity", 0)),
        ("Expense rigidity risk", rs.get("rigidity", 0)),
        ("Income volatility risk", rs.get("income_volatility", 0)),
        ("Overspending streak risk", rs.get("overspend_streak", 0)),
    ]
    risk_pairs.sort(key=lambda x: x[1], reverse=True)
    top_risks = [r[0] for r in risk_pairs[:2]]

    avg_exp = rs.get("avg_monthly_expenses", 0.0)
    fixed_share = rs.get("fixed_share", 0.0)

    if "Liquidity risk" in top_risks:
        target_buffer = round(avg_exp * 2.0, 0) if avg_exp > 0 else 1000
        actions.append({
            "title": "Build a 2-month buffer",
            "detail": f"Aim for ~${target_buffer:,.0f} cash buffer to reduce liquidity stress."
        })

    if "Expense rigidity risk" in top_risks:
        actions.append({
            "title": "Lower fixed costs (even slightly)",
            "detail": f"Fixed-like share is ~{int(fixed_share*100)}%. Renegotiate one bill or remove 1 subscription."
        })

    if cb is not None and not cb.empty:
        top_cat = cb.iloc[0]["category"]
        top_spend = cb.iloc[0]["spend"]
        cut_target = round(top_spend * 0.15, 0)
        actions.append({
            "title": f"Cut {top_cat} by ~15%",
            "detail": f"Reducing {top_cat} by ~${cut_target:,.0f}/month improves runway and score quickly."
        })

    while len(actions) < 3:
        actions.append({
            "title": "Set a weekly discretionary cap",
            "detail": "Pick a number you can stick to (e.g., $60/week). Consistency reduces overspending streaks."
        })

    return actions[:3]


# ----------------------------
# Feature E: Target Salary / Income Threshold
# ----------------------------
def target_salary_for_healthy_score(
    rs: dict,
    cash_on_hand: float,
    target_savings_rate: float = 0.10,
    target_buffer_months: float = 3.0,
    buffer_build_months: int = 12
) -> dict:
    avg_exp = float(rs.get("avg_monthly_expenses", 0.0))
    avg_inc = float(rs.get("avg_monthly_income", 0.0))

    if avg_exp <= 0:
        return {"target_monthly_income": None, "gap_monthly_income": None}

    desired_buffer = target_buffer_months * avg_exp
    buffer_gap = max(desired_buffer - cash_on_hand, 0.0)
    buffer_build_per_month = buffer_gap / max(buffer_build_months, 1)

    denom = max(1.0 - target_savings_rate, 0.01)
    target_income = (avg_exp + buffer_build_per_month) / denom
    gap = max(target_income - avg_inc, 0.0)

    return {
        "target_monthly_income": round(target_income, 2),
        "gap_monthly_income": round(gap, 2),
        "assumptions": {
            "target_savings_rate": target_savings_rate,
            "target_buffer_months": target_buffer_months,
            "buffer_build_months": buffer_build_months
        }
    }


# ----------------------------
# GenAI Coach (optional)
# ----------------------------
def generate_ai_coaching(summary_text: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return (
            "AI coaching is currently in offline mode (no API key set).\n\n"
            "Coaching summary:\n"
            "- Focus first on reducing your highest risk score.\n"
            "- Aim for a 1–3 month emergency buffer (or more if income is volatile).\n"
            "- Reduce fixed-like costs where possible and set a weekly discretionary cap.\n"
            "- If you see consecutive deficit months, set an immediate cut plan for non-essentials."
        )

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    system = (
        "You are a cautious personal finance coach. "
        "You provide general educational guidance, not professional financial advice. "
        "Do not recommend risky or illegal actions. "
        "Be practical and kind. Use short bullet points and a 30/60/90-day plan."
    )

    user = (
        "Analyze the following cash-flow summary and write:\n"
        "1) Key risks (plain English)\n"
        "2) 3 prioritized actions\n"
        "3) A 30/60/90-day plan\n"
        "4) One sentence caution on limitations\n\n"
        f"DATA:\n{summary_text}"
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.4,
    )
    return resp.choices[0].message.content.strip()


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="CashFlow Coach AI", layout="wide")
st.title("CashFlow Coach AI — Personal Cash-Flow Risk & Behavior Coach")
st.caption(
    "Upload a transactions CSV to get cash-flow insights, risk flags, runway projection, and coaching. "
    "Educational only — not professional financial advice."
)

with st.sidebar:
    st.header("Inputs")
    cash_on_hand = st.number_input("Current cash on hand ($)", min_value=0.0, value=2000.0, step=100.0)
    do_autocat = st.checkbox("Auto-categorize missing categories", value=True)

    st.divider()
    st.subheader("What-If Scenario Simulator")
    income_change_pct = st.slider("Income change (%)", min_value=-30, max_value=30, value=0, step=1)
    dining_cut_pct = st.slider("Reduce Dining spend (%)", min_value=0, max_value=60, value=0, step=5)
    shopping_cut_pct = st.slider("Reduce Shopping spend (%)", min_value=0, max_value=60, value=0, step=5)
    rent_change_monthly = st.slider("Rent change ($/month)", min_value=-300, max_value=300, value=0, step=25)

    st.divider()
    st.subheader("Sample CSV")
    if st.button("Load built-in sample data"):
        sample = pd.DataFrame([
            {"date": "2025-08-01", "description": "Paycheck", "amount": 3500},
            {"date": "2025-08-02", "description": "Rent", "amount": -1650},
            {"date": "2025-08-03", "description": "Groceries - Walmart", "amount": -180},
            {"date": "2025-08-05", "description": "Phone - Rogers", "amount": -75},
            {"date": "2025-08-06", "description": "Coffee - Starbucks", "amount": -22},
            {"date": "2025-08-10", "description": "Transit pass", "amount": -120},
            {"date": "2025-09-01", "description": "Paycheck", "amount": 3500},
            {"date": "2025-09-02", "description": "Rent", "amount": -1650},
            {"date": "2025-09-03", "description": "Groceries - Loblaws", "amount": -220},
            {"date": "2025-09-08", "description": "Netflix", "amount": -19},
            {"date": "2025-09-15", "description": "Restaurant", "amount": -85},
            {"date": "2025-10-01", "description": "Paycheck", "amount": 3200},
            {"date": "2025-10-02", "description": "Rent", "amount": -1650},
            {"date": "2025-10-04", "description": "Utilities - Hydro", "amount": -95},
            {"date": "2025-10-07", "description": "Groceries - Costco", "amount": -240},
            {"date": "2025-10-22", "description": "Amazon", "amount": -140},
        ])
        st.session_state["df_uploaded"] = sample

uploaded = st.file_uploader("Upload transactions CSV", type=["csv"])

df = None
if uploaded is not None:
    try:
        df_raw = pd.read_csv(uploaded)
        df = normalize_transactions(df_raw)
    except Exception as e:
        st.error(f"Could not read/parse CSV: {e}")
elif "df_uploaded" in st.session_state:
    df = normalize_transactions(st.session_state["df_uploaded"])

if df is None:
    st.info("Upload a CSV or click *Load built-in sample data* in the sidebar to start.")
    st.stop()

if do_autocat:
    df = auto_categorize(df)

df_base = df.copy()
df_scn = apply_what_if_scenario(
    df_base,
    income_change_pct=income_change_pct,
    dining_cut_pct=dining_cut_pct,
    shopping_cut_pct=shopping_cut_pct,
    rent_change_monthly=rent_change_monthly
)

left, right = st.columns([1.25, 1])

with left:
    st.subheader("Transactions (normalized)")
    st.dataframe(df.head(50), use_container_width=True)

    st.subheader("Monthly cash-flow summary (Scenario applied)")
    ms = monthly_summary(df_scn)
    if ms.empty:
        st.error("No data available after applying the scenario. Adjust sliders or upload more transactions.")
        st.stop()

    st.dataframe(ms, use_container_width=True)

    fig = plt.figure()
    plt.plot(ms["month"], ms["net_cashflow"], marker="o")
    plt.axhline(0, linewidth=1)
    plt.title("Net Cash Flow by Month (Scenario)")
    plt.xlabel("Month")
    plt.ylabel("Net Cash Flow")
    st.pyplot(fig)

    st.subheader("Spending breakdown (expenses only, Scenario)")
    cb = category_breakdown(df_scn)
    st.dataframe(cb, use_container_width=True)

with right:
    rs_base = risk_scores(df_base, cash_on_hand=cash_on_hand)
    rs = risk_scores(df_scn, cash_on_hand=cash_on_hand)
    hs = financial_health_score(rs)

    st.subheader("Financial Health Score")
    st.metric("Health score", f"{hs['score']}/100")
    st.write(f"**Status:** {hs['label']} | **Savings boost:** +{hs['savings_boost']} pts")

    st.subheader("Risk dashboard (Scenario)")
    st.metric("Liquidity risk", f"{rs['liquidity']}/100")
    st.metric("Expense rigidity risk", f"{rs['rigidity']}/100")
    st.metric("Income volatility risk", f"{rs['income_volatility']}/100")
    st.metric("Overspending streak risk", f"{rs['overspend_streak']}/100")

    st.markdown("### Before vs After (Scenario Impact)")
    c1, c2 = st.columns(2)
    c1.metric("Liquidity risk (Before)", f"{rs_base['liquidity']}/100")
    c2.metric("Liquidity risk (After)", f"{rs['liquidity']}/100")
    c3, c4 = st.columns(2)
    c3.metric("Rigidity risk (Before)", f"{rs_base['rigidity']}/100")
    c4.metric("Rigidity risk (After)", f"{rs['rigidity']}/100")

    st.caption(
        f"Avg monthly income: ${rs['avg_monthly_income']} | "
        f"Avg monthly expenses: ${rs['avg_monthly_expenses']} | "
        f"Fixed-like share: {int(rs['fixed_share']*100)}% | "
        f"Avg savings rate: {int(rs['avg_savings_rate']*100)}%"
    )

    st.subheader("Behavioral pattern flags")
    flags = behavioral_flags(df_base, df_scn)
    if flags:
        for f in flags:
            st.info(f)
    else:
        st.success("No major behavioral red flags detected from this dataset.")

    st.subheader("Priority actions (Top 3)")
    acts = priority_actions(rs, cb)
    for i, a in enumerate(acts, start=1):
        st.write(f"**{i}. {a['title']}** — {a['detail']}")

    st.subheader("Runway projection (Scenario)")
    rp_base = runway_projection(df_base, cash_on_hand=cash_on_hand)
    rp = runway_projection(df_scn, cash_on_hand=cash_on_hand)

    r1, r2, r3 = st.columns(3)
    r1.metric("30 days", f"${rp['30d']}")
    r2.metric("60 days", f"${rp['60d']}")
    r3.metric("90 days", f"${rp['90d']}")

    st.caption(f"Before scenario → 30d: ${rp_base['30d']} | 60d: ${rp_base['60d']} | 90d: ${rp_base['90d']}")
    st.caption(f"Avg monthly net cashflow (historical): ${rp.get('avg_monthly_net', 0)}")

    st.subheader("Target salary / income threshold")
    target = target_salary_for_healthy_score(rs=rs, cash_on_hand=cash_on_hand)
    if target.get("target_monthly_income") is None:
        st.write("Not enough history to estimate a target income.")
    else:
        st.write(
            f"To aim for a **healthy** profile with ~**10% saving** and building a **3-month buffer** over **12 months**, "
            f"target about **${target['target_monthly_income']:,.0f}/month** income."
        )
        if target["gap_monthly_income"] > 0:
            st.warning(f"That is about **${target['gap_monthly_income']:,.0f}/month** above your current estimated average income.")
        else:
            st.success("Your estimated income is already at/above this target under these assumptions.")
        st.caption("Educational heuristic only — not financial advice.")

    st.subheader("AI coaching summary")
    summary_blob = {
        "cash_on_hand": cash_on_hand,
        "scenario": {
            "income_change_pct": income_change_pct,
            "dining_cut_pct": dining_cut_pct,
            "shopping_cut_pct": shopping_cut_pct,
            "rent_change_monthly": rent_change_monthly,
        },
        "financial_health": hs,
        "risk_scores": {k: rs[k] for k in ["liquidity", "rigidity", "income_volatility", "overspend_streak"]},
        "behavioral_flags": flags,
        "priority_actions": acts,
        "context": {
            "avg_monthly_income": rs["avg_monthly_income"],
            "avg_monthly_expenses": rs["avg_monthly_expenses"],
            "fixed_like_share": rs["fixed_share"],
            "avg_savings_rate": rs["avg_savings_rate"],
            "runway_projection": rp,
            "target_salary_estimate": target,
        },
        "top_expense_categories": cb.head(5).to_dict(orient="records"),
    }
    summary_text = pd.Series(summary_blob).to_string()

    if st.button("Generate coaching note"):
        with st.spinner("Generating coaching note..."):
            coaching = generate_ai_coaching(summary_text)
        st.text_area("Coach output", coaching, height=360)

st.divider()
st.caption(
    "Note: This app provides educational insights based on uploaded transactions. "
    "It may mis-categorize merchants and cannot account for future shocks. "
    "For major decisions, consult a qualified professional."
)
