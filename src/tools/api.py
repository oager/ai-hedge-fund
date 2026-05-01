"""
yfinance-backed replacement for src/tools/api.py
Drop-in: same function signatures, no FINANCIAL_DATASETS_API_KEY required.
"""
import datetime
import logging
import os
import warnings

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)
logger = logging.getLogger(__name__)

from src.data.cache import get_cache
from src.data.models import (
    CompanyFacts,
    CompanyFactsResponse,
    CompanyNews,
    CompanyNewsResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    InsiderTrade,
    InsiderTradeResponse,
    LineItem,
    LineItemResponse,
    Price,
    PriceResponse,
)

_cache = get_cache()

# ---------------------------------------------------------------------------
# yfinance helpers
# ---------------------------------------------------------------------------

import threading
_yf_cache: dict = {}
_yf_lock = threading.Lock()


def _get_ticker(symbol: str) -> yf.Ticker:
    """Return a cached yf.Ticker, pre-fetching all data in a thread-safe way."""
    if symbol in _yf_cache:
        return _yf_cache[symbol]
    with _yf_lock:
        if symbol not in _yf_cache:
            t = yf.Ticker(symbol)
            # Pre-fetch all expensive properties under the lock so parallel
            # agents get fully-populated data instead of racing on the network.
            try:
                _ = t.info
                _ = t.income_stmt
                _ = t.balance_sheet
                _ = t.cashflow
                _ = t.quarterly_income_stmt
                _ = t.quarterly_balance_sheet
                _ = t.quarterly_cashflow
            except Exception:
                pass
            _yf_cache[symbol] = t
    return _yf_cache[symbol]


def _safe_row(df: pd.DataFrame, candidates: list[str]):
    """Return the first row that exists in df.index, or None."""
    if df is None or df.empty:
        return None
    for name in candidates:
        if name in df.index:
            return df.loc[name]
    return None


def _val(series, col_idx: int = 0):
    """Safely get a scalar from a Series by positional index."""
    if series is None:
        return None
    try:
        v = series.iloc[col_idx]
        return None if pd.isna(v) else float(v)
    except Exception:
        return None


def _ttm_flow(q_df: pd.DataFrame, candidates: list[str]) -> float | None:
    """Sum last 4 quarterly values for a flow statement item (TTM)."""
    row = _safe_row(q_df, candidates)
    if row is None:
        return None
    try:
        vals = [float(v) for v in row.iloc[:4] if not pd.isna(v)]
        return sum(vals) if vals else None
    except Exception:
        return None


# Map from financialdatasets line-item names → (statement, [yfinance row candidates])
# statement: 'income' | 'balance' | 'cashflow' | 'q_income' | 'q_balance' | 'q_cashflow'
_ITEM_MAP: dict[str, tuple[str, list[str]]] = {
    # Income statement
    "revenue":                           ("income", ["Total Revenue"]),
    "gross_profit":                       ("income", ["Gross Profit"]),
    "operating_income":                   ("income", ["Operating Income", "EBIT"]),
    "operating_expense":                  ("income", ["Total Expenses", "Operating Expense", "Total Operating Expenses"]),
    "ebit":                               ("income", ["EBIT", "Operating Income"]),
    "ebitda":                             ("income", ["EBITDA", "Normalized EBITDA"]),
    "net_income":                         ("income", ["Net Income", "Net Income Common Stockholders"]),
    "earnings_per_share":                 ("income", ["Diluted EPS", "Basic EPS"]),
    "research_and_development":           ("income", ["Research And Development"]),
    "interest_expense":                   ("income", ["Interest Expense"]),
    # Balance sheet
    "total_assets":                       ("balance", ["Total Assets"]),
    "total_liabilities":                  ("balance", ["Total Liabilities Net Minority Interest", "Total Liabilities"]),
    "shareholders_equity":                ("balance", ["Stockholders Equity", "Total Equity Gross Minority Interest"]),
    "total_debt":                         ("balance", ["Total Debt", "Long Term Debt And Capital Lease Obligation"]),
    "cash_and_equivalents":               ("balance", ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"]),
    "current_assets":                     ("balance", ["Current Assets"]),
    "current_liabilities":                ("balance", ["Current Liabilities"]),
    "goodwill_and_intangible_assets":     ("balance", ["Goodwill And Other Intangible Assets", "Total Intangible Assets"]),
    "intangible_assets":                  ("balance", ["Other Intangible Assets"]),
    # Cash flow
    "depreciation_and_amortization":      ("cashflow", ["Depreciation And Amortization", "Depreciation Amortization Depletion"]),
    "free_cash_flow":                     ("cashflow", ["Free Cash Flow"]),
    "capital_expenditure":                ("cashflow", ["Capital Expenditure"]),
    "dividends_and_other_cash_distributions": ("cashflow", ["Cash Dividends Paid", "Dividends Paid", "Payment Of Dividends"]),
    "issuance_or_purchase_of_equity_shares":  ("cashflow", ["Repurchase Of Capital Stock", "Common Stock Issuance", "Purchase Of Business"]),
}

_COMPUTED_ITEMS = {
    "working_capital", "outstanding_shares", "gross_margin",
    "operating_margin", "debt_to_equity", "return_on_invested_capital",
}


def _get_stmts(t: yf.Ticker, period: str):
    """Return (income, balance, cashflow) DataFrames for the given period."""
    if period == "quarterly":
        return t.quarterly_income_stmt, t.quarterly_balance_sheet, t.quarterly_cashflow
    # annual and ttm both start from annual; TTM is computed separately for flow items
    return t.income_stmt, t.balance_sheet, t.cashflow


# ---------------------------------------------------------------------------
# Public API — same signatures as original api.py
# ---------------------------------------------------------------------------

def get_prices(ticker: str, start_date: str, end_date: str, api_key: str = None) -> list[Price]:
    cache_key = f"{ticker}_{start_date}_{end_date}"
    if cached := _cache.get_prices(cache_key):
        return [Price(**p) for p in cached]

    t = _get_ticker(ticker)
    try:
        hist = t.history(start=start_date, end=end_date, interval="1d", auto_adjust=True)
    except Exception as e:
        logger.warning("yfinance price fetch failed for %s: %s", ticker, e)
        return []

    if hist.empty:
        return []

    prices = []
    for ts, row in hist.iterrows():
        prices.append(Price(
            open=round(float(row["Open"]), 6),
            close=round(float(row["Close"]), 6),
            high=round(float(row["High"]), 6),
            low=round(float(row["Low"]), 6),
            volume=int(row["Volume"]),
            time=ts.strftime("%Y-%m-%dT%H:%M:%S"),
        ))

    _cache.set_prices(cache_key, [p.model_dump() for p in prices])
    return prices


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[FinancialMetrics]:
    cache_key = f"{ticker}_{period}_{end_date}_{limit}"
    if cached := _cache.get_financial_metrics(cache_key):
        return [FinancialMetrics(**m) for m in cached]

    t = _get_ticker(ticker)
    info = t.info or {}

    def _i(key, default=None):
        v = info.get(key, default)
        return None if v in (None, "N/A", "None", float("inf"), float("-inf")) else float(v) if v is not None else None

    # TTM / latest metrics from info
    market_cap = _i("marketCap")
    ev = _i("enterpriseValue")
    pe = _i("trailingPE") or _i("forwardPE")
    pb = _i("priceToBook")
    ps = _i("priceToSalesTrailing12Months")
    ev_ebitda = _i("enterpriseToEbitda")
    ev_rev = _i("enterpriseToRevenue")
    peg = _i("pegRatio")
    gross_m = _i("grossMargins")
    op_m = _i("operatingMargins")
    net_m = _i("profitMargins")
    roe = _i("returnOnEquity")
    roa = _i("returnOnAssets")
    de = _i("debtToEquity")
    if de is not None:
        de = de / 100  # yfinance gives D/E as percentage
    rev_growth = _i("revenueGrowth")
    earn_growth = _i("earningsGrowth")
    eps = _i("trailingEps")
    bvps = _i("bookValue")
    current_r = _i("currentRatio")
    quick_r = _i("quickRatio")
    payout = _i("payoutRatio")
    currency = info.get("currency", "USD")

    # FCF yield = FCF / market_cap
    fcf = _i("freeCashflow")
    fcf_yield = (fcf / market_cap) if fcf and market_cap else None

    # FCF per share
    shares = _i("sharesOutstanding") or _i("impliedSharesOutstanding")
    fcf_ps = (fcf / shares) if fcf and shares else None

    today = end_date or datetime.date.today().isoformat()

    m = FinancialMetrics(
        ticker=ticker,
        report_period=today,
        period=period,
        currency=currency,
        market_cap=market_cap,
        enterprise_value=ev,
        price_to_earnings_ratio=pe,
        price_to_book_ratio=pb,
        price_to_sales_ratio=ps,
        enterprise_value_to_ebitda_ratio=ev_ebitda,
        enterprise_value_to_revenue_ratio=ev_rev,
        free_cash_flow_yield=fcf_yield,
        peg_ratio=peg,
        gross_margin=gross_m,
        operating_margin=op_m,
        net_margin=net_m,
        return_on_equity=roe,
        return_on_assets=roa,
        return_on_invested_capital=None,  # not in info; TODO compute
        asset_turnover=None,
        inventory_turnover=None,
        receivables_turnover=None,
        days_sales_outstanding=None,
        operating_cycle=None,
        working_capital_turnover=None,
        current_ratio=current_r,
        quick_ratio=quick_r,
        cash_ratio=None,
        operating_cash_flow_ratio=None,
        debt_to_equity=de,
        debt_to_assets=None,
        interest_coverage=None,
        revenue_growth=rev_growth,
        earnings_growth=earn_growth,
        book_value_growth=None,
        earnings_per_share_growth=None,
        free_cash_flow_growth=None,
        operating_income_growth=None,
        ebitda_growth=None,
        payout_ratio=payout,
        earnings_per_share=eps,
        book_value_per_share=bvps,
        free_cash_flow_per_share=fcf_ps,
    )

    result = [m]
    _cache.set_financial_metrics(cache_key, [r.model_dump() for r in result])
    return result


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[LineItem]:
    t = _get_ticker(ticker)
    info = t.info or {}
    currency = info.get("currency", "USD")

    if period == "ttm":
        return _search_line_items_ttm(t, ticker, line_items, end_date, currency, limit)
    elif period == "annual":
        income, balance, cashflow = t.income_stmt, t.balance_sheet, t.cashflow
    else:  # quarterly
        income, balance, cashflow = t.quarterly_income_stmt, t.quarterly_balance_sheet, t.quarterly_cashflow

    # Determine available periods (columns of any non-empty df)
    for df in (income, balance, cashflow):
        if df is not None and not df.empty:
            cols = list(df.columns)
            break
    else:
        return []

    results = []
    for i, col in enumerate(cols[:limit]):
        report_period = col.strftime("%Y-%m-%d") if hasattr(col, "strftime") else str(col)[:10]
        fields = _extract_fields(line_items, income, balance, cashflow, col, info)
        item = LineItem(ticker=ticker, report_period=report_period, period=period, currency=currency, **fields)
        results.append(item)

    return results


def _search_line_items_ttm(
    t: yf.Ticker, ticker: str, line_items: list[str],
    end_date: str, currency: str, limit: int
) -> list[LineItem]:
    """Build TTM LineItem: sum last 4 quarters for flow items, latest balance for stock items."""
    q_income = t.quarterly_income_stmt
    q_balance = t.quarterly_balance_sheet
    q_cashflow = t.quarterly_cashflow
    annual_income = t.income_stmt
    annual_balance = t.balance_sheet
    annual_cashflow = t.cashflow
    info = t.info or {}

    today = end_date or datetime.date.today().isoformat()
    fields: dict = {}

    for item_name in line_items:
        if item_name in _COMPUTED_ITEMS:
            fields[item_name] = _compute_item(item_name, q_income, q_balance, q_cashflow, info, ttm=True)
            continue

        mapping = _ITEM_MAP.get(item_name)
        if not mapping:
            fields[item_name] = None
            continue

        stmt_type, candidates = mapping

        if stmt_type == "balance":
            # Point-in-time: use latest quarterly
            row = _safe_row(q_balance, candidates)
            if row is None:
                row = _safe_row(annual_balance, candidates)
            fields[item_name] = _val(row, 0)
        elif stmt_type in ("income", "cashflow"):
            # Flow: sum last 4 quarters
            q_df = q_income if stmt_type == "income" else q_cashflow
            a_df = annual_income if stmt_type == "income" else annual_cashflow
            ttm = _ttm_flow(q_df, candidates)
            if ttm is None:
                # Fallback to most recent annual
                row = _safe_row(a_df, candidates)
                ttm = _val(row, 0)
            fields[item_name] = ttm

    result = LineItem(ticker=ticker, report_period=today, period="ttm", currency=currency, **fields)
    return [result]


def _extract_fields(
    line_items: list[str],
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cashflow: pd.DataFrame,
    col,
    info: dict,
) -> dict:
    fields: dict = {}
    for item_name in line_items:
        if item_name in _COMPUTED_ITEMS:
            fields[item_name] = _compute_item(item_name, income, balance, cashflow, info, col=col)
            continue

        mapping = _ITEM_MAP.get(item_name)
        if not mapping:
            fields[item_name] = None
            continue

        stmt_type, candidates = mapping
        df = {"income": income, "balance": balance, "cashflow": cashflow}.get(stmt_type)
        if df is None or df.empty or col not in df.columns:
            fields[item_name] = None
            continue

        row = _safe_row(df, candidates)
        if row is None:
            fields[item_name] = None
            continue

        try:
            v = row[col]
            fields[item_name] = None if pd.isna(v) else float(v)
        except Exception:
            fields[item_name] = None

    return fields


def _compute_item(
    item_name: str,
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cashflow: pd.DataFrame,
    info: dict,
    col=None,
    ttm: bool = False,
) -> float | None:
    def get_val(df, candidates):
        if df is None or df.empty:
            return None
        row = _safe_row(df, candidates)
        if row is None:
            return None
        if ttm:
            return _ttm_flow(df, candidates) if df is not None else None
        if col is not None and col in row.index:
            v = row[col]
            return None if pd.isna(v) else float(v)
        return _val(row, 0)

    if item_name == "working_capital":
        ca = get_val(balance, ["Current Assets"])
        cl = get_val(balance, ["Current Liabilities"])
        return (ca - cl) if ca is not None and cl is not None else None

    if item_name == "outstanding_shares":
        v = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
        return float(v) if v else None

    if item_name == "gross_margin":
        gp = get_val(income, ["Gross Profit"])
        rev = get_val(income, ["Total Revenue"])
        return (gp / rev) if gp and rev else None

    if item_name == "operating_margin":
        oi = get_val(income, ["Operating Income", "EBIT"])
        rev = get_val(income, ["Total Revenue"])
        return (oi / rev) if oi and rev else None

    if item_name == "debt_to_equity":
        debt = get_val(balance, ["Total Debt"])
        eq = get_val(balance, ["Stockholders Equity"])
        return (debt / eq) if debt and eq else None

    if item_name == "return_on_invested_capital":
        ni = get_val(income, ["Net Income"])
        debt = get_val(balance, ["Total Debt"])
        eq = get_val(balance, ["Stockholders Equity"])
        invested = (debt or 0) + (eq or 0)
        return (ni / invested) if ni and invested else None

    return None


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[InsiderTrade]:
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"
    if cached := _cache.get_insider_trades(cache_key):
        return [InsiderTrade(**t) for t in cached]

    t = _get_ticker(ticker)
    try:
        df = t.insider_transactions
    except Exception:
        return []

    if df is None or df.empty:
        return []

    trades = []
    for _, row in df.iterrows():
        try:
            filing_date = str(row.get("Start Date", row.get("Date", "")))[:10]
            if not filing_date:
                continue
            if end_date and filing_date > end_date:
                continue
            if start_date and filing_date < start_date:
                continue

            shares = row.get("Shares", None)
            value = row.get("Value", None)

            trades.append(InsiderTrade(
                ticker=ticker,
                issuer=None,
                name=str(row.get("Insider", row.get("Name", ""))) or None,
                title=str(row.get("Position", row.get("Title", ""))) or None,
                is_board_director=None,
                transaction_date=filing_date,
                transaction_shares=float(shares) if shares is not None and not pd.isna(shares) else None,
                transaction_price_per_share=None,
                transaction_value=float(value) if value is not None and not pd.isna(value) else None,
                shares_owned_before_transaction=None,
                shares_owned_after_transaction=None,
                security_title=None,
                filing_date=filing_date,
            ))
        except Exception as e:
            logger.debug("Skipping insider trade row: %s", e)
            continue

    trades = trades[:limit]
    _cache.set_insider_trades(cache_key, [tr.model_dump() for tr in trades])
    return trades


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 50,
    api_key: str = None,
) -> list[CompanyNews]:
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"
    if cached := _cache.get_company_news(cache_key):
        return [CompanyNews(**n) for n in cached]

    t = _get_ticker(ticker)
    try:
        raw_news = t.news or []
    except Exception:
        raw_news = []

    news_items = []
    for item in raw_news[:limit]:
        try:
            content = item.get("content", {})
            pub_date = content.get("pubDate", "")
            if not pub_date:
                ts = item.get("providerPublishTime")
                pub_date = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%S") if ts else ""
            date_str = pub_date[:10] if pub_date else ""
            if end_date and date_str > end_date:
                continue
            if start_date and date_str < start_date:
                continue

            title = content.get("title", item.get("title", ""))
            source_info = content.get("provider", {})
            source = source_info.get("displayName", "") if isinstance(source_info, dict) else str(source_info)
            url = content.get("canonicalUrl", {}).get("url", "") if isinstance(content.get("canonicalUrl"), dict) else ""
            if not url:
                url = item.get("link", "")

            news_items.append(CompanyNews(
                ticker=ticker,
                title=title or "N/A",
                author=None,
                source=source or "Yahoo Finance",
                date=pub_date or datetime.date.today().isoformat(),
                url=url or "",
                sentiment=None,
            ))
        except Exception as e:
            logger.debug("Skipping news item: %s", e)
            continue

    _cache.set_company_news(cache_key, [n.model_dump() for n in news_items])
    return news_items


def get_market_cap(ticker: str, end_date: str, api_key: str = None) -> float | None:
    t = _get_ticker(ticker)
    info = t.info or {}
    mc = info.get("marketCap")
    return float(mc) if mc else None


def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    df = pd.DataFrame([p.model_dump() for p in prices])
    if df.empty:
        return df
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    for col in ["open", "close", "high", "low", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


def get_price_data(ticker: str, start_date: str, end_date: str, api_key: str = None) -> pd.DataFrame:
    return prices_to_df(get_prices(ticker, start_date, end_date))
