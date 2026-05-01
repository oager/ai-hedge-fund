"""Non-interactive wrapper for ai-hedge-fund. Run directly with python."""
import sys
import os
import datetime
sys.path.insert(0, "/home/oager/ai-hedge-fund")
os.chdir("/home/oager/ai-hedge-fund")

from dotenv import load_dotenv
load_dotenv("/home/oager/ai-hedge-fund/.env")

from src.main import run_hedge_fund, print_trading_output

tickers  = sys.argv[1].split(",") if len(sys.argv) > 1 else ["AAPL", "NVDA"]
model    = sys.argv[2] if len(sys.argv) > 2 else "llama-3.3-70b-versatile"
provider = sys.argv[3] if len(sys.argv) > 3 else "Groq"

end_date   = datetime.date.today().isoformat()
start_date = (datetime.date.today() - datetime.timedelta(days=365)).isoformat()

portfolio = {
    "cash": 100_000.0,
    "margin_requirement": 0.0,
    "margin_used": 0.0,
    "positions": {t: {"long": 0, "short": 0, "long_cost_basis": 0.0,
                      "short_cost_basis": 0.0, "short_margin_used": 0.0}
                  for t in tickers},
    "realized_gains": {t: {"long": 0.0, "short": 0.0} for t in tickers},
}

print(f"\nRunning analysis on {tickers} with {provider}/{model}")
print(f"Date range: {start_date} → {end_date}\n")

result = run_hedge_fund(
    tickers=tickers,
    start_date=start_date,
    end_date=end_date,
    portfolio=portfolio,
    show_reasoning=True,
    selected_analysts=None,   # None = all analysts
    model_name=model,
    model_provider=provider,
)

print_trading_output(result)
