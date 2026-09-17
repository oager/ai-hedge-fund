from langchain_core.messages import HumanMessage
from src.graph.state import AgentState, show_agent_reasoning
from src.utils.progress import progress
import pandas as pd
import numpy as np
import json
from src.utils.api_key import get_api_key_from_state
from src.tools.api import get_insider_trades, get_company_news


##### Sentiment Agent #####
def sentiment_analyst_agent(state: AgentState, agent_id: str = "sentiment_analyst_agent"):
    """Analyzes market sentiment and generates trading signals for multiple tickers."""
    data = state.get("data", {})
    end_date = data.get("end_date")
    tickers = data.get("tickers")
    api_key = get_api_key_from_state(state, "FINANCIAL_DATASETS_API_KEY")
    # Initialize sentiment analysis for each ticker
    sentiment_analysis = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Fetching insider trades")

        # Get the insider trades
        insider_trades = get_insider_trades(
            ticker=ticker,
            end_date=end_date,
            limit=1000,
            api_key=api_key,
        )

        progress.update_status(agent_id, ticker, "Analyzing trading patterns")

        # Get the signals from the insider trades.
        # transaction_shares is nullable -- that is why .dropna() is here -- so the
        # post-filter list is NOT a fetch count. Same trap the news leg had: reporting
        # "no insider trades returned" when 40 Form 4 rows came back with null share
        # counts is a false statement about the fetch.
        trades_fetched = len(insider_trades)
        transaction_shares = pd.Series([t.transaction_shares for t in insider_trades]).dropna()
        insider_signals = np.where(transaction_shares < 0, "bearish", "bullish").tolist()

        progress.update_status(agent_id, ticker, "Fetching company news")

        # Get the company news
        company_news = get_company_news(ticker, end_date, limit=100, api_key=api_key)

        # Get the sentiment from the company news.
        # The provider does NOT populate CompanyNews.sentiment -- every article comes
        # back with sentiment=None, so .dropna() empties the series and this leg
        # contributes nothing. Keep the fetched count separately: reporting
        # total_articles=0 when 10 articles were fetched reads as "no news exists"
        # rather than "the sentiment field was empty", and that inflated this agent
        # to BULLISH 100% on NKE (insider-only) while the separate news_sentiment
        # agent scored BEARISH 80.5% on the same 5-10 articles by classifying text.
        articles_fetched = len(company_news)
        sentiment = pd.Series([n.sentiment for n in company_news]).dropna()
        articles_usable = len(sentiment)
        news_signals = np.where(sentiment == "negative", "bearish", 
                              np.where(sentiment == "positive", "bullish", "neutral")).tolist()
        # Partial loss matters too: 3 usable of 10 still applies the full 0.7 weight
        # to a 70%-discarded sample, and "ok" would hide that.
        news_leg_degraded = articles_fetched > 0 and articles_usable < articles_fetched
        
        progress.update_status(agent_id, ticker, "Combining signals")
        # Combine signals from both sources with weights
        insider_weight = 0.3
        news_weight = 0.7
        
        # Calculate weighted signal counts
        bullish_signals = (
            insider_signals.count("bullish") * insider_weight +
            news_signals.count("bullish") * news_weight
        )
        bearish_signals = (
            insider_signals.count("bearish") * insider_weight +
            news_signals.count("bearish") * news_weight
        )

        if bullish_signals > bearish_signals:
            overall_signal = "bullish"
        elif bearish_signals > bullish_signals:
            overall_signal = "bearish"
        else:
            overall_signal = "neutral"

        # Calculate confidence level based on the weighted proportion
        total_weighted_signals = len(insider_signals) * insider_weight + len(news_signals) * news_weight
        confidence = 0  # Default confidence when there are no signals
        if total_weighted_signals > 0:
            confidence = round((max(bullish_signals, bearish_signals) / total_weighted_signals) * 100, 2)
        
        # Create structured reasoning similar to technical analysis
        reasoning = {
            "insider_trading": {
                "signal": "bullish" if insider_signals.count("bullish") > insider_signals.count("bearish") else 
                         "bearish" if insider_signals.count("bearish") > insider_signals.count("bullish") else "neutral",
                "confidence": round((max(insider_signals.count("bullish"), insider_signals.count("bearish")) / max(len(insider_signals), 1)) * 100),
                "metrics": {
                    "trades_fetched": trades_fetched,
                    "trades_with_usable_share_count": len(insider_signals),
                    "total_trades": len(insider_signals),
                    "bullish_trades": insider_signals.count("bullish"),
                    "bearish_trades": insider_signals.count("bearish"),
                    "weight": insider_weight,
                    "weighted_bullish": round(insider_signals.count("bullish") * insider_weight, 1),
                    "weighted_bearish": round(insider_signals.count("bearish") * insider_weight, 1),
                }
            },
            "news_sentiment": {
                "signal": "bullish" if news_signals.count("bullish") > news_signals.count("bearish") else 
                         "bearish" if news_signals.count("bearish") > news_signals.count("bullish") else "neutral",
                "confidence": round((max(news_signals.count("bullish"), news_signals.count("bearish")) / max(len(news_signals), 1)) * 100),
                "metrics": {
                    "articles_fetched": articles_fetched,
                    "articles_with_usable_sentiment": articles_usable,
                    "total_articles": len(news_signals),
                    "bullish_articles": news_signals.count("bullish"),
                    "bearish_articles": news_signals.count("bearish"),
                    "neutral_articles": news_signals.count("neutral"),
                    "weight": news_weight,
                    "weighted_bullish": round(news_signals.count("bullish") * news_weight, 1),
                    "weighted_bearish": round(news_signals.count("bearish") * news_weight, 1),
                }
            },
            "data_quality": {
                # Four states per leg, and "ok" is only one of them. The zero-fetch
                # arm has to come FIRST: both DEGRADED arms require a non-zero fetch,
                # so without it a thinly-covered ticker returns no news at all, the
                # 0.7-weight leg contributes nothing, and the agent still stamps the
                # result "ok" -- the exact NKE failure, one branch over.
                "news_leg": (
                    "no company news returned" if not articles_fetched else
                    ("DEGRADED (total): %d articles fetched but none carried a usable "
                     "sentiment field, so this signal is insider-only and its confidence "
                     "reflects one of two intended legs" % articles_fetched)
                    if articles_usable == 0 else
                    ("DEGRADED (partial): %d of %d fetched articles carried a usable "
                     "sentiment field; the full news weight is applied to that subset"
                     % (articles_usable, articles_fetched))
                    if news_leg_degraded else "ok"),
                "insider_leg": (
                    "no insider trades returned" if not trades_fetched else
                    ("DEGRADED (total): %d trades fetched, none with a usable share count"
                     % trades_fetched)
                    if not insider_signals else
                    ("DEGRADED (partial): %d of %d fetched trades carried a usable share "
                     "count; the full insider weight is applied to that subset"
                     % (len(insider_signals), trades_fetched))
                    if len(insider_signals) < trades_fetched else "ok"),
            },
            "combined_analysis": {
                "total_weighted_bullish": round(bullish_signals, 1),
                "total_weighted_bearish": round(bearish_signals, 1),
                "signal_determination": f"{'Bullish' if bullish_signals > bearish_signals else 'Bearish' if bearish_signals > bullish_signals else 'Neutral'} based on weighted signal comparison"
            }
        }

        sentiment_analysis[ticker] = {
            "signal": overall_signal,
            "confidence": confidence,
            "reasoning": reasoning,
        }

        progress.update_status(agent_id, ticker, "Done", analysis=json.dumps(reasoning, indent=4))

    # Create the sentiment message
    message = HumanMessage(
        content=json.dumps(sentiment_analysis),
        name=agent_id,
    )

    # Print the reasoning if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(sentiment_analysis, "Sentiment Analysis Agent")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"][agent_id] = sentiment_analysis

    progress.update_status(agent_id, None, "Done")

    return {
        "messages": [message],
        "data": data,
    }
