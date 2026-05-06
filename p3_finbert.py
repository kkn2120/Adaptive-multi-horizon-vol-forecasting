"""
P3: Financial Text Sentiment — FinBERT.

Uses ProsusAI/finbert (a BERT model fine-tuned on financial text) to extract
sentiment from NYT financial headlines and FOMC documents. No API keys needed.
Fully deterministic and reproducible.

Data sources:
  - NYT Archive API headlines (cached from prior downloads)
  - FOMC statements and minutes (cached from prior downloads)
  - RSS feeds for recent headlines (optional)

Output: 10 daily features (sentiment, uncertainty, rolling averages)
"""

import os, json, time, requests, warnings
import numpy as np, pandas as pd
import torch
from config import *

warnings.filterwarnings("ignore")

NYT_KEY = os.environ.get("NYT_API_KEY", "")


def download_nyt_headlines(start_year=2000, end_year=2025):
    """Load or download NYT financial headlines."""
    cache_path = os.path.join(RAW_DIR, "nyt_headlines.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            data = json.load(f)
        print(f"  NYT headlines (cached): {len(data)} dates", flush=True)
        return data

    if not NYT_KEY:
        # Try alternate cache location
        alt = os.path.join(RAW_DIR, "news_headlines.json")
        if os.path.exists(alt):
            with open(alt) as f:
                data = json.load(f)
            print(f"  NYT headlines (alt cache): {len(data)} dates", flush=True)
            return data
        print("  NYT API key not set and no cache found — skipping", flush=True)
        return {}

    print(f"  Downloading NYT headlines {start_year}-{end_year}...", flush=True)
    headlines = {}
    business_sections = {"business", "Business", "Business Day", "DealBook", "Financial",
                         "Economy", "Markets", "Your Money", "Economix", "Wall Street"}

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            url = f"https://api.nytimes.com/svc/archive/v1/{year}/{month}.json?api-key={NYT_KEY}"
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code == 429:
                    time.sleep(60)
                    resp = requests.get(url, timeout=30)
                if resp.status_code != 200:
                    time.sleep(12); continue
                articles = resp.json().get("response", {}).get("docs", [])
                for article in articles:
                    section = article.get("section_name", "") or ""
                    news_desk = article.get("news_desk", "") or ""
                    is_fin = (section in business_sections or news_desk in business_sections or
                              "business" in section.lower() or "financial" in section.lower() or
                              "economy" in news_desk.lower() or "market" in news_desk.lower())
                    if not is_fin: continue
                    headline = article.get("headline", {}).get("main", "")
                    pub_date = article.get("pub_date", "")
                    if headline and pub_date:
                        ds = pub_date[:10]
                        if ds not in headlines: headlines[ds] = []
                        headlines[ds].append(headline)
            except Exception as e:
                print(f"    {year}-{month:02d}: error ({e})", flush=True)
            time.sleep(12)
        print(f"    {year}: {sum(1 for d in headlines if d.startswith(str(year)))} dates", flush=True)

    with open(cache_path, 'w') as f:
        json.dump(headlines, f)
    print(f"  NYT headlines: {len(headlines)} total dates", flush=True)
    return headlines


def download_fomc_data():
    """Load or download FOMC documents."""
    cache_path = os.path.join(RAW_DIR, "fomc_data.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            data = json.load(f)
        print(f"  FOMC data (cached): {len(data)} documents", flush=True)
        return data

    print("  Downloading FOMC data...", flush=True)
    fomc_data = {}
    try:
        from FedTools import MonetaryPolicyCommittee, FederalReserveMins
        statements = MonetaryPolicyCommittee().find_statements()
        for date_idx, row in statements.iterrows():
            ds = pd.Timestamp(date_idx).strftime("%Y-%m-%d")
            text = str(row.iloc[0]) if len(row) > 0 else ""
            if text and len(text) > 100:
                fomc_data[ds] = {"type": "statement", "text": text[:3000], "date": ds}
        minutes = FederalReserveMins().find_minutes()
        for date_idx, row in minutes.iterrows():
            ds = pd.Timestamp(date_idx).strftime("%Y-%m-%d")
            text = str(row.iloc[0]) if len(row) > 0 else ""
            if text and len(text) > 100:
                fomc_data[ds] = {"type": "minutes", "text": text[:3000], "date": ds}
    except Exception as e:
        print(f"    FOMC download failed: {e}", flush=True)

    with open(cache_path, 'w') as f:
        json.dump(fomc_data, f)
    print(f"  FOMC data: {len(fomc_data)} documents", flush=True)
    return fomc_data


def load_finbert():
    """Load FinBERT model and tokenizer."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_name = "ProsusAI/finbert"
    print(f"  Loading FinBERT ({model_name})...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  FinBERT loaded on {device}", flush=True)

    return model, tokenizer, device


def finbert_analyze_batch(texts, model, tokenizer, device, max_len=128):
    """
    Run FinBERT on a batch of texts.

    Returns dict with:
      - sentiment: average sentiment score (-1 negative to +1 positive)
      - uncertainty: proportion of neutral predictions (higher = more uncertain)
      - neg_ratio: proportion of negative predictions
      - pos_ratio: proportion of positive predictions
    """
    if not texts:
        return {"sentiment": 0.0, "uncertainty": 0.5, "neg_ratio": 0.0, "pos_ratio": 0.0}

    # Truncate to max 10 texts per day to keep inference fast
    texts = texts[:10]

    try:
        inputs = tokenizer(texts, return_tensors="pt", padding=True,
                          truncation=True, max_length=max_len)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            # FinBERT outputs: [positive, negative, neutral]
            pos_probs = probs[:, 0]
            neg_probs = probs[:, 1]
            neu_probs = probs[:, 2]

        
        sentiments = pos_probs - neg_probs
        avg_sentiment = float(np.mean(sentiments))

        
        avg_uncertainty = float(np.mean(neu_probs))

     
        labels = np.argmax(probs, axis=1)  # 0=pos, 1=neg, 2=neutral
        n = len(labels)
        pos_ratio = float(np.sum(labels == 0) / n)
        neg_ratio = float(np.sum(labels == 1) / n)

        return {
            "sentiment": avg_sentiment,
            "uncertainty": avg_uncertainty,
            "neg_ratio": neg_ratio,
            "pos_ratio": pos_ratio,
        }

    except Exception as e:
        print(f"    FinBERT error: {e}", flush=True)
        return {"sentiment": 0.0, "uncertainty": 0.5, "neg_ratio": 0.0, "pos_ratio": 0.0}


def train_finbert(feat, news_data):
    """P3 entry point: build text sentiment features using FinBERT."""
    print("\n" + "=" * 60, flush=True)
    print("P3: FINANCIAL TEXT ANALYSIS (FinBERT)", flush=True)
    print("=" * 60, flush=True)

    # Check for cached results
    cache_path = os.path.join(PROC_DIR, "p3_finbert.csv")
    finbert_cache_path = os.path.join(PROC_DIR, "p3_finbert_cache.json")

    
    nyt_headlines = download_nyt_headlines()
    fomc_data = download_fomc_data()

   
    all_headlines = {}
    if news_data:
        all_headlines.update(news_data)
    all_headlines.update(nyt_headlines)

    trading_days = feat.index
    real_coverage = sum(1 for d in trading_days if d.strftime("%Y-%m-%d") in all_headlines)
    print(f"\n  Headline coverage: {real_coverage}/{len(trading_days)} "
          f"({100*real_coverage/len(trading_days):.0f}%)", flush=True)

    
    if os.path.exists(finbert_cache_path):
        with open(finbert_cache_path) as f:
            cache = json.load(f)
        print(f"  FinBERT cache: {len(cache)} dates", flush=True)
    else:
        cache = {}

    # Check how many dates need processing
    uncached = [d for d in trading_days if d.strftime("%Y-%m-%d") not in cache]
    print(f"  Dates to process: {len(uncached)}", flush=True)

    # Load FinBERT if we have uncached dates
    model, tokenizer, device = None, None, None
    if len(uncached) > 0:
        model, tokenizer, device = load_finbert()

    # Process each trading day
    total = len(trading_days)
    processed = 0

    for i, d in enumerate(trading_days):
        ds = d.strftime("%Y-%m-%d")

        # Use cache if available
        if ds in cache:
            continue

        
        day_headlines = all_headlines.get(ds, [])

        # Check for FOMC document
        fomc_text = None
        for fd, fdata in fomc_data.items():
            if abs((d - pd.Timestamp(fd)).days) <= 1:
                fomc_text = fdata["text"]
                break

    
        texts = []
        if fomc_text:
            # Split FOMC into sentence-level chunks for FinBERT
            sentences = [s.strip() for s in fomc_text.split('.') if len(s.strip()) > 20]
            texts.extend(sentences[:5])  # top 5 sentences
        texts.extend(day_headlines[:10])

      
        if texts and model is not None:
            result = finbert_analyze_batch(texts, model, tokenizer, device)
        else:
            # No text available — neutral fallback
            result = {"sentiment": 0.0, "uncertainty": 0.5, "neg_ratio": 0.0, "pos_ratio": 0.0}

        cache[ds] = result
        processed += 1

        if processed % 500 == 0:
            print(f"  Processed {processed} dates ({i+1}/{total})...", flush=True)
            # Save cache periodically
            with open(finbert_cache_path, 'w') as f:
                json.dump(cache, f)

    # Final cache save
    if processed > 0:
        with open(finbert_cache_path, 'w') as f:
            json.dump(cache, f)
        print(f"  FinBERT processing complete: {processed} new dates", flush=True)

    # Free GPU memory. Fix running into local desktop space issues
    if model is not None:
        del model, tokenizer
        torch.cuda.empty_cache()
        import gc; gc.collect()

    
    results = []
    for d in trading_days:
        ds = d.strftime("%Y-%m-%d")
        if ds in cache:
            results.append(cache[ds])
        else:
            results.append({"sentiment": 0.0, "uncertainty": 0.5,
                          "neg_ratio": 0.0, "pos_ratio": 0.0})

    text_df = pd.DataFrame(results, index=trading_days)

    
    text_df["finbert_sentiment"] = text_df["sentiment"]
    text_df["finbert_uncertainty"] = text_df["uncertainty"]
    text_df["finbert_neg_ratio"] = text_df["neg_ratio"]
    text_df["finbert_pos_ratio"] = text_df["pos_ratio"]

   
    for col in ["finbert_sentiment", "finbert_uncertainty"]:
        text_df[f"{col}_5d"] = text_df[col].rolling(5, min_periods=1).mean()
        text_df[f"{col}_22d"] = text_df[col].rolling(22, min_periods=1).mean()

    # Sentiment momentum (change over 5 days)
    text_df["finbert_sentiment_mom5d"] = text_df["finbert_sentiment"].diff(5)

    # Sentiment dispersion (std over 22 days)
    text_df["finbert_sentiment_std22d"] = text_df["finbert_sentiment"].rolling(22, min_periods=5).std()

    
    keep_cols = [
        "finbert_sentiment", "finbert_uncertainty",
        "finbert_neg_ratio", "finbert_pos_ratio",
        "finbert_sentiment_5d", "finbert_uncertainty_5d",
        "finbert_sentiment_22d", "finbert_uncertainty_22d",
        "finbert_sentiment_mom5d", "finbert_sentiment_std22d",
    ]

    output = text_df[keep_cols].copy()
    output = output.fillna(0)
    output.to_csv(cache_path)
    print(f"\n  P3 features: {output.shape}", flush=True)
    print(f"  Columns: {list(output.columns)}", flush=True)

    # Summary stats
    print(f"\n  Sentiment stats:", flush=True)
    print(f"    Mean: {output['finbert_sentiment'].mean():.4f}", flush=True)
    print(f"    Std:  {output['finbert_sentiment'].std():.4f}", flush=True)
    print(f"    Min:  {output['finbert_sentiment'].min():.4f}", flush=True)
    print(f"    Max:  {output['finbert_sentiment'].max():.4f}", flush=True)

    return output
