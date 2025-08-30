import pandas as pd

from stocks.data import fetch


def test_fetch_ticker_recovers_from_invalid_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "CACHE_DIR", tmp_path)
    cache_file = tmp_path / "FAKE_-_-_1mo.csv"
    cache_file.write_text("Open,High,Low,Close\n1,2,3,4\n")

    df_stub = pd.DataFrame({"Open": [1.0], "High": [2.0], "Low": [3.0], "Close": [4.0]}, index=pd.to_datetime(["2020-01-01"]))
    df_stub.index.name = "Date"

    calls = []

    def fake_download(*args, **kwargs):
        calls.append(1)
        return df_stub

    monkeypatch.setattr(fetch.yf, "download", fake_download)

    result1 = fetch.fetch_ticker("FAKE", period="1mo")
    assert result1.equals(df_stub)
    assert len(calls) == 1

    result2 = fetch.fetch_ticker("FAKE", period="1mo")
    assert result2.equals(df_stub)
    assert len(calls) == 1
