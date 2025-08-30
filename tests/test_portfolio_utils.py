import portfolio_utils


def test_expand_ticker_args_from_portfolio(tmp_path, monkeypatch):
    portfolios_dir = tmp_path / "portfolios"
    portfolios_dir.mkdir()
    (portfolios_dir / "sample").write_text("AAPL MSFT")
    monkeypatch.setattr(portfolio_utils, "PORTFOLIOS_DIR", portfolios_dir)

    expanded = portfolio_utils.expand_ticker_args(["+sample"])
    assert expanded == ["AAPL", "MSFT"]
