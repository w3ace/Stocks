import argparse
import pandas as pd
import matplotlib.pyplot as plt
from fetch_stock import fetch_stock
from analyze_stock import analyze_stock
from plot_stock import plot_stock
from matplotlib.backends.backend_pdf import PdfPages

def main():
    parser = argparse.ArgumentParser(description='Analyze stock data for opening and closing gaps.')
    parser.add_argument('symbols', type=str, nargs='+', help='Stock symbols separated by spaces')
    parser.add_argument('--start', type=str, help='Start date in mm-dd-yyyy format')
    parser.add_argument('--end', type=str, help='End date in mm-dd-yyyy format')
    parser.add_argument('--period', type=str, help='Period (e.g., 1mo, 6mo, 1y)')
    args = parser.parse_args()

    start_date = None
    end_date = None

    if args.start and args.end:
        start_date = pd.to_datetime(args.start, format='%m-%d-%Y')
        end_date = pd.to_datetime(args.end, format='%m-%d-%Y')
    elif args.period:
        if not args.period:
            print("Please provide either a date range (start and end) or a period.")
            return

    output_path = 'stock_analysis.pdf'
    
    with PdfPages(output_path) as pdf:
        for symbol in args.symbols:
            if start_date and end_date:
                hourly_data, daily_data = fetch_stock(symbol, start_date=start_date, end_date=end_date)
            else:
                hourly_data, daily_data = fetch_stock(symbol, period=args.period)

            print(f"Stock data info for {symbol}:")
            print(hourly_data)
            print(daily_data)

            hourly_data, daily_data = analyze_stock(hourly_data, daily_data)

            fig = plot_stock(symbol, daily_data, hourly_data)
            pdf.savefig(fig)
            plt.close(fig)

    print(f"PDF saved to {output_path}")

if __name__ == '__main__':
    main()
