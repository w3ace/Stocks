# plot_stock_data.py
import plotly.graph_objs as go

def plot_stock(stock_data, symbol):
    # Calculate the min and max values for y-axis with 10% buffer
    min_value = min(stock_data['Open to Close Cumulative'].min(), stock_data['Close to Open Cumulative'].min(), stock_data['Open'].min())
    max_value = max(stock_data['Open to Close Cumulative'].max(), stock_data['Close to Open Cumulative'].max(), stock_data['Open'].max())
    buffer = (max_value - min_value) * 0.10
    y_min = min_value - buffer
    y_max = max_value + buffer


   # Calculate the final values and percentage changes
    final_open_to_close = stock_data['Open to Close Cumulative'].iloc[-1]
    final_close_to_open = stock_data['Close to Open Cumulative'].iloc[-1]
    final_open = stock_data['Open'].iloc[-1]

    final_open_to_close_pct = stock_data['Open to Close % Change'].iloc[-1]
    final_close_to_open_pct = stock_data['Close to Open % Change'].iloc[-1]

    # Plotly visualization
    fig = go.Figure()

    # Add trace for Open to Close Cumulative
    fig.add_trace(go.Scatter(
        x=stock_data.index, 
        y=stock_data['Open to Close Cumulative'],
        mode='lines',
        name=f'Buy at Open, Sell at Close Cumulative:'
    ))

    # Add trace for Close to Open Cumulative
    fig.add_trace(go.Scatter(
        x=stock_data.index, 
        y=stock_data['Close to Open Cumulative'],
        mode='lines',
        name=f'Buy at Close, Sell at Open Cumulative'
    ))

    # Add trace for Daily Opening Price
    fig.add_trace(go.Scatter(
        x=stock_data.index, 
        y=stock_data['Open'],
        mode='lines',
        name=f'Daily Open:              {final_open:.2f}'
    ))

    # Update layout
    fig.update_layout(
        title=f'{symbol} Cumulative Gap Totals',
        xaxis_title='Date',
        yaxis_title='Cumulative Gap Amount / Daily Open',
        yaxis=dict(range=[y_min, y_max]),
        legend=dict(x=0, y=1, traceorder='normal')
    )

    # Show the plot
    fig.show()
