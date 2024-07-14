import matplotlib.pyplot as plt

def plot_stock(symbol, daily_data, hourly_data):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot daily open prices as a line
    ax.plot(daily_data['Date'], daily_data['Open'], label='Daily Open Price', color='blue', linestyle='-', linewidth=2)

    # Plot hourly open prices as a scatter plot
    ax.scatter(hourly_data['Datetime'], hourly_data['Open'], label='Hourly Open Price', color='red', marker='o')

    # Set title and labels
    ax.set_title(f'Daily and Hourly Open Prices for {symbol}')
    ax.set_xlabel('Date/Datetime')
    ax.set_ylabel('Open Price')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Add a legend
    ax.legend()

    # Ensure layout is tight
    plt.tight_layout()
    
    return fig
