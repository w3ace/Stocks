import matplotlib.pyplot as plt

def plot_stock(symbol, daily_data, hourly_data):

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot daily open prices as a line
    ax.plot(daily_data['Date'], daily_data['Open'], label='Daily Open Price', color='blue', linestyle='-', linewidth=2)

    # Plot hourly open prices as a scatter plot
    ax.scatter(hourly_data['Datetime'], hourly_data['Open'], label='Hourly Open Price', color='red', marker='o')

    # Plot cumulative Open_Close_Diff as a line
    ax.plot(daily_data['Date'], daily_data['Cumulative_Open_Close_Diff'], label='Cumulative Open-Close Difference', color='green', linestyle='--', linewidth=2)

    # Plot cumulative Close_NextOpen_Diff as a line
    ax.plot(daily_data['Date'], daily_data['Cumulative_Close_NextOpen_Diff'], label='Cumulative Close-Next Open Difference', color='orange', linestyle='-.', linewidth=2)

    # Set title and labels
    ax.set_title(f'Daily and Hourly Open Prices for {symbol} with Cumulative Differences')
    ax.set_xlabel('Date/Datetime')
    ax.set_ylabel('Price / Cumulative Difference')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Add a legend
    ax.legend()

    # Ensure layout is tight
    plt.tight_layout()
    
    return fig
