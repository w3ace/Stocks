import matplotlib.pyplot as plt

def plot_stock(symbol, daily_data, hourly_data):
    # Calculate cumulative sums for the differences using .loc to avoid SettingWithCopyWarning
    daily_data.loc[:, 'Cumulative_Open_Close_Diff'] = daily_data['Open_Close_Diff'].cumsum() + daily_data['Open'].iloc[0]
    daily_data.loc[:, 'Cumulative_Close_NextOpen_Diff'] = daily_data['Close_NextOpen_Diff'].cumsum() + daily_data['Close'].iloc[0]

    # Normalize the volume for scatter plot size scaling
    max_size = 100  # Define the maximum size for the scatter plot
    min_size = 10   # Define the minimum size for the scatter plot
    norm_volume = (hourly_data['Volume'] - hourly_data['Volume'].min()) / (hourly_data['Volume'].max() - hourly_data['Volume'].min())
    sizes = min_size + norm_volume * (max_size - min_size)

    # Create a figure with 2 subplots, the first one being larger
    fig, axs = plt.subplots(2, 1, figsize=(12, 12))
    grid_spec = fig.add_gridspec(2, 1, height_ratios=[3, 1])

    # Plot daily open prices as a line on the first subplot
    ax1 = fig.add_subplot(grid_spec[0])
    ax1.plot(daily_data['Date'], daily_data['Open'], label='Daily Open Price', color='blue', linestyle='-', linewidth=2)

    # Plot hourly open prices as a scatter plot on the first subplot
    ax1.scatter(hourly_data['Datetime'], hourly_data['Open'], label='Hourly Open Price', color='red', s=sizes, alpha=0.6)

    # Plot cumulative Open_Close_Diff as a line on the first subplot
    ax1.plot(daily_data['Date'], daily_data['Cumulative_Open_Close_Diff'], label='Cumulative Open-Close Difference', color='green', linestyle='--', linewidth=2)

    # Plot cumulative Close_NextOpen_Diff as a line on the first subplot
    ax1.plot(daily_data['Date'], daily_data['Cumulative_Close_NextOpen_Diff'], label='Cumulative Close-Next Open Difference', color='orange', linestyle='-.', linewidth=2)

    # Set title and labels for the first subplot
    ax1.set_title(f'Daily and Hourly Open Prices for {symbol} with Cumulative Differences')
    ax1.set_xlabel('Date/Datetime')
    ax1.set_ylabel('Price / Cumulative Difference')

    # Rotate x-axis labels for better readability
    ax1.tick_params(axis='x', rotation=45)

    # Add a legend to the first subplot
    ax1.legend()

    # Add a table for first hour classifications
    first_hour_counts = daily_data['First_Hour_Classification'].value_counts().to_frame()
    table_data = first_hour_counts.T
    table_data.columns.name = None

    table = ax1.table(cellText=table_data.values, colLabels=table_data.columns, cellLoc='center', loc='top', bbox=[0.2, 1.05, 0.6, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Plot MACD and Signal Line on the second subplot
    ax2 = fig.add_subplot(grid_spec[1])
    ax2.plot(daily_data['Date'], daily_data['MACD'], label='MACD', color='purple', linestyle='-', linewidth=2)
    ax2.plot(daily_data['Date'], daily_data['Signal_Line'], label='Signal Line', color='black', linestyle='--', linewidth=2)
    
    # Add a bar plot for MACD
    ax2.bar(daily_data['Date'], daily_data['MACD'], label='MACD Bar', color='gray', alpha=0.5)

    # Set title and labels for the second subplot
    ax2.set_title(f'MACD for {symbol}')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('MACD Value')

    # Rotate x-axis labels for better readability
    ax2.tick_params(axis='x', rotation=45)

    # Add a legend to the second subplot
    ax2.legend()

    # Ensure layout is tight
    plt.tight_layout()

    return fig
