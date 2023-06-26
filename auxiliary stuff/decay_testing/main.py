import datetime

DECAY_FACTOR = 0.9
MAXIMUM_DECAY = 0.1

import matplotlib.pyplot as plt


def plot_decay_values(decay_values):
    # Calculate the x-axis values (lifespan in minutes)
    x = [i * 30 for i in range(len(decay_values))]

    # Plot the decay values
    plt.plot(x, decay_values)

    # Set x-axis tick labels in increments of 3 hours
    tick_positions = list(range(0, len(x) * 30, 3 * 60))
    tick_labels = [str(t // 60) + 'h' for t in tick_positions]
    plt.xticks(tick_positions, tick_labels)

    # Set labels and title
    plt.xlabel('Lifespan (minutes)')
    plt.ylabel('Decay Value')
    plt.title('Decay Progression at decay factor ' + str(DECAY_FACTOR))

    # Get the decay value at 24 hours
    decay_24h = decay_values[24 * 60 // 30]

    # Display the decay value at 24 hours in the top right corner
    plt.text(x[-1], decay_24h, f'Decay at 24h: {decay_24h:.2f}', ha='right', va='top')

    # Set the position of the text box in the top right corner
    plt.tight_layout(rect=[0.85, 0.9, 1, 1])

    # Display the plot
    plt.show()



def handle_time_decay(date_values):
    decay_values = []
    nr_minutes_per_day = 24 * 60

    for i, date_value in enumerate(date_values):
        # Calculate total lifespan in minutes of the current submission
        lifespan = (date_value - date_values[0]).total_seconds() / 60

        # Calculate the decay value
        decay_value = (1 - DECAY_FACTOR) ** (lifespan / nr_minutes_per_day)

        # Cap maximum decay at MAXIMUM_DECAY
        decay_value = max(MAXIMUM_DECAY, decay_value)

        decay_values.append(decay_value)

    return decay_values



if __name__ == '__main__':
    # Create the list of dates with 30-minute increments for 48 hours
    start_date = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    date_values = [start_date + datetime.timedelta(minutes=30 * i) for i in range(0, 48 * 60 // 30 + 1)]


    for i in range(len(date_values)):
        print(date_values[i])

    decay = handle_time_decay(date_values)

    print(decay[int(len(decay)/2)])

    plot_decay_values(decay)


