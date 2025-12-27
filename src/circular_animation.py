"""
circular_animation.py:
This Python script generates a side-by-side animation comparing two datasets
representing vehicle movement on circular lanes. It visualizes cars' positions
at each timestep in two different models ("Human-Based" vs. "Smart-City") using
an animated scatter plot. The animation is saved to an utput GIF file and also
displayed interactively.

INPUT:
------
1. Two CSV files containing vehicle data with the following columns:
   - car_id: Unique identifier for each vehicle
   - timestep: Time at which the measurements are recorded
   - alpha: Arc length around the circular track (range [0, CIRCUMFERENCE))
   - lane: Lane index (0 for the innermost lane, 1 for the next, etc.)
   - speed: Speed of the vehicle at the given timestep
2. An output file name (for saving the resulting GIF).

FUNCTIONALITY:
--------------
1. Reads the two input CSV files as Pandas DataFrames.
2. Calculates positions in Cartesian coordinates (x, y) from (alpha, lane) on a
   circular track.
3. Creates a figure with two subplots, each showing animated car positions for
   a given dataset.
4. Uses Matplotlib's FuncAnimation to animate the positions over time.
5. Displays speed as a color scale, with a shared colorbar for both subplots.
6. Saves the final animation to a GIF file.
7. Displays the animation in a pop-up window when execution finishes.

OUTPUT:
-------
- A GIF file (specified as the third argument) showing the evolving positions
  of vehicles in both datasets over time.
- An interactive animation window appears on screen, allowing you to watch the
  animation in real time.

EXECUTION:
----------
If you want to run the file from the src directory directly:
- python3 circular_animation.py <data_file1.csv> <data_file2.csv> <output.gif>
Or if you want to run it from the root directory as a module:
- python3 -m src.circular_animation <data_file1.csv> <data_file2.csv> <output.gif>
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl

########################################
# GLOBAL PARAMETERS
########################################
CIRCUMFERENCE = 2000.0  # meters
LANE_WIDTH = 10         # meters

# Calculate the base radius for the innermost lane
BASE_RADIUS = CIRCUMFERENCE / (2.0 * np.pi)

def get_lane_radius(lane_index):
    """
    Return the radius of a given lane (0-indexed)
    based on the innermost lane's radius.
    """
    return BASE_RADIUS + lane_index * LANE_WIDTH

def alpha_to_xy(alpha, lane):
    """
    Convert arc-length alpha to Cartesian (x, y) for a given lane.
    - alpha in [0, 2000)
    - lane is an integer (0 = innermost lane, 1, 2, etc.)
    """
    # Convert alpha to angle (theta) in radians
    theta = 2.0 * np.pi * (alpha / CIRCUMFERENCE)
    # Get lane radius
    r = get_lane_radius(lane)
    # Convert to (x, y)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def load_data(csv_path):
    """
    Load the CSV data into a Pandas DataFrame and sort timesteps.
    Expected CSV columns: car_id, timestep, alpha, lane, speed.
    Returns:
      df: The entire DataFrame
      timesteps: Sorted list of unique timesteps
    """
    df = pd.read_csv(csv_path)
    timesteps = sorted(df["timestep"].unique())
    return df, timesteps

def setup_figure_and_axes(df1, df2):
    """
    Create a matplotlib figure with two subplots (ax1 and ax2),
    and a single colorbar for speed.
      1) Compute global min/max speed across both data sets.
      2) Create a single colorbar for them.
      3) Draw the lane circles in each subplot.
    """
    # 1 row, 2 columns, each subplot 6 inches wide => total (12,6)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    fig.suptitle(
        "Circular Animation comparison between Human-Based and Smart-Based"
        " Model with Density = 140", fontsize=14, fontweight='bold')
    # Capped max speed for better visualization
    capped_max1 = df1["speed"].quantile(0.95)
    capped_max2 = df2["speed"].quantile(0.95)
    # Compute min/max speed across both DataFrames
    global_min_speed = min(df1["speed"].min(), df2["speed"].min())
    # global_max_speed = max(df1["speed"].max(), df2["speed"].max())
    global_max_speed = max(capped_max1.max(), capped_max2.max())

    # Create a shared colormap and norm
    cmap = plt.colormaps.get_cmap("coolwarm")
    norm = mpl.colors.Normalize(vmin=global_min_speed, vmax=global_max_speed)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    # A single colorbar for both subplots
    cbar = fig.colorbar(
        sm, ax=[ax1, ax2],
        orientation="vertical",
        fraction=0.05,
        pad=0.05
    )
    cbar.set_label("Speed (m/s)")

    # For a consistent aspect ratio
    ax1.set_aspect("equal", adjustable="box")
    ax2.set_aspect("equal", adjustable="box")

    # Titles for each subplot
    ax1.set_title("Human-Based", fontsize=12)
    ax2.set_title("Smart-City", fontsize=12)

    # Remove axis ticks
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Determine a global bounding box that fits the largest lane in both datasets
    max_lane_df1 = df1["lane"].max()
    max_lane_df2 = df2["lane"].max()
    global_max_lane = max(max_lane_df1, max_lane_df2)
    outer_radius = get_lane_radius(global_max_lane)
    padding = 100  # extra space

    for ax in (ax1, ax2):
        ax.set_xlim(-outer_radius - padding, outer_radius + padding)
        ax.set_ylim(-outer_radius - padding, outer_radius + padding)

    # Draw lane circles on each subplot
    theta_vals = np.linspace(0, 2 * np.pi, 360)
    for lane_i in range(global_max_lane + 1):
        r_lane = get_lane_radius(lane_i)
        x_lane = r_lane * np.cos(theta_vals)
        y_lane = r_lane * np.sin(theta_vals)
        ax1.plot(x_lane, y_lane, "--", color="gray", linewidth=0.5)
        ax2.plot(x_lane, y_lane, "--", color="gray", linewidth=0.5)

    return fig, ax1, ax2, cmap, norm

def init_animation(scat1, scat2, time_text):
    """
    Initialize the scatter plots (clear them). Return both as a tuple.
    Also reset the time_text to an empty string.
    """
    scat1.set_offsets(np.empty((0, 2)))
    scat2.set_offsets(np.empty((0, 2)))

    # Clear the time text
    time_text.set_text("")

    # We return everything that needs to be re-drawn
    return (scat1, scat2, time_text)

def update_animation(frame, df1, df2, objects, cmap, norm):
    """
    For each frame (timestep), update both subplots:
      - Filter df1 for this timestep, convert (alpha, lane) to (x,y).
      - Filter df2 for this timestep, convert (alpha, lane) to (x,y).
      - Update each scatter (positions and colors).
      - Update time_text to show the current frame (timestep).
    """
    scat1, scat2, time_text = objects  # each is a PathCollection or Text

    # --- Update subplot 1 ---
    data_t1 = df1[df1["timestep"] == frame]
    x_vals1 = []
    y_vals1 = []
    colors1 = []
    for _, row in data_t1.iterrows():
        x, y = alpha_to_xy(row["alpha"], row["lane"])
        x_vals1.append(x)
        y_vals1.append(y)
        colors1.append(row["speed"])

    if len(x_vals1) > 0:
        scat1.set_offsets(np.column_stack((x_vals1, y_vals1)))
        scat1.set_color(cmap(norm(colors1)))
    else:
        # Provide an empty 2D array
        scat1.set_offsets(np.empty((0, 2)))

    # --- Update subplot 2 ---
    data_t2 = df2[df2["timestep"] == frame]
    x_vals2 = []
    y_vals2 = []
    colors2 = []
    for _, row in data_t2.iterrows():
        x, y = alpha_to_xy(row["alpha"], row["lane"])
        x_vals2.append(x)
        y_vals2.append(y)
        colors2.append(row["speed"])

    if len(x_vals2) > 0:
        scat2.set_offsets(np.column_stack((x_vals2, y_vals2)))
        scat2.set_color(cmap(norm(colors2)))
    else:
        scat2.set_offsets(np.empty((0, 2)))

    # Update the time text #
    time_text.set_text(f"t = {frame}")

    # Return the updated artists
    return (scat1, scat2, time_text)

def main():
    """
    Main function to:
      1) Parse command-line arguments.
      2) Load two CSV data files.
      3) Create a figure with two subplots (side by side).
      4) Animate both subplots using FuncAnimation.
      5) Save animation to a GIF file.
      6) Show the figure window.
    """
    if len(sys.argv) != 4:
        print("Usage: python circular_animation.py <data_file1.csv>"
              "<data_file2.csv> <output.gif>")
        sys.exit(1)

    data_file1 = sys.argv[1]
    data_file2 = sys.argv[2]
    output_file = sys.argv[3]

    # 1) Load both data files
    df1, timesteps1 = load_data(data_file1)
    df2, timesteps2 = load_data(data_file2)

    # Combine all timesteps so both animations advance in lockstep
    all_timesteps = sorted(set(timesteps1) | set(timesteps2))

    # 2) Create figure and axes for the two datasets
    fig, ax1, ax2, cmap, norm = setup_figure_and_axes(df1, df2)

    # 3) Create two empty scatter plots for the animation
    scat1 = ax1.scatter([], [], c="red", s=15, alpha=0.8)
    scat2 = ax2.scatter([], [], c="blue", s=15, alpha=0.8)

    #Text on the first subplot to display the current time
    time_text = ax1.text(
        0.05, 0.95, "",
        transform=ax1.transAxes,
        fontsize=12,
        color="black",
        verticalalignment="top"
    )

    # 4) Build the animation
    ani = animation.FuncAnimation(
        fig,
        func=update_animation,
        frames=all_timesteps,
        init_func=lambda: init_animation(scat1, scat2, time_text),
        fargs=(df1, df2, (scat1, scat2, time_text), cmap, norm),
        blit=True,
        repeat=True
    )

    # 5) Save the resulting GIF
    ani.save(filename=output_file, writer="pillow")
    print(f"Animation saved to {output_file}")

    # 6) Show the figure window
    plt.show()

if __name__ == "__main__":
    main()
