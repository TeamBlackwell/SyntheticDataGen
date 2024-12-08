import matplotlib.pyplot as plt
import pandas as pd


def drone_visualization(map_path, drone_path, mapsize, figsize):
    """
    Visualize the cityscape with buildings and drone positions.
    
    Parameters
    - map_path: path to cityscape csv file.
    - drone_path: path to drone csv file.
    - figsize: size of the plot.
    """

    # Create the main plot
    plt.figure(figsize=figsize)
    
    # Plot buildings
    buildings_df = pd.read_csv(map_path)
    buildings_df.columns = ["x1", "y1", "x2", "y2", "height"]
    
    # Plot each building as a rectangle
    for _, building in buildings_df.iterrows():
        plt.gca().add_patch(
            plt.Rectangle(
                (building['x1'], building['y1']),
                building['x2'] - building['x1'],
                building['y2'] - building['y1'],
                fill=False,
                edgecolor='gray',
                linewidth=1
            )
        )
    
    # Plot drone positions

    robot_coords = pd.read_csv(drone_path)
    plt.scatter(
        robot_coords['xr'], 
        robot_coords['yr'], 
        color='red', 
        s=100, 
        label='Drone Positions'
    )
    
    # Annotate drone positions
    for i, row in robot_coords.iterrows():
        plt.annotate(
            f'Drone {i+1}', 
            (row['xr'], row['yr']), 
            xytext=(10, 10),
            textcoords='offset points',
            color='red',
            fontweight='bold'
        )
    
    # Set plot properties
    plt.title('Cityscape Visualization')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.xlim(0, mapsize)
    plt.ylim(0, mapsize)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.axis('equal')
    
    # Show the plot
    plt.tight_layout()
    plt.show()
