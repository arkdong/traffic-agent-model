import matplotlib.pyplot as plt
import numpy as np
from .track_interface import Track
from itertools import chain
import csv


class Model(object):
    def __init__(
        self,
        dt: float = 1.0,
        total_time: int = 2000,
        road_length: int = 2000,
        lane_count: int = 1,
        central_control=False,
        max_accel=1,
        speed_push=0.5,
    ) -> None:
        """
        The parameters of the simulation model are:
        - dt: the timestep in each iteration of the simulation
        - total_time: the total duration of the simulation
        - road_length: the length of the single lane
        - lane_count: The amount of lanes
        - central_control: if the central control model is used
        - mac_accel: the maximum acceleration to meet the mean speed of an agent
        - speed_push: the maximum acceleration added to push the average of the mean speed

        """

        self.dt = dt
        self.total_time = total_time
        self.road_length = road_length
        self.lane_count = lane_count
        self.road_length_km = self.road_length / 1000
        self.density_values = np.linspace(0, 140, 10)
        self.total_runs = 20  # In the paper they also did 20 runs for each density
        self.flow_results = [[] for _ in range(self.total_runs)]
        self.speed_results = [[] for _ in range(self.total_runs)]

        # central control arguments
        self.central_control = central_control
        self.max_accel = max_accel
        self.speed_push = speed_push

    def run(self, idx, export_data=False) -> None:
        """
        A single run of the simulation. In total, we will perform 20 runs,
        which is also done in the paper which we wanted to validate our model with.
        """

        for density in self.density_values:
            # N is amount of vehicles
            track = Track(
                lane_count=self.lane_count,
                length=self.road_length,
                central_control=self.central_control,
                max_accel=self.max_accel,
                speed_push=self.speed_push,
            )
            track.init_cars(density)

            # USED FOR CENTRAL CONTROL ==============================================
            total_cars = sum([len(lane) for lane in track.lanes_list])
            # prefered amount of cars per lane
            prefered_per_lane = [
                total_cars // track.lanes_count for lane in range(len(track.lanes_list))
            ]
            for i in range(total_cars % track.lanes_count):
                prefered_per_lane[i] += 1
            # ======================================================================
            total_crossings = (
                0  # count the total crossings at a fixed reference point in time
            )
            time_data = []
            for t in range(int(self.total_time / self.dt)):
                time_data.append(t * self.dt)
                # We don't need to run code that won't do anything
                if self.lane_count > 1:
                    if self.central_control:
                        track.lane_switches_central(prefered_per_lane)
                    else:
                        track.lane_switches()
                track.calculate_next_state()
                track.update_state()

                for lane in track.lanes_list:
                    for vehicle in lane:
                        # Periodic boundary condition
                        if vehicle.position >= self.road_length:
                            vehicle.position -= self.road_length
                            total_crossings += 1

            flow = total_crossings
            mean_speed = np.mean(
                [veh.current_speed for veh in chain(*track.lanes_list)]
            )

            self.flow_results[idx].append(flow)
            self.speed_results[idx].append(mean_speed)
            # per multiple runs only ones will the data be exported
            if idx == 0 and export_data:
                self.export_data(time_data, track.lanes_list, density=density)

    def export_data(self, time_data, lanes_list, density):
        """Exports the data of the position, speed, lane of every car in lanes_list into a csv file"""
        # filename contains arguments of the simulation
        filename = "data_"
        if self.central_control:
            filename += "central_"
        else:
            filename += "individual_"

        filename += str(len(lanes_list))
        filename += "lane_density_"
        filename += str(round(density))
        filename += ".csv"

        # write data to the file
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["id", "timestep", "alpha", "lane", "speed"])
            car_id = 0
            for lane in lanes_list:
                for vehicle in lane:
                    for i in range(len(time_data)):
                        writer.writerow(
                            [
                                car_id,
                                time_data[i],
                                vehicle.position_list[i],
                                vehicle.lane_list[i],
                                vehicle.speed_list[i],
                            ]
                        )
                    car_id += 1

    def plot(self, stat: str = "position", export_data=False, out_file=None) -> None:
        """
        Make the plot, which is either a flow-density graph or mean-speed-density graph.
        If stat = position, then the plot is a flow-density graph. If stat = velocity
        then the plot is a mean-speed-density graph.

        Export data means data of every agent in ONLY the first run will be saved to csv files
        """

        if stat == "position":
            for idx in range(self.total_runs):  # Run the simulation 20 times
                print("run " + idx)
                if idx == 1:
                    break
                self.run(idx, export_data=export_data)

                # Scatter plot for current run
                plt.scatter(
                    self.density_values,
                    self.flow_results[idx],
                    alpha=0.5,
                    color="gray",
                )

            # average flow
            avg_flow = (
                np.mean(self.flow_results, axis=0)
                if isinstance(self.flow_results[0], list)
                else self.flow_results
            )
            plt.plot(self.density_values, avg_flow, color="black", label="Mean Flow")
            plt.xlabel("Density (veh/km)")
            plt.ylabel("Flow (veh/h)")
            plt.title("The relationship between flow and density")
            plt.legend()

        elif stat == "velocity":
            for idx in range(self.total_runs):  # Run the simulation 20 times
                print(idx)
                self.run(idx, export_data=export_data)

                # Scatter plot for current run
                plt.scatter(
                    self.density_values,
                    self.speed_results[idx],
                    alpha=0.5,
                    color="gray",
                )

            # average speed
            avg_speed = (
                np.mean(self.speed_results, axis=0)
                if isinstance(self.speed_results[0], list)
                else self.speed_results
            )
            plt.plot(
                self.density_values,
                avg_speed,
                color="black",
                label="Mean Speed",
            )
            plt.xlabel("Density (veh/km)")
            plt.ylabel("Mean Speed (m/s)")
            plt.title("The relationship between mean-speed and density")
            plt.legend()

        else:
            raise ValueError(f"Unrecognised statistic '{stat}'")

        if out_file is None:
            plt.show()
        else:
            plt.savefig(out_file)


if __name__ == "__main__":
    # when you run "python -m src.model" you land here: the simulation will run
    model = Model(lane_count=2, central_control=False)
    model.plot(stat="velocity", export_data=True)
