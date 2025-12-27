import numpy as np


# car object that behaves like a car
class VehicleAgent(object):
    def __init__(
        self,
        position,
        current_speed,
        desired_speed=30,
        max_speed=35,
        length=5,
        a_normal=3.05,
        a_max=6.04,
        b=0.2,
        TP=1.2,
        AC=0.5,
    ):
        """
        The parameters of the vehicle agent are:
        - position: the range is [0, 2000> and indicates the position of the vehicle on the lane
        - current speed: the current speed of the vehicle in m/s
        - desired speed: the desired speed of the vehicle, 30 m/s
        - max speed: the maximum speed of the vehicle, 35 m/s
        - length: the length of the vehicle, which is 5m
        - a normal: the normal acceleration, used in specific scenarios (refer to decceleration_rate())
        - a max: the maximum acceleration, used in specific scenarios (refer to decceleration_rate())
        - b: the noise of the model, accounting for the real world environment
        - TP: the time headway the follower prefers to the vehicle in front
        - AC: the chance that in a centralized control environment this car tries to increase the mean spead
        """

        self.position = position
        self.current_speed = current_speed
        self.desired_speed = desired_speed
        self.max_speed = max_speed
        self.length = length
        self.a_normal = a_normal
        self.a_max = a_max
        self.b = b
        self.TP = TP
        self.acceleration = 0
        self.AC = AC

        # used to save data which can be exported
        self.speed_list = [0]
        self.position_list = [0]
        self.lane_list = [0]

    def __lt__(self, other):
        return self.position < other.position

    def reset_data(self):
        """
        Resets the data the Agent hold, can be used for example when resetting a simulation
        with the same cars
        """
        self.speed_list = [0]
        self.position_list = [0]
        self.lane_list = [0]
        pass

    def compute_decision(self, gap, leader_speed, leader_acceleration):
        """
        The Decision Tree. Returns the decision: accelerate,
        deccelerate or cruise.
        """
        vF = self.current_speed
        vL = leader_speed
        gap_desire = vF * self.TP
        if gap < 6 * vF:
            delta = gap - gap_desire

            if delta > 0:
                if vL >= vF:
                    # accelerate
                    self.acceleration = self.acceleration_rate(vF)

                else:
                    if gap > 3 * vF:
                        # accelerate
                        self.acceleration = self.acceleration_rate(vF)

                    else:
                        if (gap > 2 * vF) and (gap > 7.5):
                            # cruise
                            self.acceleration = 0

                        else:
                            # deccelerate
                            self.acceleration = self.decceleration_rate(
                                vF, vL, leader_acceleration, gap, gap_desire
                            )

            elif delta == 0:
                if vL >= vF:
                    # cruise
                    self.acceleration = 0

                else:
                    # deccelerate
                    self.acceleration = self.decceleration_rate(
                        vF, vL, leader_acceleration, gap, gap_desire
                    )

            elif delta < 0:
                if vL > vF:
                    # cruise
                    self.acceleration = 0

                else:
                    # deccelerate
                    self.acceleration = self.decceleration_rate(
                        vF, vL, leader_acceleration, gap, gap_desire
                    )

        else:
            delta = vF - self.desired_speed

            if delta < 0:
                # accelerate
                self.acceleration = self.acceleration_rate(vF)

            elif delta == 0:
                # cruise
                self.acceleration = 0

            elif delta > 0:
                # deccelerate
                self.acceleration = self.decceleration_rate(
                    vF, vL, leader_acceleration, gap, gap_desire
                )

    def decceleration_rate(self, vF, vL, aL, gap, gap_desire):
        # case 1: free flowing case
        if vF > self.desired_speed:
            aF = min((vF - self.desired_speed) / 3, self.a_normal)
            return aF

        # case 2: car-following regime, normal decceleration
        if gap - gap_desire == 0:
            if not vL >= vF:
                aF = (vF**2 - vL**2) / (2 * gap)
                return aF

        # case 3: emergency decceleration
        if gap - gap_desire < 0:
            if not vL > vF:
                aF = aL - 0.25 * self.a_normal
                return aF

        # case 4: near-collision decceleration
        if (
            (gap - gap_desire > 0)
            and (not vL >= vF)
            and (not gap > 3 * vF)
            and (not ((gap > 2 * vF) and (gap > 7.5)))
        ):
            aF = min(aL + ((vF - vL) ** 2 / (2 * gap)), self.a_max)
            return aF

        return 0

    def acceleration_rate(self, vF):
        if vF <= 12.19:
            aF = 1.1

        elif vF > 12.19:
            aF = 0.37

        return aF

    def can_switch_lane(self, cars):
        # TODO(Seb): account for the speed of other cars
        if cars is None:
            # No lane here
            return False
        elif cars == (None, None):
            # Lane is empty
            return True

        car_front, car_behind = cars
        if car_front == car_behind:
            return (
                self.position < car_front.position - car_front.length
                or self.position - self.length > car_front.position
            )
        if self.position > car_front.position - car_front.length:
            return False
        elif self.position - self.length < car_behind.position:
            return False
        return True

    def can_switch_lanes(self, cars_left, cars_right):
        assert cars_left != cars_right, f"{cars_left} == {cars_right}"
        return (
            self.can_switch_lane(cars_left),
            self.can_switch_lane(cars_right),
        )

    def greedy_lane_switch(self, car_front, cars_left, cars_right, road_length):
        """Greedily switch to any lane if current lane has a slow leader."""
        can_go_left, can_go_right = self.can_switch_lanes(cars_left, cars_right)
        if can_go_left == can_go_right == False:
            return 0

        speed_difference = car_front.current_speed - self.current_speed
        gap = (car_front.position - self.position) % road_length
        if speed_difference < -5.0 and gap < self.current_speed * 5.0:
            # Car in front is close and slow enough that another lane is preferred.
            # These cases are ordered such that the left lane takes priority.
            if can_go_left:
                return 1
            elif can_go_right:
                return -1
            else:
                return 0
        return 0

    def traditional_lane_switch(self, car_front, cars_left, cars_right, road_length):
        """Switch lanes by using fast (left) lanes and slow (right) lanes."""
        can_go_left, can_go_right = self.can_switch_lanes(cars_left, cars_right)
        if can_go_left == can_go_right == False:
            return 0

        speed_difference = car_front.current_speed - self.current_speed
        gap = (car_front.position - self.position) % road_length
        if speed_difference < -5.0 and gap < self.current_speed * 5.0:
            # Car in front is close and slow enough the passing lane is preferred.
            if can_go_left:
                return 1
            else:
                return 0

        if can_go_right:
            car_front_right = cars_right[0]
            if car_front_right is None:
                # Slow lane is empty
                return -1
            gap_fr = (car_front_right.position - self.position) % road_length
            if gap_fr > 200:
                # Next car in slow lane is far enough away
                return -1

        return 0

    def lane_switch(self, car_front, cars_left, cars_right, road_length):
        """
        Returns where this agent wants to switch lanes to. This is given by an
        integer -1, 0, or 1, meaning switch left, stay in lane, and switch right,
        respectively.

        WARNING: Must not be resolved concurrently with acceleration changes,
        otherwise agents might not respond to the new car in their lane.
        """
        return self.traditional_lane_switch(
            car_front, cars_left, cars_right, road_length
        )

    def compute_safe_speed(self, gap, leader_speed):
        reaction_time = 1
        v_safe = leader_speed + (
            (gap - leader_speed * reaction_time)
            / (reaction_time + ((self.current_speed + leader_speed) / (2 * self.a_max)))
        )
        return v_safe

    def calculate_next_state(
        self,
        gap,
        leader_speed,
        leader_acceleration,
        dt,
        mean_speed=None,
        max_accel=1,
        speed_push=0.5,
    ):
        """
        Updates the state of the follower (current vehicle), which
        depends on the speed and acceleration of the leader (car in front),
        and the gap between them.

        mean_spead = None means no centralized control
        mac_accel = the maximum acceleration to meet the mean speed
        speed_push = the maximum acceleration added to push the average on mean speed
        """

        self.compute_decision(gap, leader_speed, leader_acceleration)
        v_safe = self.compute_safe_speed(gap, leader_speed)
        v_ideal = min(
            self.max_speed, self.current_speed + self.acceleration * dt, v_safe
        )
        eta = None
        # eta is only random when no centralized control
        if mean_speed == None:
            eta = np.random.rand()
        else:
            eta = 0.5

        self.next_speed = max(0, v_ideal - self.b * eta)
        # if no central control
        if mean_speed != None:
            dif_speed = mean_speed - self.next_speed
            # normalized dif_speed
            dif_speed = max(-max_accel, min(max_accel, dif_speed))
            self.next_speed += dif_speed
            # if with these speeds there is a collision in 3 seconds don't try and increase the average mean_speed
            if (
                0 < gap + 3 * (leader_speed - self.next_speed)
            ) and np.random.rand() < self.AC:
                self.next_speed += speed_push

    def update_state(self, dt, lane):
        self.current_speed = self.next_speed
        self.position += self.current_speed * dt

        # updata data
        self.speed_list.append(self.current_speed)
        self.position_list.append(self.position)
        self.lane_list.append(lane)
