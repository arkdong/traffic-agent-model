import unittest
from src.Agent import VehicleAgent


class AgentTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(AgentTest, self).__init__(*args, **kwargs)
        self.agent = VehicleAgent(position=1000, current_speed=25)

    def test_compute_decision(self):
        """
        This function tests the Decision Tree. In this specific scenario the speed of the
        follower is 25 m/s, the speed of the leader is also 25 m/s, the gap between them
        is 100 m. According to the decision tree, we should get in the condition where
        gap < 6 * vF(speed of follower) and delta > 0 and vL(speed of leader) >=
        vF, because vF = 25 m/s, vL = 25 m/s, gap_desire = vF * TP = 25 * 1.2 = 30 m, and
        therefore delta = 100 - 30 = 70m and vL >= vF also holds. So we should return the
        acceleration rate and that should be 0.37 as vF > 12.19 m/s.
        """

        decision = self.agent.compute_decision(
            gap=100, leader_speed=25, leader_acceleration=0.37
        )

        acceleration_decision = self.agent.acceleration

        self.assertEqual(acceleration_decision, 0.37)

    def test_decceleration_rate(self):
        """
        This function tests the decceleration rate. We use the exact same
        settings as before, to ensure consistency. In this case, the decceleration
        rate should be 0 m/s^2, this is because none of the 4 cases holds in this
        scenario, namely: 1) vF is not bigger than the desired speed. 2) gap - gap_desire
        is not equal to 0. 3) gap - gap_desire is not lower than 0. 4) gap - gap_desire is
        indeed bigger than 0, but vL is not smaller or equal than vF, so we know directly
        that this condition also does not hold. So the base case will be returned:
        decceleration_rate = 0 m/s^2.
        """

        vF = 25
        vL = 25
        aL = 0.37
        gap = 100
        gap_desire = 25 * 1.2

        decceleration_rate = self.agent.decceleration_rate(
            vF=vF, vL=vL, aL=aL, gap=100, gap_desire=gap_desire
        )

        self.assertEqual(decceleration_rate, 0)

    def test_acceleration_rate(self):
        """
        This function test the acceleration rate. If the speed of
        the follower, vF, is smaller or equal to 12.19 m/s, then
        the acceleration rate should be 1.1 m/s^2 as this test
        will confirm.
        """

        vF = 12.19
        acceleration_rate = self.agent.acceleration_rate(vF=vF)

        self.assertEqual(acceleration_rate, 1.1)

    def test_compute_safe_speed(self):
        """
        The safe speed is computed as:
        v_safe = vL(t) + ((gap - vL(t) * tau) / (tau + (vF(t) + vL(t) / 2*a_max)))
        Where vL(t) is the speed of the leader at time t, gap is the distance
        between the follower and leader, tau is de greek symbol representing
        the reaction time of a driver, which is 1s as in the paper, a_max
        is the maximum acceleration rate. This test should check that, if:
        gap = 100m, vL = 25 m/s, vF = 25 m/s, tau = 1s, a_max = 6.04 m/s^2,
        that then v_safe = 25 + ((100 - 25*1) / (1 + (25 + 25 / 2 * 6.04)))
        is approximately 39.59 m/s.
        """

        gap = 100
        leader_speed = 25

        v_safe = self.agent.compute_safe_speed(
            gap=gap, leader_speed=leader_speed
        )

        self.assertAlmostEqual(
            v_safe, 39.59, delta=0.1
        )  # within a standard deviation of 0.1 we find it OK


if __name__ == "__main__":
    unittest.main()  # run the tests
