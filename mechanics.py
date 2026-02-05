"""Classical mechanics formulas."""


def momentum(mass: float, velocity: float):
    """
    mass: kg
    velocity: m/s

    In this case, the momentum is obtained in kg·m/s.
    """
    return mass * velocity


def newton_second_law(mass: float, acceleration: float):
    """
    mass: kg
    acceleration: m/s^2

    In this case, the force is obtained in Newtons.
    """
    return mass * acceleration


def spring_potential_energy(k: float, x: float):
    """
    k: N/m
    x: m

    In this case, the spring potential energy is obtained in Joules.
    """
    return 0.5 * k * (x ** 2)


def centripetal_acceleration(velocity: float, radius: float):
    """
    velocity: m/s
    radius: m

    In this case, the centripetal acceleration is obtained in m/s^2.
    """
    return (velocity ** 2) / radius


def centripetal_force(mass: float, velocity: float, radius: float):
    """
    mass: kg
    velocity: m/s
    radius: m

    In this case, the centripetal force is obtained in Newtons.
    """
    return mass * (velocity ** 2) / radius
