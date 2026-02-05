"""Classical mechanics formulas."""


def momentum(mass: float, velocity: float):
    """Calculate linear momentum.

    Args:
        mass: Mass in kilograms (kg).
        velocity: Velocity in meters per second (m/s).

    Returns:
        Momentum in kilogram meters per second (kg·m/s).
    """
    return mass * velocity


def newton_second_law(mass: float, acceleration: float):
    """Calculate force using Newton's second law.

    Args:
        mass: Mass in kilograms (kg).
        acceleration: Acceleration in meters per second squared (m/s²).

    Returns:
        Force in newtons (N).
    """
    return mass * acceleration


def spring_potential_energy(k: float, x: float):
    """Calculate potential energy stored in a spring.

    Args:
        k: Spring constant in newtons per meter (N/m).
        x: Displacement from equilibrium in meters (m).

    Returns:
        Spring potential energy in joules (J).
    """
    return 0.5 * k * (x ** 2)


def centripetal_acceleration(velocity: float, radius: float):
    """Calculate centripetal acceleration.

    Args:
        velocity: Tangential speed in meters per second (m/s).
        radius: Radius of circular motion in meters (m).

    Returns:
        Centripetal acceleration in meters per second squared (m/s²).
    """
    return (velocity ** 2) / radius


def centripetal_force(mass: float, velocity: float, radius: float):
    """Calculate centripetal force for circular motion.

    Args:
        mass: Mass in kilograms (kg).
        velocity: Tangential speed in meters per second (m/s).
        radius: Radius of circular motion in meters (m).

    Returns:
        Centripetal force in newtons (N).
    """
    return mass * (velocity ** 2) / radius
