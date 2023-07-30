def KINETIC_ENERGY(Mass: float, Velocity: float):
    """
    Velocity: m/s
    Mass: kg
    
    In this case, The Kinetic energy is obtained in Joules.
    """
    return 0.5 * Mass * (Velocity ** 2)

def GRAVITATIONAL_POTENTIAL_ENERGY(Mass: float, Gravitational_acceleration: float, Height: float):
    """
    Mass: kg
    Gravitational acceleration: m/s^2
    Height: m
    
    In this case, The Gravitational potential energy is obtained in Joules.
    """
    return Mass * Gravitational_acceleration * Height

def MECHANICAL_ENERGY(Kinetic_Energy: float, Gravitational_Potential_Energy: float):
    """
    Kinetic energy: J
    Gravitational potential energy: J
    """
    return Kinetic_Energy + Gravitational_Potential_Energy

def ELASTIC_POTENTIAL_ENERGY(k: float, x: float):
    """
    k: N/m
    x: m
    
    In this case, The Elastic potential energy is obtained in Joules.
    """
    return 0.5 * k * (x ** 2)