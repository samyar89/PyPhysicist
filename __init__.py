def V(I: float, R: float):
    """
    I: A
    
    R: Ω
    
    In this case, The Voltage is obtained in Volts.
    """
    return I * R

def I(V: float, R: float):
    """
    V: V
    
    R: Ω
    
    In this case, The Electric current intensity is obtained in Amperes. 
    """
    return V / R

def R(V: float, I: float):
    """
    V: V
    
    I: A
    
    In this case, The Electrical resistance is obtained in Ohms.
    """
    return V / I

def Velocity(d: float, t: float):
    """
    d: m
    
    t: s
    
    In this case, The Velocity is obtained in m/s.
    """
    return d / t

def F(m: float, a: float):
    """
    m : kg
    
    a: m/s^2
    
    In this case, the force is obtained in Newtons.
    """
    return m * a
    
def Weight(m: float, g: float):
    """
    m: kg
    
    g: m/s^2
    
    In this case, the Weight is obtained in Newtons.
    """
    return m * g

def Work(F: float, d: float):
    """
    F: N
    
    d: m
    
    In this case, the Weight is obtained in Joules.
    """
    return F * d

def Schwarzschild_radius(m: float):
    """
    m: kg
    
    G: Gravitational constant
    c: Speed of light
    
    In this case, the Schwarzschild radius is obtained in Meters.
    """
    G = 6.67430 * (10 ** -11)
    c = 299792458
    
    return (2 * G * m) / (c ** 2)