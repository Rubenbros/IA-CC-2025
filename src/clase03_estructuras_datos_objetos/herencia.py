"""
Ejemplos de herencia en Python.
Concepto: crear clases hijas que heredan de clases padres.
"""


class Animal:
    """Clase base (padre/superclase)."""
    
    def __init__(self, nombre, edad):
        self.nombre = nombre
        self.edad = edad
    
    def comer(self):
        return f"{self.nombre} está comiendo"
    
    def dormir(self):
        return f"{self.nombre} está durmiendo"
    
    def hacer_sonido(self):
        """Método que será sobrescrito por las clases hijas."""
        return "Algún sonido"


class Perro(Animal):
    """Clase hija que hereda de Animal."""
    
    def __init__(self, nombre, edad, raza):
        # Llamar al constructor del padre con super()
        super().__init__(nombre, edad)
        # Añadir atributos propios
        self.raza = raza
    
    def hacer_sonido(self):
        """Sobrescribir el método del padre."""
        return f"{self.nombre} dice: ¡Guau guau!"
    
    def buscar_pelota(self):
        """Método específico de Perro."""
        return f"{self.nombre} está buscando la pelota"


class Gato(Animal):
    """Otra clase hija que hereda de Animal."""
    
    def __init__(self, nombre, edad, color):
        super().__init__(nombre, edad)
        self.color = color
    
    def hacer_sonido(self):
        """Sobrescribir el método del padre."""
        return f"{self.nombre} dice: ¡Miau!"
    
    def arañar_sofa(self):
        """Método específico de Gato."""
        return f"{self.nombre} está arañando el sofá"


def ejemplo_basico_herencia():
    """Ejemplo básico de herencia."""
    print("=" * 60)
    print("EJEMPLO BÁSICO DE HERENCIA")
    print("=" * 60)
    
    # Crear objetos de las clases hijas
    perro = Perro("Max", 3, "Labrador")
    gato = Gato("Luna", 2, "Gris")
    
    print(f"Perro: {perro.nombre}, {perro.edad} años, raza: {perro.raza}")
    print(f"Gato: {gato.nombre}, {gato.edad} años, color: {gato.color}")
    print()
    
    # Métodos heredados del padre
    print(perro.comer())
    print(gato.dormir())
    print()
    
    # Métodos sobrescritos (polimorfismo)
    print(perro.hacer_sonido())
    print(gato.hacer_sonido())
    print()
    
    # Métodos específicos de cada clase
    print(perro.buscar_pelota())
    print(gato.arañar_sofa())
    print()


class Vehiculo:
    """Clase base para vehículos."""
    
    def __init__(self, marca, modelo, año):
        self.marca = marca
        self.modelo = modelo
        self.año = año
        self.velocidad = 0
    
    def acelerar(self, incremento):
        self.velocidad += incremento
        return f"{self.marca} {self.modelo} acelera a {self.velocidad} km/h"
    
    def frenar(self):
        self.velocidad = 0
        return f"{self.marca} {self.modelo} se detiene"
    
    def descripcion(self):
        return f"{self.marca} {self.modelo} ({self.año})"


class Coche(Vehiculo):
    """Coche: vehículo con número de puertas."""
    
    def __init__(self, marca, modelo, año, num_puertas):
        super().__init__(marca, modelo, año)
        self.num_puertas = num_puertas
    
    def descripcion(self):
        """Sobrescribir y extender el método del padre."""
        base = super().descripcion()  # Llamar al método del padre
        return f"{base}, {self.num_puertas} puertas"
    
    def abrir_maletero(self):
        return f"Abriendo el maletero del {self.marca} {self.modelo}"


class Moto(Vehiculo):
    """Moto: vehículo con tipo (deportiva, custom, etc)."""
    
    def __init__(self, marca, modelo, año, tipo):
        super().__init__(marca, modelo, año)
        self.tipo = tipo
    
    def descripcion(self):
        base = super().descripcion()
        return f"{base}, tipo: {self.tipo}"
    
    def hacer_caballito(self):
        return f"¡{self.marca} {self.modelo} hace un caballito!"


def ejemplo_vehiculos():
    """Ejemplo con vehículos."""
    print("=" * 60)
    print("EJEMPLO: HERENCIA CON VEHÍCULOS")
    print("=" * 60)
    
    coche = Coche("Toyota", "Corolla", 2023, 4)
    moto = Moto("Yamaha", "MT-07", 2023, "Deportiva")
    
    print(coche.descripcion())
    print(moto.descripcion())
    print()
    
    # Métodos heredados
    print(coche.acelerar(50))
    print(moto.acelerar(80))
    print()
    
    print(coche.frenar())
    print(moto.frenar())
    print()
    
    # Métodos específicos
    print(coche.abrir_maletero())
    print(moto.hacer_caballito())
    print()


class Empleado:
    """Clase base para empleados."""
    
    def __init__(self, nombre, dni, salario_base):
        self.nombre = nombre
        self.dni = dni
        self.salario_base = salario_base
    
    def calcular_salario(self):
        """Método base, será sobrescrito por las clases hijas."""
        return self.salario_base
    
    def presentarse(self):
        return f"Empleado: {self.nombre}"


class EmpleadoTiempoCompleto(Empleado):
    """Empleado a tiempo completo con bonus."""
    
    def __init__(self, nombre, dni, salario_base, bonus):
        super().__init__(nombre, dni, salario_base)
        self.bonus = bonus
    
    def calcular_salario(self):
        """Salario base + bonus."""
        return self.salario_base + self.bonus
    
    def presentarse(self):
        return f"Empleado tiempo completo: {self.nombre}"


class EmpleadoPorHoras(Empleado):
    """Empleado por horas."""
    
    def __init__(self, nombre, dni, tarifa_por_hora, horas_trabajadas):
        # No usamos salario_base, pero lo ponemos a 0
        super().__init__(nombre, dni, 0)
        self.tarifa_por_hora = tarifa_por_hora
        self.horas_trabajadas = horas_trabajadas
    
    def calcular_salario(self):
        """Salario = tarifa * horas."""
        return self.tarifa_por_hora * self.horas_trabajadas
    
    def presentarse(self):
        return f"Empleado por horas: {self.nombre}"
    
    def registrar_horas(self, horas):
        """Método específico para registrar horas."""
        self.horas_trabajadas += horas
        print(f"{self.nombre}: {horas} horas registradas. Total: {self.horas_trabajadas}h")


def ejemplo_empleados():
    """Ejemplo con empleados y polimorfismo."""
    print("=" * 60)
    print("EJEMPLO: EMPLEADOS (polimorfismo)")
    print("=" * 60)
    
    emp1 = EmpleadoTiempoCompleto("Ana García", "12345A", 2000, 300)
    emp2 = EmpleadoPorHoras("Carlos López", "67890B", 15, 80)
    
    # Lista de empleados (polimorfismo)
    empleados = [emp1, emp2]
    
    print("Nómina del mes:")
    for emp in empleados:
        print(f"{emp.presentarse()}: {emp.calcular_salario()}€")
    print()
    
    # Método específico de EmpleadoPorHoras
    emp2.registrar_horas(10)
    print(f"Nuevo salario de {emp2.nombre}: {emp2.calcular_salario()}€")
    print()


class Figura:
    """Clase base abstracta para figuras geométricas."""
    
    def __init__(self, nombre):
        self.nombre = nombre
    
    def area(self):
        """Este método debe ser implementado por las clases hijas."""
        raise NotImplementedError("Las clases hijas deben implementar area()")
    
    def perimetro(self):
        """Este método debe ser implementado por las clases hijas."""
        raise NotImplementedError("Las clases hijas deben implementar perimetro()")


class Cuadrado(Figura):
    """Cuadrado."""
    
    def __init__(self, lado):
        super().__init__("Cuadrado")
        self.lado = lado
    
    def area(self):
        return self.lado ** 2
    
    def perimetro(self):
        return 4 * self.lado


class Circulo(Figura):
    """Círculo."""
    
    def __init__(self, radio):
        super().__init__("Círculo")
        self.radio = radio
    
    def area(self):
        return 3.14159 * (self.radio ** 2)
    
    def perimetro(self):
        return 2 * 3.14159 * self.radio


class Triangulo(Figura):
    """Triángulo."""
    
    def __init__(self, base, altura, lado1, lado2, lado3):
        super().__init__("Triángulo")
        self.base = base
        self.altura = altura
        self.lado1 = lado1
        self.lado2 = lado2
        self.lado3 = lado3
    
    def area(self):
        return (self.base * self.altura) / 2
    
    def perimetro(self):
        return self.lado1 + self.lado2 + self.lado3


def ejemplo_figuras():
    """Ejemplo con figuras geométricas (clases abstractas)."""
    print("=" * 60)
    print("EJEMPLO: FIGURAS GEOMÉTRICAS (clases abstractas)")
    print("=" * 60)
    
    figuras = [
        Cuadrado(5),
        Circulo(3),
        Triangulo(4, 3, 3, 4, 5)
    ]
    
    print("Cálculo de áreas y perímetros:")
    for figura in figuras:
        print(f"{figura.nombre}:")
        print(f"  Área: {figura.area():.2f}")
        print(f"  Perímetro: {figura.perimetro():.2f}")
    print()


def ejemplo_isinstance():
    """Comprobar tipos con isinstance()."""
    print("=" * 60)
    print("COMPROBAR TIPOS CON isinstance()")
    print("=" * 60)
    
    perro = Perro("Max", 3, "Labrador")
    gato = Gato("Luna", 2, "Gris")
    
    # isinstance: comprueba si un objeto es de una clase
    print(f"¿perro es Perro? {isinstance(perro, Perro)}")
    print(f"¿perro es Animal? {isinstance(perro, Animal)}")  # True (hereda)
    print(f"¿perro es Gato? {isinstance(perro, Gato)}")
    print()
    
    # issubclass: comprueba si una clase hereda de otra
    print(f"¿Perro es subclase de Animal? {issubclass(Perro, Animal)}")
    print(f"¿Gato es subclase de Animal? {issubclass(Gato, Animal)}")
    print(f"¿Perro es subclase de Gato? {issubclass(Perro, Gato)}")
    print()


class Usuario:
    """Clase base Usuario."""
    
    def __init__(self, nombre, email):
        self.nombre = nombre
        self.email = email
        self.activo = True
    
    def desactivar(self):
        self.activo = False
        print(f"Usuario {self.nombre} desactivado")


class UsuarioGratuito(Usuario):
    """Usuario gratuito con límites."""
    
    def __init__(self, nombre, email):
        super().__init__(nombre, email)
        self.limite_descargas = 5
        self.descargas_realizadas = 0
    
    def puede_descargar(self):
        return self.descargas_realizadas < self.limite_descargas
    
    def descargar(self):
        if self.puede_descargar():
            self.descargas_realizadas += 1
            return f"Descarga {self.descargas_realizadas}/{self.limite_descargas} completada"
        else:
            return "Límite de descargas alcanzado. Actualiza a premium."


class UsuarioPremium(Usuario):
    """Usuario premium sin límites."""
    
    def __init__(self, nombre, email, fecha_expiracion):
        super().__init__(nombre, email)
        self.fecha_expiracion = fecha_expiracion
    
    def puede_descargar(self):
        return True  # Sin límites
    
    def descargar(self):
        return "Descarga completada (usuario premium, sin límites)"


def ejemplo_usuarios():
    """Ejemplo con usuarios gratuitos y premium."""
    print("=" * 60)
    print("EJEMPLO: USUARIOS GRATUITOS Y PREMIUM")
    print("=" * 60)
    
    gratuito = UsuarioGratuito("Ana", "ana@example.com")
    premium = UsuarioPremium("Carlos", "carlos@example.com", "2025-12-31")
    
    print(f"{gratuito.nombre} (gratuito):")
    for i in range(7):
        print(f"  Intento {i+1}: {gratuito.descargar()}")
    print()
    
    print(f"{premium.nombre} (premium):")
    for i in range(3):
        print(f"  Descarga {i+1}: {premium.descargar()}")
    print()


if __name__ == "__main__":
    ejemplo_basico_herencia()
    ejemplo_vehiculos()
    ejemplo_empleados()
    ejemplo_figuras()
    ejemplo_isinstance()
    ejemplo_usuarios()
    
    print("=" * 60)
    print("FIN DE LOS EJEMPLOS DE HERENCIA")
    print("=" * 60)
