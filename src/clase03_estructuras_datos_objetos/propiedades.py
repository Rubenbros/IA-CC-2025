"""
Propiedades en Python: alternativa pythónica a getters y setters.
Usando el decorador @property.
"""


class PersonaSinPropiedades:
    """Enfoque Java: getters y setters explícitos."""
    
    def __init__(self, nombre, edad):
        self._nombre = nombre
        self._edad = edad
    
    def get_nombre(self):
        return self._nombre
    
    def set_nombre(self, nombre):
        self._nombre = nombre
    
    def get_edad(self):
        return self._edad
    
    def set_edad(self, edad):
        if edad < 0:
            raise ValueError("La edad no puede ser negativa")
        self._edad = edad


def ejemplo_sin_propiedades():
    """Ejemplo sin propiedades (estilo Java)."""
    print("=" * 60)
    print("SIN PROPIEDADES (estilo Java con getters/setters)")
    print("=" * 60)
    
    persona = PersonaSinPropiedades("Ana", 20)
    
    # Acceso con getters/setters
    print(f"Nombre: {persona.get_nombre()}")
    print(f"Edad: {persona.get_edad()}")
    
    persona.set_nombre("Ana García")
    persona.set_edad(21)
    
    print(f"Nuevo nombre: {persona.get_nombre()}")
    print(f"Nueva edad: {persona.get_edad()}")
    print()
    
    print("Problema: sintaxis verbosa, no pythónica")
    print()


class PersonaConPropiedades:
    """Enfoque Python: propiedades con @property."""
    
    def __init__(self, nombre, edad):
        self._nombre = nombre
        self._edad = edad
    
    @property
    def nombre(self):
        """Getter: se accede como atributo."""
        return self._nombre
    
    @nombre.setter
    def nombre(self, valor):
        """Setter: se asigna como atributo."""
        if not valor:
            raise ValueError("El nombre no puede estar vacío")
        self._nombre = valor
    
    @property
    def edad(self):
        """Getter de edad."""
        return self._edad
    
    @edad.setter
    def edad(self, valor):
        """Setter con validación."""
        if valor < 0:
            raise ValueError("La edad no puede ser negativa")
        if valor > 150:
            raise ValueError("La edad no puede ser mayor a 150")
        self._edad = valor
    
    @property
    def es_mayor_de_edad(self):
        """Propiedad calculada (solo getter, no setter)."""
        return self._edad >= 18


def ejemplo_con_propiedades():
    """Ejemplo con propiedades (estilo Python)."""
    print("=" * 60)
    print("CON PROPIEDADES (estilo Python con @property)")
    print("=" * 60)
    
    persona = PersonaConPropiedades("Ana", 20)
    
    # Acceso como atributos (¡pero con validación!)
    print(f"Nombre: {persona.nombre}")
    print(f"Edad: {persona.edad}")
    print(f"¿Es mayor de edad? {persona.es_mayor_de_edad}")
    print()
    
    # Asignación como atributos
    persona.nombre = "Ana García"
    persona.edad = 21
    
    print(f"Nuevo nombre: {persona.nombre}")
    print(f"Nueva edad: {persona.edad}")
    print(f"¿Es mayor de edad? {persona.es_mayor_de_edad}")
    print()
    
    # Validación en acción
    print("Intentando asignar edad negativa:")
    try:
        persona.edad = -5
    except ValueError as e:
        print(f"  Error: {e}")
    print()
    
    print("Intentando asignar nombre vacío:")
    try:
        persona.nombre = ""
    except ValueError as e:
        print(f"  Error: {e}")
    print()


class Rectangulo:
    """Rectángulo con propiedades calculadas."""
    
    def __init__(self, ancho, alto):
        self._ancho = ancho
        self._alto = alto
    
    @property
    def ancho(self):
        return self._ancho
    
    @ancho.setter
    def ancho(self, valor):
        if valor <= 0:
            raise ValueError("El ancho debe ser positivo")
        self._ancho = valor
    
    @property
    def alto(self):
        return self._alto
    
    @alto.setter
    def alto(self, valor):
        if valor <= 0:
            raise ValueError("El alto debe ser positivo")
        self._alto = valor
    
    @property
    def area(self):
        """Propiedad calculada: área."""
        return self._ancho * self._alto
    
    @property
    def perimetro(self):
        """Propiedad calculada: perímetro."""
        return 2 * (self._ancho + self._alto)
    
    @property
    def es_cuadrado(self):
        """Propiedad calculada: ¿es cuadrado?"""
        return self._ancho == self._alto


def ejemplo_propiedades_calculadas():
    """Ejemplo de propiedades calculadas."""
    print("=" * 60)
    print("PROPIEDADES CALCULADAS")
    print("=" * 60)
    
    rect = Rectangulo(5, 3)
    
    print(f"Rectángulo: {rect.ancho} x {rect.alto}")
    print(f"Área: {rect.area}")  # Se calcula automáticamente
    print(f"Perímetro: {rect.perimetro}")
    print(f"¿Es cuadrado? {rect.es_cuadrado}")
    print()
    
    # Cambiar dimensiones
    rect.ancho = 4
    rect.alto = 4
    
    print(f"Nuevo rectángulo: {rect.ancho} x {rect.alto}")
    print(f"Nueva área: {rect.area}")  # Se recalcula automáticamente
    print(f"¿Es cuadrado? {rect.es_cuadrado}")
    print()


class Temperatura:
    """Temperatura con conversión automática entre escalas."""
    
    def __init__(self, celsius=0):
        self._celsius = celsius
    
    @property
    def celsius(self):
        return self._celsius
    
    @celsius.setter
    def celsius(self, valor):
        self._celsius = valor
    
    @property
    def fahrenheit(self):
        """Propiedad calculada: Fahrenheit desde Celsius."""
        return self._celsius * 9/5 + 32
    
    @fahrenheit.setter
    def fahrenheit(self, valor):
        """Setter: actualiza Celsius desde Fahrenheit."""
        self._celsius = (valor - 32) * 5/9
    
    @property
    def kelvin(self):
        """Propiedad calculada: Kelvin desde Celsius."""
        return self._celsius + 273.15
    
    @kelvin.setter
    def kelvin(self, valor):
        """Setter: actualiza Celsius desde Kelvin."""
        self._celsius = valor - 273.15


def ejemplo_temperatura():
    """Ejemplo de conversión de temperatura."""
    print("=" * 60)
    print("EJEMPLO: CONVERSIÓN DE TEMPERATURA")
    print("=" * 60)
    
    temp = Temperatura(25)
    
    print(f"Temperatura: {temp.celsius}°C")
    print(f"En Fahrenheit: {temp.fahrenheit:.1f}°F")
    print(f"En Kelvin: {temp.kelvin:.2f}K")
    print()
    
    # Cambiar en Fahrenheit
    print("Cambiando a 98.6°F:")
    temp.fahrenheit = 98.6
    print(f"Celsius: {temp.celsius:.1f}°C")
    print(f"Kelvin: {temp.kelvin:.2f}K")
    print()
    
    # Cambiar en Kelvin
    print("Cambiando a 300K:")
    temp.kelvin = 300
    print(f"Celsius: {temp.celsius:.2f}°C")
    print(f"Fahrenheit: {temp.fahrenheit:.1f}°F")
    print()


class CuentaBancaria:
    """Cuenta bancaria con saldo protegido."""
    
    def __init__(self, titular, saldo_inicial=0):
        self._titular = titular
        self._saldo = saldo_inicial
        self._movimientos = []
    
    @property
    def titular(self):
        """El titular solo puede leerse, no modificarse."""
        return self._titular
    
    @property
    def saldo(self):
        """El saldo solo puede leerse, no modificarse directamente."""
        return self._saldo
    
    @property
    def movimientos(self):
        """Devolver copia de movimientos (para que no los modifiquen)."""
        return self._movimientos.copy()
    
    def depositar(self, cantidad):
        """Modificar saldo a través de método (controlado)."""
        if cantidad > 0:
            self._saldo += cantidad
            self._movimientos.append(f"Depósito: +{cantidad}€")
            return True
        return False
    
    def retirar(self, cantidad):
        """Modificar saldo a través de método (controlado)."""
        if 0 < cantidad <= self._saldo:
            self._saldo -= cantidad
            self._movimientos.append(f"Retirada: -{cantidad}€")
            return True
        return False


def ejemplo_propiedades_solo_lectura():
    """Ejemplo de propiedades de solo lectura."""
    print("=" * 60)
    print("PROPIEDADES DE SOLO LECTURA")
    print("=" * 60)
    
    cuenta = CuentaBancaria("Ana García", 1000)
    
    # Lectura OK
    print(f"Titular: {cuenta.titular}")
    print(f"Saldo: {cuenta.saldo}€")
    print()
    
    # Modificación a través de métodos
    cuenta.depositar(500)
    cuenta.retirar(200)
    print(f"Nuevo saldo: {cuenta.saldo}€")
    print()
    
    # Intentar modificar directamente: error
    print("Intentando modificar saldo directamente:")
    try:
        cuenta.saldo = 9999  # ¡Esto da error!
    except AttributeError as e:
        print(f"  Error: {e}")
    print()
    
    print("Intentar modificar titular:")
    try:
        cuenta.titular = "Otro"  # ¡Esto también da error!
    except AttributeError as e:
        print(f"  Error: {e}")
    print()


class Producto:
    """Producto con precio y descuento."""
    
    def __init__(self, nombre, precio_base):
        self.nombre = nombre
        self._precio_base = precio_base
        self._descuento = 0  # Porcentaje (0-100)
    
    @property
    def precio_base(self):
        return self._precio_base
    
    @precio_base.setter
    def precio_base(self, valor):
        if valor < 0:
            raise ValueError("El precio no puede ser negativo")
        self._precio_base = valor
    
    @property
    def descuento(self):
        return self._descuento
    
    @descuento.setter
    def descuento(self, valor):
        if not 0 <= valor <= 100:
            raise ValueError("El descuento debe estar entre 0 y 100")
        self._descuento = valor
    
    @property
    def precio_final(self):
        """Precio con descuento aplicado."""
        return self._precio_base * (1 - self._descuento / 100)
    
    @property
    def ahorro(self):
        """Cantidad ahorrada por el descuento."""
        return self._precio_base - self.precio_final


def ejemplo_producto():
    """Ejemplo de producto con descuento."""
    print("=" * 60)
    print("EJEMPLO: PRODUCTO CON DESCUENTO")
    print("=" * 60)
    
    producto = Producto("Portátil", 1000)
    
    print(f"Producto: {producto.nombre}")
    print(f"Precio base: {producto.precio_base}€")
    print(f"Descuento: {producto.descuento}%")
    print(f"Precio final: {producto.precio_final}€")
    print()
    
    # Aplicar descuento
    producto.descuento = 20
    print(f"Aplicado descuento del {producto.descuento}%")
    print(f"Precio final: {producto.precio_final}€")
    print(f"Ahorro: {producto.ahorro}€")
    print()
    
    # Validación
    print("Intentando descuento inválido (150%):")
    try:
        producto.descuento = 150
    except ValueError as e:
        print(f"  Error: {e}")
    print()


class Circulo:
    """Círculo: puedes configurar radio o diámetro."""
    
    def __init__(self, radio):
        self._radio = radio
    
    @property
    def radio(self):
        return self._radio
    
    @radio.setter
    def radio(self, valor):
        if valor <= 0:
            raise ValueError("El radio debe ser positivo")
        self._radio = valor
    
    @property
    def diametro(self):
        """El diámetro se calcula desde el radio."""
        return self._radio * 2
    
    @diametro.setter
    def diametro(self, valor):
        """Configurar diámetro actualiza el radio."""
        if valor <= 0:
            raise ValueError("El diámetro debe ser positivo")
        self._radio = valor / 2
    
    @property
    def area(self):
        return 3.14159 * (self._radio ** 2)
    
    @property
    def circunferencia(self):
        return 2 * 3.14159 * self._radio


def ejemplo_circulo():
    """Ejemplo con radio y diámetro."""
    print("=" * 60)
    print("EJEMPLO: CÍRCULO (radio y diámetro)")
    print("=" * 60)
    
    circulo = Circulo(5)
    
    print(f"Radio: {circulo.radio}")
    print(f"Diámetro: {circulo.diametro}")
    print(f"Área: {circulo.area:.2f}")
    print(f"Circunferencia: {circulo.circunferencia:.2f}")
    print()
    
    # Cambiar por radio
    circulo.radio = 10
    print(f"Nuevo radio: {circulo.radio}")
    print(f"Nuevo diámetro: {circulo.diametro}")
    print()
    
    # Cambiar por diámetro
    circulo.diametro = 30
    print(f"Diámetro configurado a: {circulo.diametro}")
    print(f"Radio actualizado a: {circulo.radio}")
    print()


def ventajas_propiedades():
    """Resumen de ventajas de usar propiedades."""
    print("=" * 60)
    print("VENTAJAS DE USAR @property")
    print("=" * 60)
    
    print("""
1. SINTAXIS LIMPIA:
   Java:   persona.getEdad() / persona.setEdad(21)
   Python: persona.edad / persona.edad = 21

2. VALIDACIÓN TRANSPARENTE:
   Puedes añadir validación sin cambiar la interfaz.
   El código que usa la clase no cambia.

3. PROPIEDADES CALCULADAS:
   rectangulo.area se calcula automáticamente.
   No necesitas llamar a calcular_area().

4. CONTROL DE ACCESO:
   Propiedades de solo lectura (sin setter).
   Proteger el estado interno.

5. CAMBIAR DE ATRIBUTO A PROPIEDAD:
   Puedes empezar con atributo simple y luego añadir
   validación sin romper el código existente.

CUÁNDO USAR:
- Validación de valores
- Cálculos automáticos
- Conversión entre unidades
- Acceso controlado a datos privados
- Cuando necesitas lazy loading

CUÁNDO NO USAR:
- Atributos simples sin lógica adicional
- Operaciones costosas (usa métodos explícitos)
    """)
    print()


if __name__ == "__main__":
    ejemplo_sin_propiedades()
    ejemplo_con_propiedades()
    ejemplo_propiedades_calculadas()
    ejemplo_temperatura()
    ejemplo_propiedades_solo_lectura()
    ejemplo_producto()
    ejemplo_circulo()
    ventajas_propiedades()
    
    print("=" * 60)
    print("FIN DE LOS EJEMPLOS DE PROPIEDADES")
    print("=" * 60)
