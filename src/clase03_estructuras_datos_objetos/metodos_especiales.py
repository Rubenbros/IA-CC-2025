"""
Métodos especiales (dunder methods) en Python.
También llamados "magic methods" o métodos mágicos.
Permiten personalizar el comportamiento de las clases.
"""


class Punto:
    """Clase con métodos especiales básicos."""
    
    def __init__(self, x, y):
        """Constructor."""
        self.x = x
        self.y = y
    
    def __str__(self):
        """
        Representación legible para humanos.
        Se usa con print() y str().
        Equivalente a toString() en Java.
        """
        return f"Punto({self.x}, {self.y})"
    
    def __repr__(self):
        """
        Representación técnica del objeto.
        Debe idealmente permitir recrear el objeto.
        Se usa en el intérprete interactivo.
        """
        return f"Punto(x={self.x}, y={self.y})"
    
    def __eq__(self, otro):
        """
        Comparación de igualdad (==).
        Equivalente a equals() en Java.
        """
        if not isinstance(otro, Punto):
            return False
        return self.x == otro.x and self.y == otro.y


def ejemplo_str_repr():
    """Ejemplo de __str__ y __repr__."""
    print("=" * 60)
    print("MÉTODOS __str__ Y __repr__")
    print("=" * 60)
    
    p = Punto(3, 4)
    
    # print() usa __str__
    print(f"print(p): {p}")
    
    # str() usa __str__
    print(f"str(p): {str(p)}")
    
    # repr() usa __repr__
    print(f"repr(p): {repr(p)}")
    
    # En el intérprete se usa __repr__
    print(f"Representación técnica: {p!r}")
    print()


def ejemplo_eq():
    """Ejemplo de comparación de igualdad."""
    print("=" * 60)
    print("MÉTODO __eq__ (comparación ==)")
    print("=" * 60)
    
    p1 = Punto(3, 4)
    p2 = Punto(3, 4)
    p3 = Punto(5, 6)
    
    print(f"p1: {p1}")
    print(f"p2: {p2}")
    print(f"p3: {p3}")
    print()
    
    # Sin __eq__, compararía identidad (si son el mismo objeto)
    # Con __eq__, compara valores
    print(f"p1 == p2: {p1 == p2}")  # True (mismos valores)
    print(f"p1 == p3: {p1 == p3}")  # False
    print(f"p1 is p2: {p1 is p2}")  # False (diferentes objetos en memoria)
    print()


class Libro:
    """Clase con más métodos especiales."""
    
    def __init__(self, titulo, autor, paginas):
        self.titulo = titulo
        self.autor = autor
        self.paginas = paginas
    
    def __str__(self):
        return f"'{self.titulo}' de {self.autor}"
    
    def __repr__(self):
        return f"Libro('{self.titulo}', '{self.autor}', {self.paginas})"
    
    def __len__(self):
        """
        Longitud del objeto (usada por len()).
        """
        return self.paginas
    
    def __eq__(self, otro):
        """Dos libros son iguales si tienen mismo título y autor."""
        if not isinstance(otro, Libro):
            return False
        return self.titulo == otro.titulo and self.autor == otro.autor
    
    def __lt__(self, otro):
        """
        Menor que (<).
        Permite ordenar libros por número de páginas.
        """
        return self.paginas < otro.paginas
    
    def __le__(self, otro):
        """Menor o igual (<=)."""
        return self.paginas <= otro.paginas
    
    def __gt__(self, otro):
        """Mayor que (>)."""
        return self.paginas > otro.paginas
    
    def __ge__(self, otro):
        """Mayor o igual (>=)."""
        return self.paginas >= otro.paginas


def ejemplo_comparaciones():
    """Ejemplo de métodos de comparación."""
    print("=" * 60)
    print("MÉTODOS DE COMPARACIÓN (__lt__, __gt__, etc.)")
    print("=" * 60)
    
    libro1 = Libro("El Quijote", "Cervantes", 1000)
    libro2 = Libro("1984", "Orwell", 300)
    libro3 = Libro("El Hobbit", "Tolkien", 310)
    
    print(f"Libro 1: {libro1} ({len(libro1)} páginas)")
    print(f"Libro 2: {libro2} ({len(libro2)} páginas)")
    print(f"Libro 3: {libro3} ({len(libro3)} páginas)")
    print()
    
    # Comparaciones
    print(f"libro1 > libro2: {libro1 > libro2}")
    print(f"libro2 < libro1: {libro2 < libro1}")
    print(f"libro2 < libro3: {libro2 < libro3}")
    print()
    
    # Ordenar lista de libros
    libros = [libro1, libro2, libro3]
    libros_ordenados = sorted(libros)
    
    print("Libros ordenados por número de páginas:")
    for libro in libros_ordenados:
        print(f"  {libro} ({len(libro)} páginas)")
    print()


class Vector:
    """Vector matemático con operaciones aritméticas."""
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __str__(self):
        return f"Vector({self.x}, {self.y})"
    
    def __add__(self, otro):
        """
        Suma de vectores (+).
        """
        return Vector(self.x + otro.x, self.y + otro.y)
    
    def __sub__(self, otro):
        """Resta de vectores (-)."""
        return Vector(self.x - otro.x, self.y - otro.y)
    
    def __mul__(self, escalar):
        """Multiplicación por escalar (*)."""
        return Vector(self.x * escalar, self.y * escalar)
    
    def __truediv__(self, escalar):
        """División por escalar (/)."""
        return Vector(self.x / escalar, self.y / escalar)
    
    def __abs__(self):
        """Magnitud del vector (abs())."""
        return (self.x ** 2 + self.y ** 2) ** 0.5
    
    def __neg__(self):
        """Negación del vector (-)."""
        return Vector(-self.x, -self.y)


def ejemplo_operaciones_aritmeticas():
    """Ejemplo de operaciones aritméticas."""
    print("=" * 60)
    print("OPERACIONES ARITMÉTICAS (__add__, __mul__, etc.)")
    print("=" * 60)
    
    v1 = Vector(3, 4)
    v2 = Vector(1, 2)
    
    print(f"v1 = {v1}")
    print(f"v2 = {v2}")
    print()
    
    # Operaciones
    suma = v1 + v2
    resta = v1 - v2
    multiplicacion = v1 * 2
    division = v1 / 2
    
    print(f"v1 + v2 = {suma}")
    print(f"v1 - v2 = {resta}")
    print(f"v1 * 2 = {multiplicacion}")
    print(f"v1 / 2 = {division}")
    print()
    
    # abs y negación
    print(f"|v1| (magnitud) = {abs(v1):.2f}")
    print(f"-v1 = {-v1}")
    print()


class ListaPersonalizada:
    """Clase que se comporta como una lista."""
    
    def __init__(self):
        self._items = []
    
    def __len__(self):
        """len(lista)."""
        return len(self._items)
    
    def __getitem__(self, indice):
        """
        Acceso por índice: lista[indice].
        Permite usar [] para acceder a elementos.
        """
        return self._items[indice]
    
    def __setitem__(self, indice, valor):
        """
        Asignación por índice: lista[indice] = valor.
        """
        self._items[indice] = valor
    
    def __delitem__(self, indice):
        """
        Eliminación por índice: del lista[indice].
        """
        del self._items[indice]
    
    def __contains__(self, item):
        """
        Pertenencia: item in lista.
        """
        return item in self._items
    
    def __iter__(self):
        """
        Iteración: for item in lista.
        """
        return iter(self._items)
    
    def __str__(self):
        return f"ListaPersonalizada({self._items})"
    
    def append(self, item):
        """Añadir elemento."""
        self._items.append(item)


def ejemplo_contenedor():
    """Ejemplo de clase que se comporta como contenedor."""
    print("=" * 60)
    print("COMPORTAMIENTO DE CONTENEDOR (__getitem__, __setitem__, etc.)")
    print("=" * 60)
    
    lista = ListaPersonalizada()
    
    # Añadir elementos
    lista.append(10)
    lista.append(20)
    lista.append(30)
    
    print(f"Lista: {lista}")
    print(f"Longitud: {len(lista)}")
    print()
    
    # Acceso por índice
    print(f"lista[0]: {lista[0]}")
    print(f"lista[1]: {lista[1]}")
    print()
    
    # Modificar por índice
    lista[1] = 25
    print(f"Después de lista[1] = 25: {lista}")
    print()
    
    # Pertenencia
    print(f"¿10 está en la lista? {10 in lista}")
    print(f"¿99 está en la lista? {99 in lista}")
    print()
    
    # Iteración
    print("Iterando sobre la lista:")
    for item in lista:
        print(f"  {item}")
    print()


class CuentaBancaria:
    """Cuenta bancaria con métodos especiales."""
    
    def __init__(self, titular, saldo=0):
        self.titular = titular
        self._saldo = saldo  # "Privado" por convención
    
    def __str__(self):
        return f"Cuenta de {self.titular}: {self._saldo}€"
    
    def __repr__(self):
        return f"CuentaBancaria('{self.titular}', {self._saldo})"
    
    def __bool__(self):
        """
        Valor booleano: True si hay saldo positivo.
        Se usa en if cuenta: ...
        """
        return self._saldo > 0
    
    def __float__(self):
        """Conversión a float."""
        return float(self._saldo)
    
    def __int__(self):
        """Conversión a int."""
        return int(self._saldo)
    
    def __add__(self, otra):
        """Sumar saldos de dos cuentas (crear nueva cuenta)."""
        total = self._saldo + otra._saldo
        return CuentaBancaria(f"{self.titular}+{otra.titular}", total)
    
    def depositar(self, cantidad):
        self._saldo += cantidad
    
    def retirar(self, cantidad):
        if cantidad <= self._saldo:
            self._saldo -= cantidad
            return True
        return False


def ejemplo_cuenta_especial():
    """Ejemplo con cuenta bancaria y métodos especiales."""
    print("=" * 60)
    print("EJEMPLO INTEGRADOR: CUENTA BANCARIA")
    print("=" * 60)
    
    cuenta1 = CuentaBancaria("Ana", 1000)
    cuenta2 = CuentaBancaria("Carlos", 500)
    cuenta_vacia = CuentaBancaria("David", 0)
    
    print(cuenta1)
    print(cuenta2)
    print(cuenta_vacia)
    print()
    
    # __bool__
    if cuenta1:
        print(f"Cuenta de {cuenta1.titular} tiene saldo")
    
    if not cuenta_vacia:
        print(f"Cuenta de {cuenta_vacia.titular} está vacía")
    print()
    
    # Conversiones
    print(f"Saldo como float: {float(cuenta1)}")
    print(f"Saldo como int: {int(cuenta1)}")
    print()
    
    # Suma de cuentas
    cuenta_conjunta = cuenta1 + cuenta2
    print(f"Cuenta conjunta: {cuenta_conjunta}")
    print()


class Temperatura:
    """Clase con conversiones."""
    
    def __init__(self, celsius):
        self.celsius = celsius
    
    def __str__(self):
        return f"{self.celsius}°C"
    
    def __float__(self):
        """Devuelve en Celsius."""
        return float(self.celsius)
    
    def __int__(self):
        """Devuelve en Celsius redondeado."""
        return int(self.celsius)
    
    def __format__(self, format_spec):
        """
        Formateo personalizado.
        format_spec puede ser 'C' (Celsius), 'F' (Fahrenheit), 'K' (Kelvin)
        """
        if format_spec == 'F':
            fahrenheit = self.celsius * 9/5 + 32
            return f"{fahrenheit:.1f}°F"
        elif format_spec == 'K':
            kelvin = self.celsius + 273.15
            return f"{kelvin:.2f}K"
        else:  # Default: Celsius
            return f"{self.celsius}°C"


def ejemplo_format():
    """Ejemplo de __format__."""
    print("=" * 60)
    print("MÉTODO __format__ (formateo personalizado)")
    print("=" * 60)
    
    temp = Temperatura(25)
    
    print(f"Temperatura por defecto: {temp}")
    print(f"En Celsius: {temp:C}")
    print(f"En Fahrenheit: {temp:F}")
    print(f"En Kelvin: {temp:K}")
    print()


def resumen_metodos():
    """Resumen de métodos especiales importantes."""
    print("=" * 60)
    print("RESUMEN DE MÉTODOS ESPECIALES")
    print("=" * 60)
    
    print("""
REPRESENTACIÓN:
  __str__()      → str(), print() [legible para humanos]
  __repr__()     → repr() [técnica, reproducible]
  __format__()   → format(), f-strings con especificador

COMPARACIÓN:
  __eq__()       → ==
  __ne__()       → !=
  __lt__()       → <
  __le__()       → <=
  __gt__()       → >
  __ge__()       → >=

ARITMÉTICA:
  __add__()      → +
  __sub__()      → -
  __mul__()      → *
  __truediv__()  → /
  __floordiv__() → //
  __mod__()      → %
  __pow__()      → **

UNARIOS:
  __neg__()      → -x
  __pos__()      → +x
  __abs__()      → abs()

CONTENEDORES:
  __len__()      → len()
  __getitem__()  → obj[key]
  __setitem__()  → obj[key] = value
  __delitem__()  → del obj[key]
  __contains__() → in
  __iter__()     → for x in obj

CONVERSIÓN:
  __int__()      → int()
  __float__()    → float()
  __str__()      → str()
  __bool__()     → bool(), if obj

OTROS:
  __call__()     → obj() (hacer objeto llamable)
  __hash__()     → hash() (para usar como clave de dict)
    """)
    print()


if __name__ == "__main__":
    ejemplo_str_repr()
    ejemplo_eq()
    ejemplo_comparaciones()
    ejemplo_operaciones_aritmeticas()
    ejemplo_contenedor()
    ejemplo_cuenta_especial()
    ejemplo_format()
    resumen_metodos()
    
    print("=" * 60)
    print("FIN DE LOS EJEMPLOS DE MÉTODOS ESPECIALES")
    print("=" * 60)
