"""
Ejemplos prácticos de clases y objetos en Python.
Conceptos fundamentales de POO: clases, objetos, atributos, métodos.
"""


def ejemplo_sin_clases():
    """Por qué necesitamos clases: comparación con enfoque sin POO."""
    print("=" * 60)
    print("SIN CLASES (enfoque procedural)")
    print("=" * 60)
    
    # Sin clases: usar diccionarios y funciones sueltas
    persona1 = {"nombre": "Ana", "edad": 20, "email": "ana@example.com"}
    persona2 = {"nombre": "Carlos", "edad": 22, "email": "carlos@example.com"}
    
    def presentarse(persona):
        return f"Hola, soy {persona['nombre']} y tengo {persona['edad']} años"
    
    print(presentarse(persona1))
    print(presentarse(persona2))
    print()
    print("Problema: los datos y las funciones están separados.")
    print("Solución: clases que agrupan datos y comportamiento.")
    print()


class Persona:
    """Clase básica para representar una persona."""
    
    def __init__(self, nombre, edad, email):
        """
        Constructor: se ejecuta al crear un objeto.
        self es como 'this' en Java, pero DEBE ser explícito.
        """
        self.nombre = nombre
        self.edad = edad
        self.email = email
    
    def presentarse(self):
        """Método de instancia: tiene acceso a los atributos del objeto."""
        return f"Hola, soy {self.nombre} y tengo {self.edad} años"
    
    def cumplir_años(self):
        """Método que modifica el estado del objeto."""
        self.edad += 1
        print(f"¡Feliz cumpleaños! Ahora {self.nombre} tiene {self.edad} años")
    
    def es_mayor_de_edad(self):
        """Método que devuelve un booleano."""
        return self.edad >= 18


def ejemplo_con_clases():
    """Ejemplo básico usando clases."""
    print("=" * 60)
    print("CON CLASES (enfoque POO)")
    print("=" * 60)
    
    # Crear objetos (instancias de la clase)
    # En Python NO se usa 'new'
    persona1 = Persona("Ana", 20, "ana@example.com")
    persona2 = Persona("Carlos", 22, "carlos@example.com")
    
    print(f"persona1: {type(persona1)}")
    print(f"persona1.nombre: {persona1.nombre}")
    print(f"persona1.edad: {persona1.edad}")
    print()
    
    # Llamar a métodos
    print(persona1.presentarse())
    print(persona2.presentarse())
    print()
    
    # Modificar el objeto
    persona1.cumplir_años()
    print()
    
    # Método que devuelve booleano
    print(f"¿{persona1.nombre} es mayor de edad? {persona1.es_mayor_de_edad()}")
    print()


class CuentaBancaria:
    """Ejemplo más completo: cuenta bancaria."""
    
    def __init__(self, titular, saldo_inicial=0):
        """
        Parámetros por defecto: saldo_inicial=0
        """
        self.titular = titular
        self.saldo = saldo_inicial
        self.movimientos = []  # Lista para historial
    
    def depositar(self, cantidad):
        """Añadir dinero a la cuenta."""
        if cantidad <= 0:
            print("Error: la cantidad debe ser positiva")
            return False
        
        self.saldo += cantidad
        self.movimientos.append(f"Depósito: +{cantidad}€")
        print(f"Depositados {cantidad}€. Saldo: {self.saldo}€")
        return True
    
    def retirar(self, cantidad):
        """Retirar dinero de la cuenta."""
        if cantidad <= 0:
            print("Error: la cantidad debe ser positiva")
            return False
        
        if cantidad > self.saldo:
            print(f"Error: saldo insuficiente (disponible: {self.saldo}€)")
            return False
        
        self.saldo -= cantidad
        self.movimientos.append(f"Retirada: -{cantidad}€")
        print(f"Retirados {cantidad}€. Saldo: {self.saldo}€")
        return True
    
    def consultar_saldo(self):
        """Ver el saldo actual."""
        return self.saldo
    
    def ver_historial(self):
        """Mostrar todos los movimientos."""
        print(f"Historial de {self.titular}:")
        if not self.movimientos:
            print("  No hay movimientos")
        for movimiento in self.movimientos:
            print(f"  {movimiento}")
        print(f"Saldo actual: {self.saldo}€")


def ejemplo_cuenta_bancaria():
    """Usar la clase CuentaBancaria."""
    print("=" * 60)
    print("EJEMPLO: CUENTA BANCARIA")
    print("=" * 60)
    
    # Crear cuenta con saldo inicial
    cuenta = CuentaBancaria("Ana García", 1000)
    print(f"Cuenta creada: {cuenta.titular}, saldo: {cuenta.saldo}€")
    print()
    
    # Operaciones
    cuenta.depositar(500)
    cuenta.retirar(200)
    cuenta.retirar(2000)  # Error: insuficiente
    print()
    
    # Ver historial
    cuenta.ver_historial()
    print()


class Rectangulo:
    """Ejemplo con métodos calculados."""
    
    def __init__(self, ancho, alto):
        self.ancho = ancho
        self.alto = alto
    
    def area(self):
        """Calcular el área."""
        return self.ancho * self.alto
    
    def perimetro(self):
        """Calcular el perímetro."""
        return 2 * (self.ancho + self.alto)
    
    def es_cuadrado(self):
        """Comprobar si es un cuadrado."""
        return self.ancho == self.alto
    
    def escalar(self, factor):
        """Cambiar el tamaño manteniendo las proporciones."""
        self.ancho *= factor
        self.alto *= factor


def ejemplo_rectangulo():
    """Usar la clase Rectangulo."""
    print("=" * 60)
    print("EJEMPLO: RECTÁNGULO")
    print("=" * 60)
    
    rect = Rectangulo(5, 3)
    print(f"Rectángulo: {rect.ancho} x {rect.alto}")
    print(f"Área: {rect.area()}")
    print(f"Perímetro: {rect.perimetro()}")
    print(f"¿Es cuadrado? {rect.es_cuadrado()}")
    print()
    
    # Escalar
    rect.escalar(2)
    print(f"Después de escalar x2: {rect.ancho} x {rect.alto}")
    print(f"Nueva área: {rect.area()}")
    print()
    
    # Cuadrado
    cuadrado = Rectangulo(4, 4)
    print(f"Cuadrado: {cuadrado.ancho} x {cuadrado.alto}")
    print(f"¿Es cuadrado? {cuadrado.es_cuadrado()}")
    print()


class Estudiante:
    """Ejemplo con lista de calificaciones."""
    
    def __init__(self, nombre, dni):
        self.nombre = nombre
        self.dni = dni
        self.notas = []  # Lista vacía al crear el estudiante
    
    def añadir_nota(self, asignatura, nota):
        """Añadir una nota."""
        if 0 <= nota <= 10:
            self.notas.append({"asignatura": asignatura, "nota": nota})
            print(f"Nota añadida: {asignatura} = {nota}")
        else:
            print("Error: la nota debe estar entre 0 y 10")
    
    def calcular_promedio(self):
        """Calcular el promedio de todas las notas."""
        if not self.notas:
            return 0
        
        suma = sum(n["nota"] for n in self.notas)
        return suma / len(self.notas)
    
    def mostrar_notas(self):
        """Mostrar todas las notas."""
        print(f"Notas de {self.nombre} ({self.dni}):")
        if not self.notas:
            print("  No hay notas registradas")
        else:
            for item in self.notas:
                print(f"  {item['asignatura']}: {item['nota']}")
            print(f"Promedio: {self.calcular_promedio():.2f}")


def ejemplo_estudiante():
    """Usar la clase Estudiante."""
    print("=" * 60)
    print("EJEMPLO: ESTUDIANTE")
    print("=" * 60)
    
    estudiante = Estudiante("Carlos López", "12345678A")
    
    estudiante.añadir_nota("Matemáticas", 8.5)
    estudiante.añadir_nota("Física", 7.0)
    estudiante.añadir_nota("Programación", 9.5)
    estudiante.añadir_nota("Inglés", 6.5)
    print()
    
    estudiante.mostrar_notas()
    print()


class Contador:
    """Ejemplo simple para entender self."""
    
    def __init__(self, valor_inicial=0):
        self.valor = valor_inicial
    
    def incrementar(self):
        """Incrementar en 1."""
        self.valor += 1
    
    def decrementar(self):
        """Decrementar en 1."""
        self.valor -= 1
    
    def obtener_valor(self):
        """Obtener el valor actual."""
        return self.valor
    
    def reiniciar(self):
        """Volver a 0."""
        self.valor = 0


def ejemplo_contador():
    """Demostrar que cada objeto tiene su propio estado."""
    print("=" * 60)
    print("EJEMPLO: MÚLTIPLES CONTADORES (estado independiente)")
    print("=" * 60)
    
    # Crear dos contadores
    c1 = Contador(0)
    c2 = Contador(10)
    
    print(f"Contador 1 inicial: {c1.obtener_valor()}")
    print(f"Contador 2 inicial: {c2.obtener_valor()}")
    print()
    
    # Modificar c1
    c1.incrementar()
    c1.incrementar()
    c1.incrementar()
    
    # Modificar c2
    c2.decrementar()
    c2.decrementar()
    
    print(f"Contador 1 después de incrementar x3: {c1.obtener_valor()}")
    print(f"Contador 2 después de decrementar x2: {c2.obtener_valor()}")
    print()
    print("Cada objeto tiene su propio estado (self).")
    print()


def diferencias_con_java():
    """Resumen de diferencias con Java."""
    print("=" * 60)
    print("DIFERENCIAS CON JAVA")
    print("=" * 60)
    
    print("""
1. SELF vs THIS:
   Java:   this.nombre = nombre;
   Python: self.nombre = nombre  (self es explícito)

2. CREAR OBJETOS:
   Java:   Persona p = new Persona("Ana", 20);
   Python: p = Persona("Ana", 20)  (sin 'new')

3. CONSTRUCTOR:
   Java:   public Persona(String nombre, int edad) { ... }
   Python: def __init__(self, nombre, edad): ...

4. MODIFICADORES DE ACCESO:
   Java:   private, protected, public
   Python: No hay privado real, por convención:
           - público: nombre
           - "protegido": _nombre
           - "privado": __nombre

5. MÉTODOS:
   Java:   public void metodo() { ... }
   Python: def metodo(self): ...

6. TIPOS:
   Java:   Tipado estático (int edad)
   Python: Tipado dinámico (edad puede ser cualquier tipo)
    """)
    print()


if __name__ == "__main__":
    ejemplo_sin_clases()
    ejemplo_con_clases()
    ejemplo_cuenta_bancaria()
    ejemplo_rectangulo()
    ejemplo_estudiante()
    ejemplo_contador()
    diferencias_con_java()
    
    print("=" * 60)
    print("FIN DE LOS EJEMPLOS DE CLASES BÁSICAS")
    print("=" * 60)
