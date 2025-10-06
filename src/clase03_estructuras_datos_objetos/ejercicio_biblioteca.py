"""
Ejercicio integrador: Sistema de gestión de biblioteca.
Combina estructuras de datos (listas, diccionarios, sets) y POO (clases, herencia).
"""


class Libro:
    """Representa un libro de la biblioteca."""
    
    def __init__(self, isbn, titulo, autor, año, genero):
        self.isbn = isbn
        self.titulo = titulo
        self.autor = autor
        self.año = año
        self.genero = genero
        self.disponible = True
        self.prestado_a = None
    
    def __str__(self):
        estado = "Disponible" if self.disponible else f"Prestado a {self.prestado_a}"
        return f"'{self.titulo}' de {self.autor} ({self.año}) - {estado}"
    
    def __repr__(self):
        return f"Libro('{self.isbn}', '{self.titulo}', '{self.autor}', {self.año}, '{self.genero}')"
    
    def __eq__(self, otro):
        """Dos libros son iguales si tienen el mismo ISBN."""
        return isinstance(otro, Libro) and self.isbn == otro.isbn
    
    def __hash__(self):
        """Permite usar libros en sets y como claves de diccionarios."""
        return hash(self.isbn)


class Usuario:
    """Clase base para usuarios de la biblioteca."""
    
    def __init__(self, dni, nombre, email):
        self.dni = dni
        self.nombre = nombre
        self.email = email
        self.libros_prestados = []  # Lista de ISBNs
    
    def __str__(self):
        return f"{self.nombre} ({self.dni})"
    
    def __repr__(self):
        return f"Usuario('{self.dni}', '{self.nombre}', '{self.email}')"
    
    def __eq__(self, otro):
        """Dos usuarios son iguales si tienen el mismo DNI."""
        return isinstance(otro, Usuario) and self.dni == otro.dni
    
    def limite_prestamos(self):
        """Método que será sobrescrito por las clases hijas."""
        return 3
    
    def puede_pedir_prestamo(self):
        """Comprobar si el usuario puede pedir más libros."""
        return len(self.libros_prestados) < self.limite_prestamos()
    
    def num_libros_prestados(self):
        """Número de libros que tiene el usuario."""
        return len(self.libros_prestados)


class UsuarioEstandar(Usuario):
    """Usuario estándar: límite de 3 libros."""
    
    def limite_prestamos(self):
        return 3


class UsuarioPremium(Usuario):
    """Usuario premium: límite de 10 libros."""
    
    def __init__(self, dni, nombre, email, fecha_premium):
        super().__init__(dni, nombre, email)
        self.fecha_premium = fecha_premium
    
    def limite_prestamos(self):
        return 10
    
    def __str__(self):
        return f"{self.nombre} ({self.dni}) [PREMIUM]"


class Biblioteca:
    """Sistema de gestión de biblioteca."""
    
    def __init__(self, nombre):
        self.nombre = nombre
        self.libros = {}  # ISBN -> Libro
        self.usuarios = {}  # DNI -> Usuario
        self.historial_prestamos = []  # Lista de tuplas (isbn, dni, fecha)
    
    # ========== GESTIÓN DE LIBROS ==========
    
    def añadir_libro(self, libro):
        """Añadir un libro al catálogo."""
        if libro.isbn in self.libros:
            print(f"Error: Ya existe un libro con ISBN {libro.isbn}")
            return False
        
        self.libros[libro.isbn] = libro
        print(f"✓ Libro añadido: {libro.titulo}")
        return True
    
    def eliminar_libro(self, isbn):
        """Eliminar un libro del catálogo."""
        if isbn not in self.libros:
            print(f"Error: No existe libro con ISBN {isbn}")
            return False
        
        libro = self.libros[isbn]
        if not libro.disponible:
            print(f"Error: No se puede eliminar '{libro.titulo}' (está prestado)")
            return False
        
        del self.libros[isbn]
        print(f"✓ Libro eliminado: {libro.titulo}")
        return True
    
    def buscar_por_titulo(self, titulo):
        """Buscar libros por título (búsqueda parcial, case-insensitive)."""
        titulo_lower = titulo.lower()
        resultados = [
            libro for libro in self.libros.values()
            if titulo_lower in libro.titulo.lower()
        ]
        return resultados
    
    def buscar_por_autor(self, autor):
        """Buscar libros por autor."""
        autor_lower = autor.lower()
        resultados = [
            libro for libro in self.libros.values()
            if autor_lower in libro.autor.lower()
        ]
        return resultados
    
    def buscar_por_genero(self, genero):
        """Buscar libros por género."""
        genero_lower = genero.lower()
        resultados = [
            libro for libro in self.libros.values()
            if genero_lower == libro.genero.lower()
        ]
        return resultados
    
    def libros_disponibles(self):
        """Obtener lista de libros disponibles."""
        return [libro for libro in self.libros.values() if libro.disponible]
    
    # ========== GESTIÓN DE USUARIOS ==========
    
    def registrar_usuario(self, usuario):
        """Registrar un nuevo usuario."""
        if usuario.dni in self.usuarios:
            print(f"Error: Ya existe un usuario con DNI {usuario.dni}")
            return False
        
        self.usuarios[usuario.dni] = usuario
        print(f"✓ Usuario registrado: {usuario.nombre}")
        return True
    
    def eliminar_usuario(self, dni):
        """Eliminar un usuario."""
        if dni not in self.usuarios:
            print(f"Error: No existe usuario con DNI {dni}")
            return False
        
        usuario = self.usuarios[dni]
        if usuario.libros_prestados:
            print(f"Error: {usuario.nombre} tiene libros prestados. Debe devolverlos primero.")
            return False
        
        del self.usuarios[dni]
        print(f"✓ Usuario eliminado: {usuario.nombre}")
        return True
    
    # ========== PRÉSTAMOS Y DEVOLUCIONES ==========
    
    def prestar_libro(self, isbn, dni):
        """Prestar un libro a un usuario."""
        # Validaciones
        if isbn not in self.libros:
            print(f"Error: No existe libro con ISBN {isbn}")
            return False
        
        if dni not in self.usuarios:
            print(f"Error: No existe usuario con DNI {dni}")
            return False
        
        libro = self.libros[isbn]
        usuario = self.usuarios[dni]
        
        if not libro.disponible:
            print(f"Error: '{libro.titulo}' no está disponible (prestado a {libro.prestado_a})")
            return False
        
        if not usuario.puede_pedir_prestamo():
            print(f"Error: {usuario.nombre} ha alcanzado el límite de préstamos ({usuario.limite_prestamos()})")
            return False
        
        # Realizar préstamo
        libro.disponible = False
        libro.prestado_a = usuario.nombre
        usuario.libros_prestados.append(isbn)
        self.historial_prestamos.append((isbn, dni, "2025-01-15"))  # Fecha simulada
        
        print(f"✓ Préstamo realizado: '{libro.titulo}' → {usuario.nombre}")
        return True
    
    def devolver_libro(self, isbn, dni):
        """Devolver un libro."""
        # Validaciones
        if isbn not in self.libros:
            print(f"Error: No existe libro con ISBN {isbn}")
            return False
        
        if dni not in self.usuarios:
            print(f"Error: No existe usuario con DNI {dni}")
            return False
        
        libro = self.libros[isbn]
        usuario = self.usuarios[dni]
        
        if libro.disponible:
            print(f"Error: '{libro.titulo}' no estaba prestado")
            return False
        
        if isbn not in usuario.libros_prestados:
            print(f"Error: {usuario.nombre} no tiene prestado '{libro.titulo}'")
            return False
        
        # Realizar devolución
        libro.disponible = True
        libro.prestado_a = None
        usuario.libros_prestados.remove(isbn)
        
        print(f"✓ Devolución realizada: '{libro.titulo}' ← {usuario.nombre}")
        return True
    
    # ========== ESTADÍSTICAS E INFORMES ==========
    
    def estadisticas(self):
        """Mostrar estadísticas de la biblioteca."""
        total_libros = len(self.libros)
        disponibles = len([l for l in self.libros.values() if l.disponible])
        prestados = total_libros - disponibles
        
        total_usuarios = len(self.usuarios)
        premium = len([u for u in self.usuarios.values() if isinstance(u, UsuarioPremium)])
        
        print(f"{'=' * 60}")
        print(f"ESTADÍSTICAS DE {self.nombre.upper()}")
        print(f"{'=' * 60}")
        print(f"Libros:")
        print(f"  Total: {total_libros}")
        print(f"  Disponibles: {disponibles}")
        print(f"  Prestados: {prestados}")
        print(f"Usuarios:")
        print(f"  Total: {total_usuarios}")
        print(f"  Premium: {premium}")
        print(f"  Estándar: {total_usuarios - premium}")
        print(f"Préstamos realizados: {len(self.historial_prestamos)}")
    
    def listar_libros(self, mostrar_todos=True):
        """Listar todos los libros o solo los disponibles."""
        if mostrar_todos:
            libros = self.libros.values()
            titulo = "TODOS LOS LIBROS"
        else:
            libros = self.libros_disponibles()
            titulo = "LIBROS DISPONIBLES"
        
        print(f"\n{titulo}:")
        if not libros:
            print("  (vacío)")
        else:
            for libro in sorted(libros, key=lambda l: l.titulo):
                print(f"  {libro}")
    
    def listar_usuarios(self):
        """Listar todos los usuarios."""
        print(f"\nUSUARIOS REGISTRADOS:")
        if not self.usuarios:
            print("  (vacío)")
        else:
            for usuario in sorted(self.usuarios.values(), key=lambda u: u.nombre):
                prestamos = f"{usuario.num_libros_prestados()}/{usuario.limite_prestamos()}"
                print(f"  {usuario} - Libros: {prestamos}")
    
    def autores_populares(self, top=5):
        """Autores con más libros en la biblioteca."""
        conteo_autores = {}
        for libro in self.libros.values():
            conteo_autores[libro.autor] = conteo_autores.get(libro.autor, 0) + 1
        
        # Ordenar por número de libros (descendente)
        top_autores = sorted(conteo_autores.items(), key=lambda x: x[1], reverse=True)[:top]
        
        print(f"\nTOP {top} AUTORES:")
        for i, (autor, cantidad) in enumerate(top_autores, 1):
            print(f"  {i}. {autor}: {cantidad} libro(s)")
    
    def generos_disponibles(self):
        """Géneros únicos en la biblioteca."""
        generos = {libro.genero for libro in self.libros.values()}
        return sorted(generos)


def ejemplo_completo():
    """Ejemplo completo del sistema de biblioteca."""
    print("=" * 60)
    print("SISTEMA DE GESTIÓN DE BIBLIOTECA")
    print("=" * 60)
    print()
    
    # Crear biblioteca
    biblioteca = Biblioteca("Biblioteca Municipal")
    
    # Añadir libros
    print("--- AÑADIENDO LIBROS ---")
    libros = [
        Libro("978-0-123456-47-2", "1984", "George Orwell", 1949, "Distopía"),
        Libro("978-0-234567-58-3", "Cien años de soledad", "Gabriel García Márquez", 1967, "Realismo mágico"),
        Libro("978-0-345678-69-4", "El Quijote", "Miguel de Cervantes", 1605, "Clásico"),
        Libro("978-0-456789-70-5", "Rebelión en la granja", "George Orwell", 1945, "Distopía"),
        Libro("978-0-567890-81-6", "El amor en los tiempos del cólera", "Gabriel García Márquez", 1985, "Romance"),
    ]
    
    for libro in libros:
        biblioteca.añadir_libro(libro)
    print()
    
    # Registrar usuarios
    print("--- REGISTRANDO USUARIOS ---")
    usuarios = [
        UsuarioEstandar("12345678A", "Ana García", "ana@example.com"),
        UsuarioPremium("87654321B", "Carlos López", "carlos@example.com", "2025-12-31"),
        UsuarioEstandar("11223344C", "Beatriz Ruiz", "beatriz@example.com"),
    ]
    
    for usuario in usuarios:
        biblioteca.registrar_usuario(usuario)
    print()
    
    # Listar todo
    biblioteca.listar_libros()
    biblioteca.listar_usuarios()
    print()
    
    # Buscar libros
    print("--- BÚSQUEDAS ---")
    print("Buscar 'Orwell':")
    for libro in biblioteca.buscar_por_autor("Orwell"):
        print(f"  {libro}")
    print()
    
    print("Buscar género 'Distopía':")
    for libro in biblioteca.buscar_por_genero("Distopía"):
        print(f"  {libro}")
    print()
    
    # Realizar préstamos
    print("--- PRÉSTAMOS ---")
    biblioteca.prestar_libro("978-0-123456-47-2", "12345678A")  # Ana: 1984
    biblioteca.prestar_libro("978-0-456789-70-5", "12345678A")  # Ana: Rebelión en la granja
    biblioteca.prestar_libro("978-0-234567-58-3", "87654321B")  # Carlos: Cien años
    print()
    
    # Intentar préstamo inválido
    print("--- INTENTOS INVÁLIDOS ---")
    biblioteca.prestar_libro("978-0-123456-47-2", "11223344C")  # Ya prestado
    print()
    
    # Listar disponibles
    biblioteca.listar_libros(mostrar_todos=False)
    print()
    
    # Devoluciones
    print("--- DEVOLUCIONES ---")
    biblioteca.devolver_libro("978-0-123456-47-2", "12345678A")
    print()
    
    # Estadísticas
    biblioteca.estadisticas()
    biblioteca.autores_populares()
    print()
    
    # Géneros disponibles
    print("Géneros disponibles:")
    for genero in biblioteca.generos_disponibles():
        print(f"  - {genero}")
    print()


def ejercicio_propuesto():
    """Descripción del ejercicio para los alumnos."""
    print("=" * 60)
    print("EJERCICIO PROPUESTO PARA PRACTICAR")
    print("=" * 60)
    print("""
Extiende el sistema de biblioteca con las siguientes funcionalidades:

1. NIVEL BÁSICO:
   - Añadir método para listar libros de un autor específico
   - Añadir método para ver historial de un usuario
   - Implementar búsqueda por año de publicación

2. NIVEL INTERMEDIO:
   - Añadir clase UsuarioEstudiante (hereda de Usuario, límite 5 libros)
   - Implementar sistema de multas por retraso
   - Añadir fecha real a los préstamos (usar datetime)
   - Método para calcular multa según días de retraso

3. NIVEL AVANZADO:
   - Sistema de reservas (si un libro está prestado, poder reservarlo)
   - Implementar lista de espera con notificaciones
   - Añadir valoraciones y reseñas de libros
   - Estadísticas avanzadas: libro más popular, usuario más activo
   - Exportar catálogo a archivo JSON o CSV

PISTAS:
- Usa diccionarios para mapeos rápidos (ISBN->Libro, DNI->Usuario)
- Usa listas para colecciones ordenadas (historial, prestamos)
- Usa sets para elementos únicos (géneros, autores)
- Aprovecha la herencia para diferentes tipos de usuarios
- Usa propiedades para cálculos automáticos (multa, etc.)
    """)


if __name__ == "__main__":
    ejemplo_completo()
    print()
    ejercicio_propuesto()
    
    print("=" * 60)
    print("FIN DEL EJERCICIO INTEGRADOR")
    print("=" * 60)
