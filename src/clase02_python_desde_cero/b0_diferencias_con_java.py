"""
Diferencias clave: Java → Python (chuleta rápida)

Este archivo imprime y comenta diferencias habituales al cambiar de Java a Python.
Léelo y ejecútalo para ver ejemplos mínimos.
"""

if __name__ == "__main__":
    print("1) Estructura y entrada del programa")
    print("   - Java: clase con public static void main(String[] args)")
    print("   - Python: archivo .py ejecutable y bloque if __name__ == '__main__':")

    print("\n2) Tipado y variables")
    print("   - Java: tipado estático; declaras tipo de variable.")
    print("   - Python: tipado dinámico (pero admite anotaciones opcionales).")

    print("\n3) Sintaxis y bloques")
    print("   - Java: llaves { } y ; al final de línea.")
    print("   - Python: sin ;, bloques por indentación y dos puntos ':' al inicio del bloque.")

    print("\n4) Strings y formato")
    print("   - Java: String.format, +, text blocks.")
    print("   - Python: f-strings → f'Hola, {nombre}'.")

    print("\n5) Colecciones")
    print("   - Java: ArrayList, HashSet, HashMap...")
    print("   - Python: list, set, dict (métodos simples y bucles for directos sobre elementos).")

    print("\n6) Bucles")
    print("   - Java: for tradicional con índices; for-each.")
    print("   - Python: for itera sobre elementos; range(n) para secuencias numéricas.")

    print("\n7) Funciones y sobrecarga")
    print("   - Java: sobrecarga por firma.")
    print("   - Python: no hay sobrecarga; usa parámetros por defecto y argumentos con nombre.")

    print("\n8) Excepciones")
    print("   - Java: checked y unchecked exceptions.")
    print("   - Python: no hay checked exceptions; try/except para capturar específicas si hace falta.")

    print("\n9) POO y self")
    print("   - Java: this.")
    print("   - Python: self como primer parámetro de métodos de instancia (no es palabra reservada, es convención).")

    print("\n10) Estilo y convención")
    print("   - Java: camelCase, PascalCase.")
    print("   - Python: PEP8, snake_case para variables/funciones y MAYÚSCULAS para constantes.")

    print("\n11) Paquetes y módulos")
    print("   - Java: packages con directorio y clases.")
    print("   - Python: paquetes = carpetas con __init__.py; módulos = archivos .py.")

    print("\n12) Comparaciones: == vs is")
    print("   - Java: '==' vs equals().")
    print("   - Python: '==' compara valores; 'is' compara identidad de objeto.")
