"""
Ejemplos de exportación de datos a CSV usando Python estándar
Clase 4: Alternativa a pandas usando el módulo csv
"""

import csv
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any


def ejemplo_basico_writer():
    """Ejemplo básico usando csv.writer"""
    print("=== EJEMPLO BÁSICO CON csv.writer ===")

    # Datos a exportar (lista de listas)
    datos = [
        ['nombre', 'edad', 'ciudad', 'profesion'],
        ['Ana García', 25, 'Madrid', 'Ingeniera'],
        ['Carlos López', 30, 'Barcelona', 'Médico'],
        ['María Rodríguez', 28, 'Valencia', 'Profesora'],
        ['Juan Martín', 35, 'Sevilla', 'Arquitecto']
    ]

    # Escribir archivo CSV
    with open('personas.csv', 'w', newline='', encoding='utf-8') as archivo:
        escritor = csv.writer(archivo)
        escritor.writerows(datos)

    print("✓ Archivo 'personas.csv' creado")

    # Verificar contenido
    print("\nContenido del archivo:")
    with open('personas.csv', 'r', encoding='utf-8') as archivo:
        print(archivo.read())


def ejemplo_dict_writer():
    """Ejemplo usando csv.DictWriter para datos estructurados"""
    print("\n=== EJEMPLO CON csv.DictWriter ===")

    # Datos como lista de diccionarios
    productos = [
        {'producto': 'Laptop HP', 'precio': 899.99, 'stock': 15, 'categoria': 'Computadoras'},
        {'producto': 'Mouse Logitech', 'precio': 25.50, 'stock': 50, 'categoria': 'Accesorios'},
        {'producto': 'Teclado Mecánico', 'precio': 120.00, 'stock': 30, 'categoria': 'Accesorios'},
        {'producto': 'Monitor LG', 'precio': 299.99, 'stock': 10, 'categoria': 'Monitores'}
    ]

    # Definir el orden de las columnas
    columnas = ['producto', 'precio', 'stock', 'categoria']

    # Escribir CSV usando DictWriter
    with open('inventario.csv', 'w', newline='', encoding='utf-8') as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=columnas)

        # Escribir encabezados
        escritor.writeheader()

        # Escribir datos
        escritor.writerows(productos)

    print("✓ Archivo 'inventario.csv' creado con DictWriter")
    print(f"  Columnas: {', '.join(columnas)}")


def ejemplo_separadores_personalizados():
    """Ejemplo con diferentes separadores y formatos"""
    print("\n=== SEPARADORES PERSONALIZADOS ===")

    datos = [
        ['fecha', 'temperatura', 'humedad', 'presion'],
        ['2024-01-01', 22.5, 65, 1013.25],
        ['2024-01-02', 23.1, 62, 1014.10],
        ['2024-01-03', 21.8, 68, 1012.50]
    ]

    # 1. Separador punto y coma
    with open('datos_clima_semicolon.csv', 'w', newline='', encoding='utf-8') as archivo:
        escritor = csv.writer(archivo, delimiter=';')
        escritor.writerows(datos)
    print("✓ Creado con punto y coma como separador")

    # 2. Separador tabulador (TSV)
    with open('datos_clima.tsv', 'w', newline='', encoding='utf-8') as archivo:
        escritor = csv.writer(archivo, delimiter='\t')
        escritor.writerows(datos)
    print("✓ Creado como TSV (separado por tabuladores)")

    # 3. Separador pipe
    with open('datos_clima_pipe.csv', 'w', newline='', encoding='utf-8') as archivo:
        escritor = csv.writer(archivo, delimiter='|')
        escritor.writerows(datos)
    print("✓ Creado con pipe (|) como separador")


def ejemplo_manejo_comillas():
    """Ejemplo de manejo de comillas y caracteres especiales"""
    print("\n=== MANEJO DE COMILLAS Y CARACTERES ESPECIALES ===")

    # Datos con caracteres especiales
    datos_especiales = [
        ['titulo', 'descripcion', 'precio'],
        ['Monitor 27"', 'Pantalla LED de 27 pulgadas', 299.99],
        ['Libro "Python"', 'Guía completa, incluye CD', 45.50],
        ['Cable USB-C', 'Cable de 2 metros, alta velocidad', 15.99]
    ]

    # Configurar el manejo de comillas
    with open('productos_especiales.csv', 'w', newline='', encoding='utf-8') as archivo:
        escritor = csv.writer(archivo, quoting=csv.QUOTE_MINIMAL)
        escritor.writerows(datos_especiales)

    print("✓ Archivo con caracteres especiales creado (QUOTE_MINIMAL)")

    # Forzar comillas en todos los campos
    with open('productos_comillas.csv', 'w', newline='', encoding='utf-8') as archivo:
        escritor = csv.writer(archivo, quoting=csv.QUOTE_ALL)
        escritor.writerows(datos_especiales)

    print("✓ Archivo con todas las comillas creado (QUOTE_ALL)")


def ejemplo_append_datos():
    """Ejemplo de cómo añadir datos a un archivo CSV existente"""
    print("\n=== AÑADIR DATOS A CSV EXISTENTE ===")

    # Crear archivo inicial
    encabezados = ['mes', 'ventas', 'gastos', 'beneficio']
    datos_iniciales = [
        ['Enero', 50000, 35000, 15000],
        ['Febrero', 52000, 36000, 16000]
    ]

    with open('finanzas_mensual.csv', 'w', newline='', encoding='utf-8') as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(encabezados)
        escritor.writerows(datos_iniciales)

    print("✓ Archivo inicial creado con datos de Enero y Febrero")

    # Añadir más datos
    datos_nuevos = [
        ['Marzo', 55000, 37000, 18000],
        ['Abril', 53000, 36500, 16500]
    ]

    with open('finanzas_mensual.csv', 'a', newline='', encoding='utf-8') as archivo:
        escritor = csv.writer(archivo)
        escritor.writerows(datos_nuevos)

    print("✓ Datos de Marzo y Abril añadidos")

    # Verificar resultado
    print("\nContenido final:")
    with open('finanzas_mensual.csv', 'r', encoding='utf-8') as archivo:
        lector = csv.reader(archivo)
        for fila in lector:
            print(f"  {fila}")


def ejemplo_dialecto_personalizado():
    """Ejemplo creando un dialecto CSV personalizado"""
    print("\n=== DIALECTO PERSONALIZADO ===")

    # Registrar un dialecto personalizado
    csv.register_dialect('mi_formato',
                        delimiter=';',
                        quotechar='"',
                        doublequote=True,
                        skipinitialspace=True,
                        lineterminator='\r\n',
                        quoting=csv.QUOTE_MINIMAL)

    datos = [
        ['id', 'nombre', 'email', 'activo'],
        [1, 'Usuario Uno', 'usuario1@email.com', 'Sí'],
        [2, 'Usuario Dos', 'usuario2@email.com', 'No'],
        [3, 'Usuario Tres', 'usuario3@email.com', 'Sí']
    ]

    # Usar el dialecto personalizado
    with open('usuarios_custom.csv', 'w', newline='', encoding='utf-8') as archivo:
        escritor = csv.writer(archivo, dialect='mi_formato')
        escritor.writerows(datos)

    print("✓ Archivo creado con dialecto personalizado 'mi_formato'")
    print("  Características: punto y coma, comillas dobles, salto de línea Windows")


def ejemplo_procesamiento_linea_por_linea():
    """Ejemplo de procesamiento línea por línea (útil para archivos grandes)"""
    print("\n=== PROCESAMIENTO LÍNEA POR LÍNEA ===")

    # Simular procesamiento de datos grandes
    with open('ventas_diarias.csv', 'w', newline='', encoding='utf-8') as archivo:
        escritor = csv.writer(archivo)

        # Escribir encabezados
        escritor.writerow(['dia', 'producto', 'cantidad', 'precio_unitario', 'total'])

        # Procesar y escribir datos línea por línea
        fecha_base = datetime.now()
        productos = ['Laptop', 'Mouse', 'Teclado', 'Monitor', 'Webcam']
        precios = [899.99, 25.50, 120.00, 299.99, 79.99]

        for dia in range(30):  # 30 días de datos
            fecha = fecha_base - timedelta(days=30-dia)
            for i, producto in enumerate(productos):
                cantidad = (dia % 5) + 1  # Cantidad variable
                precio = precios[i]
                total = cantidad * precio

                # Escribir una línea a la vez
                escritor.writerow([
                    fecha.strftime('%Y-%m-%d'),
                    producto,
                    cantidad,
                    precio,
                    f'{total:.2f}'
                ])

    print("✓ Archivo de ventas diarias creado (150 registros)")
    print("  Procesado línea por línea para simular datos grandes")


def ejemplo_con_validacion():
    """Ejemplo con validación de datos antes de exportar"""
    print("\n=== EXPORTACIÓN CON VALIDACIÓN ===")

    def validar_email(email: str) -> bool:
        """Validación simple de email"""
        return '@' in email and '.' in email.split('@')[1]

    def validar_edad(edad: Any) -> bool:
        """Validar que la edad sea un número positivo"""
        try:
            return int(edad) > 0 and int(edad) < 150
        except:
            return False

    # Datos a validar y exportar
    usuarios_raw = [
        {'nombre': 'Ana', 'edad': 25, 'email': 'ana@email.com'},
        {'nombre': 'Carlos', 'edad': -5, 'email': 'carlos@email.com'},  # Edad inválida
        {'nombre': 'María', 'edad': 28, 'email': 'maria_sin_arroba'},  # Email inválido
        {'nombre': 'Juan', 'edad': 35, 'email': 'juan@email.com'},
        {'nombre': 'Pedro', 'edad': 'treinta', 'email': 'pedro@email.com'}  # Edad inválida
    ]

    # Archivos para datos válidos e inválidos
    with open('usuarios_validos.csv', 'w', newline='', encoding='utf-8') as archivo_validos, \
         open('usuarios_errores.csv', 'w', newline='', encoding='utf-8') as archivo_errores:

        columnas = ['nombre', 'edad', 'email']
        escritor_validos = csv.DictWriter(archivo_validos, fieldnames=columnas)
        escritor_errores = csv.DictWriter(archivo_errores,
                                         fieldnames=columnas + ['error'])

        # Escribir encabezados
        escritor_validos.writeheader()
        escritor_errores.writeheader()

        # Procesar y validar cada registro
        validos = 0
        errores = 0

        for usuario in usuarios_raw:
            error_msgs = []

            if not validar_edad(usuario['edad']):
                error_msgs.append('Edad inválida')
            if not validar_email(usuario['email']):
                error_msgs.append('Email inválido')

            if error_msgs:
                usuario['error'] = '; '.join(error_msgs)
                escritor_errores.writerow(usuario)
                errores += 1
            else:
                escritor_validos.writerow(usuario)
                validos += 1

    print(f"✓ Procesados {len(usuarios_raw)} registros:")
    print(f"  - {validos} válidos → usuarios_validos.csv")
    print(f"  - {errores} con errores → usuarios_errores.csv")


def ejemplo_clase_exportadora():
    """Ejemplo de clase reutilizable para exportar CSV"""
    print("\n=== CLASE EXPORTADORA CSV ===")

    class ExportadorCSV:
        """Clase para manejar exportaciones CSV de forma consistente"""

        def __init__(self, encoding='utf-8', delimiter=','):
            self.encoding = encoding
            self.delimiter = delimiter
            self.archivos_creados = []

        def exportar_lista(self, archivo: str, datos: List[List], encabezados: List[str] = None):
            """Exportar lista de listas a CSV"""
            with open(archivo, 'w', newline='', encoding=self.encoding) as f:
                escritor = csv.writer(f, delimiter=self.delimiter)
                if encabezados:
                    escritor.writerow(encabezados)
                escritor.writerows(datos)
            self.archivos_creados.append(archivo)
            return archivo

        def exportar_diccionarios(self, archivo: str, datos: List[Dict], columnas: List[str] = None):
            """Exportar lista de diccionarios a CSV"""
            if not datos:
                return None

            if not columnas:
                columnas = list(datos[0].keys())

            with open(archivo, 'w', newline='', encoding=self.encoding) as f:
                escritor = csv.DictWriter(f, fieldnames=columnas, delimiter=self.delimiter)
                escritor.writeheader()
                escritor.writerows(datos)

            self.archivos_creados.append(archivo)
            return archivo

        def obtener_estadisticas(self):
            """Obtener estadísticas de archivos creados"""
            stats = []
            for archivo in self.archivos_creados:
                if os.path.exists(archivo):
                    size = os.path.getsize(archivo)
                    with open(archivo, 'r', encoding=self.encoding) as f:
                        lineas = sum(1 for _ in f)
                    stats.append({
                        'archivo': archivo,
                        'tamaño_bytes': size,
                        'lineas': lineas
                    })
            return stats

    # Usar la clase
    exportador = ExportadorCSV()

    # Exportar diferentes tipos de datos
    datos_lista = [
        [1, 'Producto A', 100],
        [2, 'Producto B', 200],
        [3, 'Producto C', 150]
    ]
    exportador.exportar_lista('productos_clase.csv', datos_lista,
                             encabezados=['id', 'nombre', 'precio'])

    datos_dict = [
        {'ciudad': 'Madrid', 'poblacion': 3223000, 'pais': 'España'},
        {'ciudad': 'Barcelona', 'poblacion': 1620000, 'pais': 'España'},
        {'ciudad': 'Valencia', 'poblacion': 791000, 'pais': 'España'}
    ]
    exportador.exportar_diccionarios('ciudades_clase.csv', datos_dict)

    # Mostrar estadísticas
    print("✓ Archivos creados con la clase ExportadorCSV:")
    for stat in exportador.obtener_estadisticas():
        print(f"  - {stat['archivo']}: {stat['lineas']} líneas, {stat['tamaño_bytes']} bytes")


def limpiar_archivos_ejemplo():
    """Función para limpiar los archivos de ejemplo creados"""
    import glob

    print("\n=== LIMPIANDO ARCHIVOS DE EJEMPLO ===")

    archivos = glob.glob('*.csv') + glob.glob('*.tsv')

    for archivo in archivos:
        try:
            os.remove(archivo)
            print(f"✓ Eliminado: {archivo}")
        except:
            pass


if __name__ == "__main__":
    print("EJEMPLOS DE EXPORTACIÓN A CSV CON PYTHON ESTÁNDAR")
    print("=" * 60)

    # Ejecutar todos los ejemplos
    ejemplo_basico_writer()
    ejemplo_dict_writer()
    ejemplo_separadores_personalizados()
    ejemplo_manejo_comillas()
    ejemplo_append_datos()
    ejemplo_dialecto_personalizado()
    ejemplo_procesamiento_linea_por_linea()
    ejemplo_con_validacion()
    ejemplo_clase_exportadora()

    print("\n" + "=" * 60)
    print("Todos los ejemplos completados.")
    print("Se han creado varios archivos CSV de ejemplo en el directorio actual.")

    # Preguntar si limpiar archivos
    respuesta = input("\n¿Deseas eliminar los archivos de ejemplo creados? (s/n): ")
    if respuesta.lower() == 's':
        limpiar_archivos_ejemplo()
        print("Archivos de ejemplo eliminados.")