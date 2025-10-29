"""
Ejemplos de exportación de datos a CSV usando pandas
Clase 4: pandas
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def ejemplo_basico():
    """Ejemplo básico de exportación a CSV"""
    print("=== EJEMPLO BÁSICO ===")

    # Crear DataFrame de ejemplo
    df = pd.DataFrame({
        'nombre': ['Ana García', 'Carlos López', 'María Rodríguez', 'Juan Martín'],
        'edad': [25, 30, 28, 35],
        'salario': [35000, 42000, 38000, 45000],
        'departamento': ['Ventas', 'IT', 'Marketing', 'IT']
    })

    print("DataFrame original:")
    print(df)

    # Exportar a CSV básico
    df.to_csv('empleados.csv')
    print("\n✓ Archivo 'empleados.csv' creado con configuración por defecto")

    # Exportar sin índice
    df.to_csv('empleados_sin_indice.csv', index=False)
    print("✓ Archivo 'empleados_sin_indice.csv' creado sin índice")


def ejemplo_configuracion_avanzada():
    """Ejemplos con diferentes configuraciones de exportación"""
    print("\n=== CONFIGURACIÓN AVANZADA ===")

    # Crear DataFrame con más datos
    df = pd.DataFrame({
        'producto': ['Laptop HP', 'Mouse Logitech', 'Teclado Mecánico', 'Monitor LG', 'Webcam'],
        'precio': [899.99, 25.50, 120.00, 299.99, 79.99],
        'stock': [15, 50, 30, 10, 25],
        'categoria': ['Computadoras', 'Accesorios', 'Accesorios', 'Monitores', 'Accesorios']
    })

    # 1. Cambiar separador a punto y coma
    df.to_csv('productos_semicolon.csv', sep=';', index=False)
    print("✓ Exportado con punto y coma como separador")

    # 2. Exportar solo columnas específicas
    df.to_csv('productos_basico.csv', columns=['producto', 'precio'], index=False)
    print("✓ Exportado solo columnas 'producto' y 'precio'")

    # 3. Cambiar nombres de columnas al exportar
    df.to_csv('productos_ingles.csv',
              header=['Product', 'Price', 'Stock', 'Category'],
              index=False)
    print("✓ Exportado con nombres de columnas en inglés")

    # 4. Usar tabulador como separador (útil para Excel)
    df.to_csv('productos_tab.tsv', sep='\t', index=False)
    print("✓ Exportado como TSV (separado por tabuladores)")


def ejemplo_con_fechas_y_nulos():
    """Ejemplo con manejo de fechas y valores nulos"""
    print("\n=== FECHAS Y VALORES NULOS ===")

    # Crear datos con fechas y valores nulos
    fechas_base = datetime.now()
    df = pd.DataFrame({
        'fecha': [fechas_base + timedelta(days=i) for i in range(5)],
        'ventas': [100, np.nan, 150, 200, np.nan],
        'ciudad': ['Madrid', 'Barcelona', None, 'Valencia', 'Sevilla'],
        'comision': [10.5, np.nan, 15.75, 20.0, np.nan]
    })

    print("DataFrame con fechas y valores nulos:")
    print(df)

    # Exportar con formato de fecha personalizado
    df['fecha'] = df['fecha'].dt.strftime('%d/%m/%Y')
    df.to_csv('ventas_fechas.csv', index=False, na_rep='N/A')
    print("\n✓ Exportado con fechas formateadas y nulos como 'N/A'")

    # Exportar con nulos como cadena vacía
    df.to_csv('ventas_sin_na.csv', index=False, na_rep='')
    print("✓ Exportado con nulos como cadenas vacías")


def ejemplo_datos_grandes():
    """Ejemplo con dataset más grande y opciones de compresión"""
    print("\n=== DATOS GRANDES Y COMPRESIÓN ===")

    # Crear un DataFrame grande
    np.random.seed(42)
    n_filas = 10000

    df_grande = pd.DataFrame({
        'id': range(1, n_filas + 1),
        'valor_a': np.random.randn(n_filas),
        'valor_b': np.random.randint(0, 100, n_filas),
        'categoria': np.random.choice(['A', 'B', 'C', 'D'], n_filas),
        'activo': np.random.choice([True, False], n_filas)
    })

    print(f"DataFrame grande creado: {len(df_grande)} filas")

    # Exportar normal
    df_grande.to_csv('datos_grandes.csv', index=False)
    print("✓ Archivo CSV normal creado")

    # Exportar comprimido
    df_grande.to_csv('datos_grandes.csv.gz', index=False, compression='gzip')
    print("✓ Archivo CSV comprimido (gzip) creado")

    # Mostrar tamaños
    import os
    size_normal = os.path.getsize('datos_grandes.csv') / 1024
    size_comprimido = os.path.getsize('datos_grandes.csv.gz') / 1024
    print(f"\nTamaño archivo normal: {size_normal:.2f} KB")
    print(f"Tamaño archivo comprimido: {size_comprimido:.2f} KB")
    print(f"Reducción: {((1 - size_comprimido/size_normal) * 100):.1f}%")


def ejemplo_append_csv():
    """Ejemplo de cómo añadir datos a un CSV existente"""
    print("\n=== AÑADIR DATOS A CSV EXISTENTE ===")

    # Primer DataFrame
    df1 = pd.DataFrame({
        'mes': ['Enero', 'Febrero'],
        'ventas': [10000, 12000],
        'gastos': [8000, 9000]
    })

    # Crear archivo inicial
    df1.to_csv('finanzas.csv', index=False)
    print("✓ Archivo inicial creado con datos de Enero y Febrero")

    # Segundo DataFrame con más datos
    df2 = pd.DataFrame({
        'mes': ['Marzo', 'Abril'],
        'ventas': [15000, 14000],
        'gastos': [10000, 9500]
    })

    # Añadir al archivo existente
    df2.to_csv('finanzas.csv', mode='a', header=False, index=False)
    print("✓ Datos de Marzo y Abril añadidos al archivo")

    # Verificar resultado
    df_completo = pd.read_csv('finanzas.csv')
    print("\nContenido final del archivo:")
    print(df_completo)


def ejemplo_multiples_hojas_csv():
    """Ejemplo de exportar múltiples DataFrames relacionados"""
    print("\n=== EXPORTAR MÚLTIPLES DATAFRAMES ===")

    # Crear varios DataFrames relacionados
    clientes = pd.DataFrame({
        'id_cliente': [1, 2, 3],
        'nombre': ['Empresa A', 'Empresa B', 'Empresa C'],
        'ciudad': ['Madrid', 'Barcelona', 'Valencia']
    })

    pedidos = pd.DataFrame({
        'id_pedido': [101, 102, 103, 104],
        'id_cliente': [1, 2, 1, 3],
        'producto': ['Laptop', 'Mouse', 'Teclado', 'Monitor'],
        'cantidad': [2, 10, 5, 3],
        'precio_unitario': [899.99, 25.50, 120.00, 299.99]
    })

    # Calcular total
    pedidos['total'] = pedidos['cantidad'] * pedidos['precio_unitario']

    # Exportar cada DataFrame a su propio CSV
    clientes.to_csv('bd_clientes.csv', index=False)
    pedidos.to_csv('bd_pedidos.csv', index=False)

    print("✓ Exportados archivos:")
    print("  - bd_clientes.csv")
    print("  - bd_pedidos.csv")

    # Crear un resumen combinado
    resumen = pd.merge(pedidos, clientes, on='id_cliente')
    resumen = resumen[['nombre', 'producto', 'cantidad', 'total', 'ciudad']]
    resumen.to_csv('resumen_ventas.csv', index=False)
    print("  - resumen_ventas.csv (datos combinados)")


def ejemplo_formato_especifico():
    """Ejemplo con formato específico para diferentes sistemas"""
    print("\n=== FORMATOS ESPECÍFICOS ===")

    df = pd.DataFrame({
        'nombre': ['José María', 'Ángel Luis', 'Pilar'],
        'puntuación': [8.5, 9.25, 7.75],
        'aprobado': [True, True, False]
    })

    # Para Excel en español (encoding latin-1, separador ;)
    df.to_csv('para_excel_es.csv', sep=';', encoding='latin-1',
              index=False, decimal=',')
    print("✓ Formato para Excel español (punto y coma, coma decimal)")

    # Para sistemas en inglés
    df.to_csv('para_sistemas_en.csv', sep=',', encoding='utf-8',
              index=False, decimal='.')
    print("✓ Formato estándar internacional (coma, punto decimal)")

    # Para bases de datos (con comillas en strings)
    df.to_csv('para_bd.csv', index=False, quoting=1)  # QUOTE_ALL
    print("✓ Formato para importar en BD (todos los campos con comillas)")


def limpiar_archivos_ejemplo():
    """Función para limpiar los archivos de ejemplo creados"""
    import os
    import glob

    print("\n=== LIMPIANDO ARCHIVOS DE EJEMPLO ===")

    archivos_csv = glob.glob('*.csv') + glob.glob('*.tsv') + glob.glob('*.csv.gz')

    for archivo in archivos_csv:
        try:
            os.remove(archivo)
            print(f"✓ Eliminado: {archivo}")
        except:
            pass


if __name__ == "__main__":
    print("EJEMPLOS DE EXPORTACIÓN A CSV CON PANDAS")
    print("=" * 50)

    # Ejecutar todos los ejemplos
    ejemplo_basico()
    ejemplo_configuracion_avanzada()
    ejemplo_con_fechas_y_nulos()
    ejemplo_datos_grandes()
    ejemplo_append_csv()
    ejemplo_multiples_hojas_csv()
    ejemplo_formato_especifico()

    print("\n" + "=" * 50)
    print("Todos los ejemplos completados.")
    print("Se han creado varios archivos CSV de ejemplo en el directorio actual.")

    # Preguntar si limpiar archivos
    respuesta = input("\n¿Deseas eliminar los archivos de ejemplo creados? (s/n): ")
    if respuesta.lower() == 's':
        limpiar_archivos_ejemplo()
        print("Archivos de ejemplo eliminados.")