# -*- coding: utf-8 -*-


# ══════════════════════════════════════════════════════════════════════════════
# rankear_clientes_ventana_movil.py  —  ARCHIVO COMPLETO PARA GOOGLE COLAB
#
# Estructura:
#   1. Función solucionadora   ← lo que el validador externo usa como answer
#   2. Generador de casos      ← el original del enunciado, con _resolver_ interno
#   3. Pruebas                 ← para verificar en Colab
# ══════════════════════════════════════════════════════════════════════════════

import random
import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# 1. FUNCIÓN SOLUCIONADORA
# ══════════════════════════════════════════════════════════════════════════════

def rankear_clientes_ventana_movil(
    df: pd.DataFrame,
    cliente_col: str,
    fecha_col: str,
    monto_col: str,
    ventana: int
) -> pd.DataFrame:
    """
    Para un DataFrame de movimientos de clientes, calcula qué cliente
    lidera la suma móvil de montos en cada fecha.

    Parámetros
    ----------
    df          : DataFrame con columnas de cliente, fecha y monto.
    cliente_col : Nombre de la columna de clientes.
    fecha_col   : Nombre de la columna de fechas (acepta texto o datetime).
    monto_col   : Nombre de la columna de montos.
    ventana     : Tamaño de la ventana para la suma móvil (rolling).

    Proceso
    -------
    1. Convierte fecha_col a datetime (errors='coerce').
    2. Elimina filas con fecha inválida o NaN en cliente_col / monto_col.
    3. Agrega por (cliente, fecha) sumando los montos del mismo día.
    4. Ordena por cliente y fecha.
    5. Calcula suma móvil con ventana=ventana por cliente (min_periods=1).
    6. Por cada fecha, selecciona el cliente con mayor suma móvil.
    7. En caso de empate, elige el de menor orden alfabético.

    Retorna
    -------
    pd.DataFrame con columnas [fecha_col, cliente_col, 'suma_movil'],
    ordenado por fecha_col ascendente e índice reiniciado.
    Si no quedan datos válidos, devuelve un DataFrame vacío con esas columnas.
    """
    # 1. Copia + convertir fechas
    trabajo = df.copy()
    trabajo[fecha_col] = pd.to_datetime(trabajo[fecha_col], errors="coerce")

    # 2. Eliminar filas inválidas
    trabajo = trabajo.dropna(subset=[cliente_col, fecha_col, monto_col])

    if trabajo.empty:
        return pd.DataFrame(columns=[fecha_col, cliente_col, "suma_movil"])

    # 3 & 4. Agregar por (cliente, fecha) y ordenar
    agregado = (
        trabajo
        .groupby([cliente_col, fecha_col], as_index=False)[monto_col]
        .sum()
        .sort_values([cliente_col, fecha_col])
        .reset_index(drop=True)
    )

    # 5. Suma móvil por cliente
    agregado["suma_movil"] = agregado.groupby(cliente_col)[monto_col].transform(
        lambda serie: serie.rolling(window=ventana, min_periods=1).sum()
    )

    # 6 & 7. Ranking por fecha: mayor suma_movil, empate → orden alfabético
    ranking = agregado.sort_values(
        [fecha_col, "suma_movil", cliente_col],
        ascending=[True, False, True],
    )

    top_por_fecha = (
        ranking
        .groupby(fecha_col, as_index=False)
        .first()[[fecha_col, cliente_col, "suma_movil"]]
        .sort_values(fecha_col)
        .reset_index(drop=True)
    )

    top_por_fecha["suma_movil"] = top_por_fecha["suma_movil"].astype(float)
    return top_por_fecha


# ══════════════════════════════════════════════════════════════════════════════
# 2. GENERADOR DE CASOS DE USO
#    ─ Usa _resolver_ interno (función privada) para NO depender de
#      rankear_clientes_ventana_movil al ejecutarse como módulo aislado.
#    ─ Esto es exactamente el generador original del enunciado.
# ══════════════════════════════════════════════════════════════════════════════

def _resolver_rankear_clientes_ventana_movil(
    df, cliente_col, fecha_col, monto_col, ventana
):
    """Copia interna del resolver, usada solo por el generador."""
    trabajo = df.copy()
    trabajo[fecha_col] = pd.to_datetime(trabajo[fecha_col], errors="coerce")
    trabajo = trabajo.dropna(subset=[cliente_col, fecha_col, monto_col])

    if trabajo.empty:
        return pd.DataFrame(columns=[fecha_col, cliente_col, "suma_movil"])

    agregado = (
        trabajo.groupby([cliente_col, fecha_col], as_index=False)[monto_col]
        .sum()
        .sort_values([cliente_col, fecha_col])
        .reset_index(drop=True)
    )

    agregado["suma_movil"] = agregado.groupby(cliente_col)[monto_col].transform(
        lambda serie: serie.rolling(window=ventana, min_periods=1).sum()
    )

    ranking = agregado.sort_values(
        [fecha_col, "suma_movil", cliente_col],
        ascending=[True, False, True],
    )

    top_por_fecha = (
        ranking.groupby(fecha_col, as_index=False)
        .first()[[fecha_col, cliente_col, "suma_movil"]]
        .sort_values(fecha_col)
        .reset_index(drop=True)
    )

    top_por_fecha["suma_movil"] = top_por_fecha["suma_movil"].astype(float)
    return top_por_fecha


def generar_caso_de_uso_rankear_clientes_ventana_movil():
    cliente_col = "cliente"
    fecha_col   = "fecha"
    monto_col   = "monto"

    clientes   = [f"C{i:02d}" for i in range(1, random.randint(4, 7))]
    ventana    = random.randint(2, 5)
    base       = pd.Timestamp("2022-01-01") + pd.Timedelta(days=random.randint(0, 600))
    rango_dias = random.randint(10, 24)
    n_filas    = random.randint(70, 150)

    filas = []
    for _ in range(n_filas):
        fecha   = ("no_fecha" if random.random() < 0.10
                   else (base + pd.Timedelta(days=random.randint(0, rango_dias - 1))
                         ).strftime("%Y-%m-%d"))
        cliente = None if random.random() < 0.07 else random.choice(clientes)
        monto   = (np.nan if random.random() < 0.10
                   else round(float(np.random.normal(loc=90.0, scale=40.0)), 2))
        filas.append({cliente_col: cliente, fecha_col: fecha, monto_col: monto})

    df = pd.DataFrame(filas)

    input_data = {
        "df": df.copy(), "cliente_col": cliente_col,
        "fecha_col": fecha_col, "monto_col": monto_col, "ventana": ventana,
    }
    # Usa _resolver_ interno → sin dependencia externa
    output_data = _resolver_rankear_clientes_ventana_movil(
        df, cliente_col, fecha_col, monto_col, ventana
    )
    return input_data, output_data


# ══════════════════════════════════════════════════════════════════════════════
# 3. PRUEBAS (ejecutar en Colab con: python rankear_clientes_ventana_movil.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Test 1: caso de uso aleatorio ────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  TEST 1 — CASO DE USO ALEATORIO")
    print("═" * 65)

    caso_input, caso_output = generar_caso_de_uso_rankear_clientes_ventana_movil()
    resultado = rankear_clientes_ventana_movil(**caso_input)

    cc = caso_input['cliente_col']
    fc = caso_input['fecha_col']

    comp = caso_output.rename(columns={cc: 'esp_cliente', 'suma_movil': 'esp_suma'}).copy()
    comp['obt_cliente'] = resultado[cc].values
    comp['obt_suma']    = resultado['suma_movil'].values
    comp['ok']          = ((comp['esp_cliente'] == comp['obt_cliente']) &
                           np.isclose(comp['esp_suma'], comp['obt_suma']))

    print(f"  ventana={caso_input['ventana']}  |  filas entrada={len(caso_input['df'])}  |  fechas resultado={len(resultado)}")
    print("\n  Primeras 10 fechas:")
    print(comp[[fc, 'esp_cliente', 'obt_cliente', 'esp_suma', 'obt_suma', 'ok']]
          .head(10).to_string(index=False))
    todo_ok = comp['ok'].all()
    print(f"\n  ✓ Resultado idéntico al esperado: {todo_ok}")
    print("═" * 65)
