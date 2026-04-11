import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Teoría de los Efectos Olvidados",
    layout="wide",
    page_icon="📊"
)

# ============================================================
# FUNCIONES MATEMÁTICAS
# ============================================================
def max_min_composition(matrix1, matrix2):
    rows1, cols1 = matrix1.shape
    rows2, cols2 = matrix2.shape
    if cols1 != rows2:
        return None
    result = np.zeros((rows1, cols2))
    for i in range(rows1):
        for k in range(cols2):
            max_val = 0
            for j in range(cols1):
                min_val = min(matrix1[i, j], matrix2[j, k])
                if min_val > max_val:
                    max_val = min_val
            result[i, k] = max_val
    return result

def subtract_matrices(m2, m1):
    return np.maximum(0, m2 - m1)

def load_and_clean_csv(uploaded_file):
    df = pd.read_csv(uploaded_file, index_col=0)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0).clip(0, 1)
    return df

# ============================================================
# FUNCIONES DE VISUALIZACIÓN
# ============================================================
def draw_incidence_graph(matrix, source_labels, target_labels, title):
    G = nx.DiGraph()
    source_nodes = [f"S_{l}" for l in source_labels]
    target_nodes = [f"T_{l}" for l in target_labels]
    G.add_nodes_from(source_nodes, bipartite=0)
    G.add_nodes_from(target_nodes, bipartite=1)
    pos = {}
    pos.update((n, (1, -i)) for i, n in enumerate(source_nodes))
    pos.update((n, (2.5, -i)) for i, n in enumerate(target_nodes))
    edge_labels = {}
    for i, s in enumerate(source_labels):
        for j, t in enumerate(target_labels):
            w = matrix[i, j]
            if w > 0:
                G.add_edge(source_nodes[i], target_nodes[j], weight=w)
                edge_labels[(source_nodes[i], target_nodes[j])] = f"{w:.3f}"
    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, nodelist=source_nodes, node_color='#87CEEB', node_size=1200, edgecolors='black', ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=target_nodes, node_color='#98FB98', node_size=1200, edgecolors='black', ax=ax)
    labels_map = {n: n.split('_', 1)[1] for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels_map, font_size=10, ax=ax)
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color='gray', ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=9, ax=ax)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')
    return fig

def draw_focused_graph(center_node, connected_nodes, weights, title, mode='source'):
    G = nx.DiGraph()
    G.add_node(center_node, layer=0)
    edge_labels = {}
    for node, w in zip(connected_nodes, weights):
        G.add_node(node, layer=1)
        if mode == 'source':
            G.add_edge(center_node, node, weight=w)
            edge_labels[(center_node, node)] = f"{w:.3f}"
        else:
            G.add_edge(node, center_node, weight=w)
            edge_labels[(node, center_node)] = f"{w:.3f}"
    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='#FFFACD', edgecolors='black', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=25, edge_color='gray', ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    return fig

# ============================================================
# WIDGET REUTILIZABLE: ENTRADA DE MATRIZ
# ============================================================
def matrix_input_widget(key_prefix, titulo, row_hint="Causas (A)", col_hint="Efectos (B)"):
    """
    Devuelve un DataFrame con la matriz ingresada, o None si no hay datos.
    Soporta dos métodos: cargar CSV o ingresar manualmente.
    """
    metodo = st.radio(
        f"Método de entrada — {titulo}",
        ["📁 Cargar CSV", "✏️ Ingresar manualmente"],
        key=f"{key_prefix}_metodo",
        horizontal=True
    )

    if metodo == "📁 Cargar CSV":
        archivo = st.file_uploader(
            f"Archivo CSV para {titulo}",
            type=["csv"],
            key=f"{key_prefix}_csv",
            help="Primera columna = etiquetas de filas. Primera fila = etiquetas de columnas. Valores en [0, 1]."
        )
        if archivo:
            df = load_and_clean_csv(archivo)
            st.caption(f"✅ Cargada: {df.shape[0]} filas × {df.shape[1]} columnas")
            with st.expander("Vista previa", expanded=False):
                st.dataframe(df.style.format("{:.3f}"))
            return df
        return None

    else:  # Modo manual
        c1, c2 = st.columns(2)
        with c1:
            n_filas = st.number_input(
                f"Nº de {row_hint}", min_value=1, max_value=20, value=3,
                key=f"{key_prefix}_nfilas"
            )
        with c2:
            n_cols = st.number_input(
                f"Nº de {col_hint}", min_value=1, max_value=20, value=3,
                key=f"{key_prefix}_ncols"
            )

        # Etiquetas
        default_rows = ", ".join([f"a{i+1}" for i in range(int(n_filas))])
        default_cols = ", ".join([f"b{j+1}" for j in range(int(n_cols))])

        c1, c2 = st.columns(2)
        with c1:
            row_str = st.text_input(
                f"Etiquetas de {row_hint} (separadas por coma)",
                value=default_rows, key=f"{key_prefix}_rowlabels"
            )
        with c2:
            col_str = st.text_input(
                f"Etiquetas de {col_hint} (separadas por coma)",
                value=default_cols, key=f"{key_prefix}_collabels"
            )

        row_labels = [l.strip() for l in row_str.split(",")]
        col_labels = [l.strip() for l in col_str.split(",")]

        # Ajustar longitud si el usuario no puso suficientes etiquetas
        while len(row_labels) < int(n_filas):
            row_labels.append(f"f{len(row_labels)+1}")
        while len(col_labels) < int(n_cols):
            col_labels.append(f"c{len(col_labels)+1}")
        row_labels = row_labels[:int(n_filas)]
        col_labels = col_labels[:int(n_cols)]

        st.caption("Ingrese los valores de la matriz (entre 0.0 y 1.0). Puede editar cada celda directamente.")
        default_df = pd.DataFrame(
            np.zeros((int(n_filas), int(n_cols))),
            index=row_labels,
            columns=col_labels
        )
        edited_df = st.data_editor(
            default_df,
            key=f"{key_prefix}_editor",
            use_container_width=True
        )
        edited_df = edited_df.apply(pd.to_numeric, errors='coerce').fillna(0).clip(0, 1)
        return edited_df


# ============================================================
# MODO INSPECTOR (reutilizable)
# ============================================================
def inspector_focalizado(df_efectos, labels_causa, labels_efecto, key_prefix):
    st.divider()
    st.header("🕵️ Modo Inspector Focalizado")
    st.caption("Seleccione un elemento para analizar sus incidencias olvidadas entrantes y salientes.")

    tipo = st.radio(
        "Perspectiva de análisis:",
        ["Como CAUSA (efectos que genera)", "Como EFECTO (causas que lo determinan)"],
        horizontal=True,
        key=f"{key_prefix}_tipo_insp"
    )

    if "CAUSA" in tipo:
        el = st.selectbox("Seleccione el elemento causa:", ["-- Seleccione --"] + list(labels_causa), key=f"{key_prefix}_sel_causa")
        if el != "-- Seleccione --":
            fila = df_efectos.loc[el]
            df_foc = pd.DataFrame({'Efecto': labels_efecto, 'Incidencia': fila.values})
            df_foc = df_foc[df_foc['Incidencia'] > 0].sort_values('Incidencia', ascending=False)
            if not df_foc.empty:
                st.dataframe(df_foc.style.format({'Incidencia': "{:.3f}"}), hide_index=True)
                st.pyplot(draw_focused_graph(el, df_foc['Efecto'].tolist(), df_foc['Incidencia'].tolist(), f"Efectos olvidados de '{el}'", 'source'))
            else:
                st.info(f"'{el}' no genera efectos olvidados hacia ningún otro nodo.")
    else:
        el = st.selectbox("Seleccione el elemento efecto:", ["-- Seleccione --"] + list(labels_efecto), key=f"{key_prefix}_sel_efecto")
        if el != "-- Seleccione --":
            col = df_efectos[el]
            df_foc = pd.DataFrame({'Causa': labels_causa, 'Incidencia': col.values})
            df_foc = df_foc[df_foc['Incidencia'] > 0].sort_values('Incidencia', ascending=False)
            if not df_foc.empty:
                st.dataframe(df_foc.style.format({'Incidencia': "{:.3f}"}), hide_index=True)
                st.pyplot(draw_focused_graph(el, df_foc['Causa'].tolist(), df_foc['Incidencia'].tolist(), f"Causas olvidadas hacia '{el}'", 'target'))
            else:
                st.info(f"Ningún nodo genera efectos olvidados hacia '{el}'.")


# ============================================================
# ENCABEZADO PRINCIPAL
# ============================================================
st.title("🧩 Teoría de los Efectos Olvidados (TEO)")
st.markdown("Automatización de incidencias de orden superior y descubrimiento de efectos ocultos en sistemas complejos.")

# ============================================================
# SELECCIÓN DE MODO CON REGLAS VISIBLES
# ============================================================
st.subheader("Seleccione el modo de análisis")

MODOS = {
    "1. Sistema Cerrado  (A → A)": {
        "icono": "🔁",
        "descripcion": "Una sola matriz cuadrada donde los elementos del sistema se relacionan entre sí.",
        "reglas": [
            "✅ Requiere **1 matriz cuadrada** (n × n).",
            "✅ Calcula automáticamente **M² = M ∘ M** (efectos de 2° orden).",
            "✅ Produce la **Matriz de Efectos Olvidados M' = M² − M**.",
            "✅ Habilita el **Inspector Focalizado** sobre los efectos olvidados.",
            "⚠️ Los valores de la diagonal principal deben ser **1** (auto-incidencia).",
        ],
        "ejemplo": "Análisis de 12 sectores económicos de Kaufmann & Gil Aluja."
    },
    "2. Sistema Encadenado  (A → B → C)": {
        "icono": "🔗",
        "descripcion": "Dos matrices encadenadas. Las columnas de M1 deben coincidir con las filas de M2.",
        "reglas": [
            "✅ Requiere **2 matrices**: M1 (A → B) y M2 (B → C).",
            "✅ Calcula la composición **MAC = M1 ∘ M2**.",
            "⚙️ Para obtener Efectos Olvidados, suba o ingrese una **Matriz Directa A → C** (opcional).",
            "⚠️ Las columnas de M1 y las filas de M2 deben tener el **mismo número de elementos**.",
            "⚠️ M1 puede ser rectangular (m × n) y M2 rectangular (n × p).",
        ],
        "ejemplo": "Influencia de redes sociales (A) → Acciones del consumidor (B) → Resultados de negocio (C)."
    },
    "3. Cadena Compleja  (A → B → C → D)": {
        "icono": "⛓️",
        "descripcion": "Tres matrices en secuencia. La composición se aplica dos veces: (M1 ∘ M2) ∘ M3.",
        "reglas": [
            "✅ Requiere **3 matrices**: M1 (A→B), M2 (B→C), M3 (C→D).",
            "✅ Calcula **MAD = (M1 ∘ M2) ∘ M3** por asociatividad max-min.",
            "⚙️ Para Efectos Olvidados de 3ª generación, suba una **Matriz Directa A → D** (opcional).",
            "⚠️ Compatibilidad: cols(M1) = filas(M2) y cols(M2) = filas(M3).",
        ],
        "ejemplo": "Factores de salud (A) → Rendimiento laboral (B) → Hábitos intermedios (C) → Resultados (D)."
    },
}

modo = st.radio(
    "Modo:",
    list(MODOS.keys()),
    label_visibility="collapsed",
    horizontal=False
)

# Mostrar tarjeta informativa del modo seleccionado
info = MODOS[modo]
with st.expander(f"{info['icono']} Reglas y restricciones del modo seleccionado", expanded=True):
    st.markdown(f"**{info['descripcion']}**")
    st.markdown("**Ejemplo de uso:** " + info['ejemplo'])
    st.markdown("**Reglas:**")
    for r in info["reglas"]:
        st.markdown(f"  - {r}")

st.divider()

# ============================================================
# MODO 1: UNA MATRIZ (SISTEMA CERRADO)
# ============================================================
if "1." in modo:
    st.header("🔁 Análisis de Sistema Cerrado (A → A)")

    with st.container(border=True):
        st.markdown("**Paso 1 — Defina la Matriz M**")
        df = matrix_input_widget("m1_cerrado", "Matriz M (A → A)", row_hint="elementos (filas)", col_hint="elementos (columnas)")

    if df is not None:
        labels = list(df.index)
        matriz_np = df.values

        # Validar cuadrada
        if df.shape[0] != df.shape[1]:
            st.error(f"⚠️ La matriz debe ser **cuadrada**. Dimensión actual: {df.shape[0]} × {df.shape[1]}.")
        else:
            st.subheader("Matriz Original (M — Primer Orden)")
            st.dataframe(df.style.format("{:.3f}"), use_container_width=True)

            if st.button("🚀 Procesar Efectos Olvidados", type="primary", key="btn_cerrado"):
                st.session_state['procesado_cerrado'] = True

            if st.session_state.get('procesado_cerrado', False):
                m2 = max_min_composition(matriz_np, matriz_np)
                efectos = subtract_matrices(m2, matriz_np)

                df_m2 = pd.DataFrame(m2, index=labels, columns=labels)
                df_efectos = pd.DataFrame(efectos, index=labels, columns=labels)

                tab1, tab2 = st.tabs(["📊 Matriz de Segundo Orden (M²)", "🔍 Efectos Olvidados (M² − M)"])
                with tab1:
                    c1, c2 = st.columns([1, 1.5])
                    with c1:
                        st.dataframe(df_m2.style.format("{:.3f}"))
                    with c2:
                        st.pyplot(draw_incidence_graph(m2, labels, labels, "Red de Incidencias (2° Orden)"))
                with tab2:
                    c1, c2 = st.columns([1, 1.5])
                    with c1:
                        st.dataframe(df_efectos.style.format("{:.3f}"))
                    with c2:
                        st.pyplot(draw_incidence_graph(efectos, labels, labels, "Red de Efectos Olvidados"))

                inspector_focalizado(df_efectos, labels, labels, "cerrado")

# ============================================================
# MODO 2: DOS MATRICES (SISTEMA ENCADENADO)
# ============================================================
elif "2." in modo:
    st.header("🔗 Análisis de Sistema Encadenado (A → B → C)")

    with st.container(border=True):
        st.markdown("**Paso 1 — Defina M1 (A → B)**")
        df1 = matrix_input_widget("enc_m1", "Matriz M1 (A → B)", row_hint="elementos de A (filas)", col_hint="elementos de B (columnas)")

    with st.container(border=True):
        st.markdown("**Paso 2 — Defina M2 (B → C)**")
        df2 = matrix_input_widget("enc_m2", "Matriz M2 (B → C)", row_hint="elementos de B (filas)", col_hint="elementos de C (columnas)")

    if df1 is not None and df2 is not None:
        # Validar compatibilidad
        if df1.shape[1] != df2.shape[0]:
            st.error(
                f"⚠️ Incompatibilidad: las columnas de M1 ({df1.shape[1]}) "
                f"deben ser iguales a las filas de M2 ({df2.shape[0]})."
            )
        else:
            st.success(f"✅ Matrices compatibles: M1 es {df1.shape[0]}×{df1.shape[1]}, M2 es {df2.shape[0]}×{df2.shape[1]}.")

            # Matriz directa A→C opcional
            with st.container(border=True):
                st.markdown("**Paso 3 (Opcional) — Matriz Directa A → C**")
                st.caption("Si la proporciona, se calcularán los Efectos Olvidados = MAC_composición − MAC_directa.")
                df_dir = matrix_input_widget("enc_dir", "Matriz Directa (A → C)", row_hint="elementos de A (filas)", col_hint="elementos de C (columnas)")

            if st.button("🚀 Procesar Composición", type="primary", key="btn_encadenado"):
                st.session_state['procesado_encadenado'] = True

            if st.session_state.get('procesado_encadenado', False):
                resultado = max_min_composition(df1.values, df2.values)
                labels_A = list(df1.index)
                labels_C = list(df2.columns)
                df_res = pd.DataFrame(resultado, index=labels_A, columns=labels_C)

                tabs_list = ["📊 Matriz Resultante (MAC)"]
                if df_dir is not None:
                    tabs_list.append("🔍 Efectos Olvidados")

                tabs = st.tabs(tabs_list)
                with tabs[0]:
                    c1, c2 = st.columns([1, 1.5])
                    with c1:
                        st.dataframe(df_res.style.format("{:.3f}"))
                    with c2:
                        st.pyplot(draw_incidence_graph(resultado, labels_A, labels_C, "Red de Composición (A → C)"))

                if df_dir is not None and len(tabs_list) > 1:
                    with tabs[1]:
                        if df_dir.shape != df_res.shape:
                            st.error(f"⚠️ La matriz directa debe ser {df_res.shape[0]}×{df_res.shape[1]}, pero tiene {df_dir.shape[0]}×{df_dir.shape[1]}.")
                        else:
                            efectos = subtract_matrices(resultado, df_dir.values)
                            df_efectos = pd.DataFrame(efectos, index=labels_A, columns=labels_C)
                            c1, c2 = st.columns([1, 1.5])
                            with c1:
                                st.dataframe(df_efectos.style.format("{:.3f}"))
                            with c2:
                                st.pyplot(draw_incidence_graph(efectos, labels_A, labels_C, "Red de Efectos Olvidados"))

                            inspector_focalizado(df_efectos, labels_A, labels_C, "encadenado")

# ============================================================
# MODO 3: TRES MATRICES (CADENA COMPLEJA)
# ============================================================
elif "3." in modo:
    st.header("⛓️ Análisis de Cadena Compleja (A → B → C → D)")

    with st.container(border=True):
        st.markdown("**Paso 1 — Defina M1 (A → B)**")
        df1 = matrix_input_widget("comp_m1", "Matriz M1 (A → B)", row_hint="elementos de A (filas)", col_hint="elementos de B (columnas)")

    with st.container(border=True):
        st.markdown("**Paso 2 — Defina M2 (B → C)**")
        df2 = matrix_input_widget("comp_m2", "Matriz M2 (B → C)", row_hint="elementos de B (filas)", col_hint="elementos de C (columnas)")

    with st.container(border=True):
        st.markdown("**Paso 3 — Defina M3 (C → D)**")
        df3 = matrix_input_widget("comp_m3", "Matriz M3 (C → D)", row_hint="elementos de C (filas)", col_hint="elementos de D (columnas)")

    if df1 is not None and df2 is not None and df3 is not None:
        # Validaciones
        err1 = df1.shape[1] != df2.shape[0]
        err2 = df2.shape[1] != df3.shape[0]
        if err1:
            st.error(f"⚠️ Incompatibilidad M1→M2: cols(M1)={df1.shape[1]} ≠ filas(M2)={df2.shape[0]}.")
        if err2:
            st.error(f"⚠️ Incompatibilidad M2→M3: cols(M2)={df2.shape[1]} ≠ filas(M3)={df3.shape[0]}.")

        if not err1 and not err2:
            st.success("✅ Las tres matrices encajan correctamente.")

            with st.container(border=True):
                st.markdown("**Paso 4 (Opcional) — Matriz Directa A → D**")
                st.caption("Necesaria para calcular Efectos Olvidados de 3ª generación = MAD_composición − MAD_directa.")
                df_dir = matrix_input_widget("comp_dir", "Matriz Directa (A → D)", row_hint="elementos de A (filas)", col_hint="elementos de D (columnas)")

            if st.button("🚀 Procesar Composición Triple", type="primary", key="btn_complejo"):
                st.session_state['procesado_complejo'] = True

            if st.session_state.get('procesado_complejo', False):
                paso1 = max_min_composition(df1.values, df2.values)
                resultado = max_min_composition(paso1, df3.values)

                labels_A = list(df1.index)
                labels_D = list(df3.columns)
                df_res = pd.DataFrame(resultado, index=labels_A, columns=labels_D)

                tabs_list = ["📊 Composición Triple (MAD)"]
                if df_dir is not None:
                    tabs_list.append("🔍 Efectos Olvidados de 3ª Generación")

                tabs = st.tabs(tabs_list)
                with tabs[0]:
                    c1, c2 = st.columns([1, 1.5])
                    with c1:
                        st.dataframe(df_res.style.format("{:.3f}"))
                    with c2:
                        st.pyplot(draw_incidence_graph(resultado, labels_A, labels_D, "Red de Composición (A → D)"))

                if df_dir is not None and len(tabs_list) > 1:
                    with tabs[1]:
                        if df_dir.shape != df_res.shape:
                            st.error(f"⚠️ La matriz directa debe ser {df_res.shape[0]}×{df_res.shape[1]}, pero tiene {df_dir.shape[0]}×{df_dir.shape[1]}.")
                        else:
                            efectos = subtract_matrices(resultado, df_dir.values)
                            df_efectos = pd.DataFrame(efectos, index=labels_A, columns=labels_D)
                            c1, c2 = st.columns([1, 1.5])
                            with c1:
                                st.dataframe(df_efectos.style.format("{:.3f}"))
                            with c2:
                                st.pyplot(draw_incidence_graph(efectos, labels_A, labels_D, "Efectos Olvidados de 3ª Generación"))

                            inspector_focalizado(df_efectos, labels_A, labels_D, "complejo")
