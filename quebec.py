import streamlit as st
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


@st.cache_resource
def load_data():

    df = pd.read_csv("Canada_limpio.csv")
    df= df.drop(['Unnamed: 0'], axis=1)

    df_ds = pd.read_csv("Canada_limpio.csv")
    #convertimos string a tipos numéricos
    df['host_is_superhost'] = df['host_is_superhost'].replace({'f': 0, 't': 1})

    df['host_response_rate'] = df['host_response_rate'].astype(str).str.rstrip('%')
    df['host_response_rate'] = pd.to_numeric(df['host_response_rate'], errors='coerce').fillna(0)

    df['room_type'] = df['room_type'].map({
        'Entire home/apt': 1,
        'Private room': 2,
        'Shared room': 3,
        'Hotel room': 4
    })

    df_dico = pd.read_csv("cnDico.csv")
    df_dico= df_dico.drop(['Unnamed: 0'], axis=1)

    numeric_df = df.select_dtypes(['float', 'int'])
    numeric_cols = numeric_df.columns
    numeric_cols2 = list(numeric_cols)  # Asegúrate de hacer esto al inicio de tu código
    text_df = df.select_dtypes(['object'])
    text_cols = text_df.columns
    categorical_column = df['host_is_superhost']
    unique_categories = categorical_column.unique()
    return df,df_ds, df_dico, numeric_cols, numeric_cols2, text_cols, unique_categories, numeric_df

df,df_ds, df_dico, numeric_cols, numeric_cols2, text_cols, unique_categories, numeric_df = load_data()

###########################################
# ESTILO
st.markdown("""
    <style>
    html, body, .stApp {
        background-color: #2c2e60;
        padding: 0 !important;
        margin-top:2em;
        margin-bottom: 0 !important;
        margin-left:  0 !important;
        margin-right:  0 !important;
        height: 100%;
        width: 100%;
        font-family: 'Poppins', sans-serif;
    }

    .block-container {
        padding: 1rem 2rem;
        max-width: 1200px;
        margin: auto;
    }

    .main, .css-18e3th9, .css-1d391kg {
        width: 100%;
    }

    .stSidebar {
        background-color: #001428;
        padding-top: 2rem;
        padding-left: 10px;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #fafaff;
        font-family: 'Roboto', sans-serif;
    }

    .stHeader {
        color: #fafaff;
        font-family: 'Roboto', sans-serif;
    }

    .stMarkdown {
        color: #f0f5ff;
        font-size: 18px;
    }

    img {
        display: block;
        margin-left: auto;
    }
    </style>
""", unsafe_allow_html=True)

##########################################
# VISTA INICIAL
col1, col2 = st.columns([1, 3])  # Ajusta los tamaños para que el texto tenga más espacio

with col1:
    # Imagen del ícono
    st.image("img/icono_transparente.png", width=200)

with col2:
    # Títulos alineados a la izquierda
    st.markdown( """
        <div style="height: 200px; display: flex; align-items: center;">
            <p style="color:#fafaff; font-size:80px; font-weight:bold; font-family:'Trebuchet MS'; margin: 0;">
            GrowthTrack
            </p>
        </div>
        """, unsafe_allow_html=True)
      

##########################################
# SIDEBAR
st.sidebar.image("img/LOGO.png")
with st.sidebar:
    col1, col2 = st.columns([1, 4])  # proporción ajustable

    with col1:
        st.image("img/maple.png", width=30)

    with col2:
        st.markdown("### CANADÁ")
ViewC = st.sidebar.selectbox("CANADÁ", ["INICIO", "QUEBEC"], label_visibility="collapsed")
ViewQ = st.sidebar.selectbox("Más Sobre Quebec",["SELECCIONAR","DISTRIBUCIÓN"])

##########################################
# LÓGICA PARA LA VISTA DE INICIO
if ViewC == "INICIO":
    col1, col2 = st.columns(2)

    with col1:
        st.image("img/Quebec_city.jpg", use_container_width=True)
        st.markdown("Quebec, Canada")

    with col2:
        st.subheader("**🏙️ Datos clave de Quebec City:**")
        st.markdown("""
        - Población: 542,298 habitantes (2023)
        - Idioma oficial: Francés (94.5% francófonos)
        - Área: 485.77 km²
        - Clima: Continental húmedo (Veranos 20°C, Inviernos -10°C)
        """)
        st.subheader("**🏰 Puntos de interés turístico:**")
        st.markdown("""
        - **Château Frontenac**: Hotel icónico (el más fotografiado del mundo)
        - **Place Royale**: Lugar histórico de fundación (1608)
        - **Montmorency Falls**: Cascada 30m más alta que Niagara
        """)
##########################################
# Añade esto justo después de cargar los datos
map_df = df_ds[
    (df_ds['latitude'].notna()) & 
    (df_ds['longitude'].notna()) &
    (df_ds['latitude'].between(46.7, 46.9)) &  # Rango geográfico de Québec
    (df_ds['longitude'].between(-71.3, -71.1))
].copy()
# DISTRIBUCIÓN DE ALOJAMIENTOS
if ViewQ == "DISTRIBUCIÓN":
    st.header("Mapa de Distribución de Alojamientos en Quebec")
    
    # Mapa minimalista con zoom mejorado
    quebec_map = folium.Map(
        location=[46.8139, -71.2080],
        zoom_start=16,  
        tiles="CartoDB positron",  # Estilo minimalista
        width="100%",
        height=600,
        control_scale=True 
    )

    # Configuración minimalista de marcadores
    for idx, row in map_df.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=4,  # Tamaño reducido
            popup=folium.Popup(
                f"""
                <div style='font-family: Arial; font-size: 12px'>
                    <b>Tipo:</b> {row['room_type']}<br>
                    <b>Superhost:</b> {'Sí' if row['host_is_superhost'] == 1 else 'No'}<br>
                    <b>Puntuación:</b> {row['review_scores_rating']:.1f}/5
                </div>
                """,
                max_width=200
            ),
            color='#4169E1',  # Azul minimalista
            fill=True,
            fill_opacity=0.6,  # Transparencia
            weight=1  # Grosor del borde
        ).add_to(quebec_map)

    # Añadir controles de zoom mejorados
    folium.plugins.MousePosition().add_to(quebec_map)
    folium.plugins.Fullscreen(position="topright").add_to(quebec_map)

    # Mostrar el mapa con márgenes ajustados
    folium_static(quebec_map, width=1000, height=600)
    
    # Análisis de densidad
    st.subheader("Análisis de Densidad de Propiedades")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Concentración de Propiedades**")
        # Heatmap con personalización
        density_fig = px.density_mapbox(
            map_df,
            lat='latitude',
            lon='longitude',
            z='price',
            radius=20,  # Aumentamos el radio para mejor visualización
            zoom=15,    # Zoom máximo recomendado (valor entre 1-20)
            center={"lat": 46.8139, "lon": -71.2080},
            mapbox_style="carto-positron",  # Estilo minimalista
            color_continuous_scale=px.colors.sequential.Blues,  # Escala de colores simple
            title="<b></b>",
            hover_data={'price': ':.2f'},
            opacity=0.8  # Transparencia para mejor legibilidad
        )

        # Ajustes adicionales
        density_fig.update_layout(
            margin={"r":0,"t":40,"l":0,"b":0},
            mapbox={
                'style': "carto-positron",  # Fuerza estilo minimalista
                'zoom': 15,                 # Reafirmamos el zoom
                'center': {"lat": 46.8139, "lon": -71.2080}
            },
            coloraxis_colorbar={
                'title': 'Precio promedio',
                'tickprefix': '$'
            }
        )

        st.plotly_chart(density_fig, use_container_width=True)

    
    with col2:
        # Conteo por vecindario
        st.markdown("**Propiedades por Área**")
        map_df['area_rounded'] = map_df.apply(lambda x: f"{round(x['latitude'], 3)}, {round(x['longitude'], 3)}", axis=1)
        area_counts = map_df['area_rounded'].value_counts().head(10).reset_index()
        area_counts.columns = ['Área', 'Cantidad']
        st.bar_chart(area_counts.set_index('Área'))
        


##########################################
# LÓGICA PARA LA VISTA DE QUEBEC
if ViewC == "QUEBEC":
    st.markdown('<h3 style="color:#fff;font-weight:bold;">Selecciona Vista de Análisis de datos en Quebec</h3>', unsafe_allow_html=True)
    vista = st.selectbox("", ["📂 Dataset Quebec Airbnb", "Gráficos Univariados", "Regresiones"], label_visibility="collapsed")
    
    if vista == "📂 Dataset Quebec Airbnb":
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            Contiene **{:,} propiedades** con información detallada sobre:
            - Precios y disponibilidad
            - Evaluaciones de huéspedes
            """.format(len(df)))
        with col2:
            st.markdown("""
            - Características de los alojamientos
            - Datos de anfitriones

            *Actualizado: {}*
            """.format(pd.Timestamp.now().strftime("%d/%m/%Y")))
        st.write(df)
        
        st.markdown("""
        > ℹ️ **Nota sobre los datos**: 
        > Este dataset ha sido procesado para garantizar la privacidad de los anfitriones, 
        > eliminando información personal identificable según las políticas de Airbnb.
        """)

    elif vista == "Gráficos Univariados":
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Distribución Superhosts")
            freq_data = df['host_is_superhost'].value_counts().reset_index()
            freq_data.columns = ['category', 'count']
            
            figure1 = px.bar(
                freq_data,
                x='category',
                y='count',
                title="Superhosts (0=No, 1=Sí)",
                labels={'category': '', 'count': 'Frecuencia'},
                color='category',
                color_discrete_map={0: '#6d38aa', 1: '#fbb77c'}
            )
            st.plotly_chart(figure1, use_container_width=True)
             
        with col2:
            st.subheader("Distribución Precios")
            # Solución directa sin usar histograma intermedio
            price_counts = df['price'].value_counts().sort_index().reset_index()
            price_counts.columns = ['price', 'count']
            
            figure2 = px.line(
                price_counts,
                x='price',
                y='count',
                title=" ",
                labels={'price': 'Precio', 'count': 'Frecuencia'}
            )
            st.plotly_chart(figure2, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Distribución de Tipos de Habitación")
            room_counts = df['room_type'].value_counts().reset_index()
            room_counts.columns = ['room_type', 'count']
            
            figure3 = px.pie(
                room_counts,
                names='room_type',
                values='count',
                title="Tipo de Habitación:",
                subtitle="1.-Entire home/apt, 2.-Private room, 3.-Shared room, 4.-Hotel room.",
            )
            st.plotly_chart(figure3, use_container_width=True)

        with  col4:
            # 2. Distribución de Reviews (Scatterplot)
            st.subheader("Puntuaciones de Reviews")
            review_freq = df['review_scores_rating'].value_counts().reset_index()
            fig2 = px.scatter(
                review_freq,
                x='review_scores_rating',
                y='count',
                size='count',
                color='count',
                title="Frecuencia de Puntuaciones",
                labels={'review_scores_rating': 'Puntuación', 'count': 'Frecuencia'}
            )
            st.plotly_chart(fig2, use_container_width=True)
                
         # Tabla resumen de frecuencias
        st.subheader("Tabla Resumen de Frecuencias")
        freq_data = []
        for var in ['host_is_superhost', 'room_type', 'bedrooms', 'review_scores_rating']:
            freq = df[var].value_counts().reset_index()
            freq.columns = ['Valor', 'Frecuencia']
            freq = freq[freq['Frecuencia'] > 50]
            freq['Variable'] = var
            freq['Porcentaje'] = (freq['Frecuencia'] / freq['Frecuencia'].sum()) * 100
            freq_data.append(freq)
        
        st.dataframe(
            pd.concat(freq_data),
            use_container_width=True,
            column_order=['Variable', 'Valor', 'Frecuencia', 'Porcentaje'],
            hide_index=True
        )

    elif vista == "Regresiones":
        st.sidebar.header("Regresiones Lineales y Lógicas")

        tipo_regresion = st.sidebar.selectbox("Tipo de regresión", ["Lineal Simple", "Lineal Múltiple", "Lógica"])

        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
        import numpy as np
        import matplotlib.pyplot as plt

        if tipo_regresion == "Lineal Simple":
            st.header("Regresión Lineal Simple")
            default_x = "review_scores_communication" if "id" in numeric_cols2 else numeric_cols[0]
            default_y = "id" if "review_scores_communication" in numeric_cols2 else numeric_cols[0]

            var_x = st.sidebar.selectbox("Selecciona variable independiente (X)", options=numeric_cols, index=numeric_cols.get_loc(default_x))
            var_y = st.sidebar.selectbox("Selecciona variable dependiente (Y)", options=numeric_cols, index=numeric_cols.get_loc(default_y))

            X = df[[var_x]]
            y = df[var_y]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            col9, col10 = st.columns(2)

            with col9:
                fig, ax = plt.subplots()
                ax.scatter(X_test, y_test, color='#6d38aa', label='Datos reales')
                ax.scatter(X_test, y_pred, color='#fbb77c', label='Predicción')
                ax.set_xlabel(var_x)
                ax.set_ylabel(var_y)
                ax.legend()
                st.pyplot(fig)

            with col10:
                st.subheader("Resultados del Modelo")
                st.metric("Intercepto:" ,f"{model.intercept_:.2f}")
                r2 = r2_score(y_test, y_pred)
                st.metric("Coeficiente de determinación:" ,f"{r2:.2f}")
                corr = np.sign(model.coef_[0]) * np.sqrt(r2)
                st.metric("Coeficiente de correlación (R):" ,f"{corr:.2f}")


        elif tipo_regresion == "Lineal Múltiple":
            st.header("Regresión Lineal Múltiple")
            
            selected_features = st.sidebar.multiselect(
                "Selecciona variables independientes (X)",
                options=numeric_cols2,
                default=["reviews_per_month", "review_scores_communication"] if "reviews_per_month" in numeric_cols2 and "review_scores_communication" in numeric_cols2 else numeric_cols2[:2]
            )
            
            var_y = st.sidebar.selectbox(
                "Selecciona variable dependiente (Y)",
                options=numeric_cols2,
                index=numeric_cols2.index("id") if "id" in numeric_cols2 else 0
            )

            if selected_features and var_y:
                X = df[selected_features]
                y = df[var_y]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Gráfico comparativo usando la primera variable X
                import seaborn as sns
                df_viz = X_test.copy()
                df_viz[var_y] = y_test.values
                df_viz["Pred_" + var_y] = y_pred

            col9, col10 = st.columns(2)

            with col9:
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.scatterplot(x=selected_features[0], y=var_y, data=df_viz, color="#6d38aa", label="Real", ax=ax)
                sns.scatterplot(x=selected_features[0], y="Pred_" + var_y, data=df_viz, color="#fbb77c", label="Predicción", ax=ax)
                ax.set_xlabel(selected_features[0])
                ax.set_ylabel(var_y)
                ax.legend()
                st.pyplot(fig)

            with col10:
                # Mostrar métricas
                st.subheader("Resultados del Modelo")
                st.metric(f"Intercepto:" ,f"{model.intercept_:.4f}")
                st.metric(f"Coeficiente de determinación: ",f"{model.score(X, y):.4f}" )
                st.metric(f"Coeficiente de correlación (R): ",f"{np.sqrt(model.score(X,y)):.4f}")


        elif tipo_regresion == "Lógica":
            st.header("Regresión Logística")
            
            # Seleccionar variable objetivo binaria (convertir price a binario si es necesario)
            st.sidebar.subheader("Configuración del Modelo")
            
            # Crear variable objetivo binaria basada en el precio (por ejemplo, precio > mediana)
            median_price = df['price'].median()
            df['price_category'] = (df['price'] > median_price).astype(int)
            
            # Seleccionar características
            available_features = [col for col in numeric_cols if col != 'price']
            selected_features = st.sidebar.multiselect(
                "Selecciona variables predictoras",
                options=available_features,
                default=['host_is_superhost', 'beds', 'review_scores_rating'] if all(x in available_features for x in ['host_is_superhost', 'beds', 'review_scores_rating']) else available_features[:3]
            )
            
            if selected_features:
                X = df[selected_features]
                y = df['price_category']
                
                # Dividir datos
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                
                # Estandarizar características
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Entrenar modelo
                log_reg = LogisticRegression(max_iter=1000)
                log_reg.fit(X_train_scaled, y_train)
                
                # Predecir
                y_pred = log_reg.predict(X_test_scaled)
                y_probs = log_reg.predict_proba(X_test_scaled)[:, 1]
                
                # Métricas
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                
                col9, col10 = st.columns(2)

                with col9:
                    # Matriz de confusión
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                    ax.set_xlabel('Predicho')
                    ax.set_ylabel('Real')
                    ax.set_title('Matriz de Confusión')
                    st.pyplot(fig)
                
                with col10:
                    # Mostrar métricas
                    st.subheader("Resultados del Modelo")
                    st.metric("Exactitud (Accuracy)", f"{accuracy:.2f}")
                    st.metric("Precisión (Precision)", f"{precision:.2f}")
                    st.metric("Sensibilidad (Recall)", f"{recall:.2f}")

                # Coeficientes del modelo
                st.subheader("Coeficientes del Modelo")
                coef_df = pd.DataFrame({
                    'Variable': selected_features,
                    'Coeficiente': log_reg.coef_[0]
                })
                st.dataframe(coef_df.sort_values('Coeficiente', ascending=False))
                
                
