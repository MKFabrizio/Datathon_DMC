import polars as pl 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

# Función para mostrar las frecuencias absolutas y relativas de un campo
def frecuencia_categoria(df,columna,orden):
    resultado = df.groupby([columna]).agg(CANTIDAD=(columna, 'count')).reset_index()
    total = resultado['CANTIDAD'].sum()
    resultado['% TOTAL'] = (resultado['CANTIDAD'] / total) * 100
    if orden == 0:
        resultado = resultado.sort_values('% TOTAL', ascending=False)
        return resultado
    elif orden == 1:
        resultado = resultado.sort_values(columna, ascending=True)
        return resultado
    else:
        return print('En orden solo puede escoger los valores 0,1')
        


# Función para graficar los campos categóricos
# Recomendaciones ncol=2, color='rocket',grafico=barplot
def plot_categoria(df,lista_campos,ncol,color,ancho,largo):
    # Calcular el número de filas y columnas para organizar los subgráficos
    num_filas = (len(lista_campos) + 1) // ncol
    num_columnas = ncol if len(lista_campos) > 1 else 1

    # Crear la figura y los subgráficos
    fig, axs = plt.subplots(num_filas, num_columnas, figsize=(ancho, largo))

    # Ajustar los espacios entre los subgráficos
    plt.subplots_adjust(hspace=0.2)

    # Iterar sobre los encabezados y crear un gráfico para cada columna del DataFrame
    for i, encabezado in enumerate(lista_campos):
        fila = i // ncol
        columna = i % ncol if num_columnas > 1 else 0
        ax = axs[fila, columna] if num_filas > 1 else axs[columna]
        counts = df[encabezado].value_counts()  # Obtener conteo de valores
        sns.barplot(x=counts.values, y=counts.index, palette=color, ax=ax)
        ax.set_title(f"CATEGORÍA {encabezado}")
        ax.set_ylabel("")  # Eliminar título del eje y

    return plt.show()

def var_nan_flag(df, columna, null="None"):
    new_col_name = columna + '_flag'
    if null=="None":

        df = (
            df
            .with_columns(
                pl.when(pl.col(columna).is_null())
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .alias(new_col_name)
            )
        )
    
    elif null!="None":
        df = (
            df
            .with_columns(
                pl.when(pl.col(columna)==null)
                .then(pl.lit(1))
                .otherwise(pl.lit(0))
                .alias(new_col_name)
            )
        )

    return df

def missing_values_table(df):

    mis_val = df.select(pl.all().is_null().sum())
    mis_val_percent = 100 * mis_val / len(df)

    mis_val = mis_val.row(0)
    mis_val_percent = mis_val_percent.row(0)
    mis_val_columnas = df.columns

    mis_val_table = (

        pl.DataFrame(
            {
                'columnas':mis_val_columnas,
                'num_missings':mis_val,
                '% missings':mis_val_percent
            }
        )
        .filter(pl.col('num_missings')!=0)
        .sort('num_missings', descending=True)
    )
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
        "There are " + str(mis_val_table.shape[0]) +
          " columns that have missing values.")

    return mis_val_table, mis_val_table.get_column('columnas').to_list()

def reemplazar_por_mediana(x,limite_superior,limite_inferior,mediana):
    if (x>limite_superior)|(x<limite_inferior):
        return mediana
    else:
        return x
    
def porcentaje_outlier(df, column):
    rango_intercuartil = df.get_column(column).quantile(0.75) - df.get_column(column).quantile(0.25)
    limite_superior = df.get_column(column).quantile(0.75) + 1.5*rango_intercuartil
    limite_inferior = df.get_column(column).quantile(0.25) - 1.5*rango_intercuartil
    

    outliers_count = df.shape[0] - df.filter(pl.col(column).is_between(limite_inferior,limite_superior)).shape[0]

    return limite_superior,limite_inferior, np.round(outliers_count*100/df.shape[0],2)
    
def tratar_outlier(df, column, min=0.0001,max=0.95):

    limite_superior,limite_inferior, porcentaje = porcentaje_outlier(df, column)
        
    if (porcentaje>=5):

        percentil_95 = df.get_column(column).quantile(max)
        percentil_2 = df.get_column(column).quantile(min)
        df = df.with_columns(pl.col(column).map_elements(lambda x: percentil_95 if x > percentil_95 else x))
        df = df.with_columns(pl.col(column).map_elements(lambda x: percentil_2 if x < percentil_2 else x))
        return df
    else:
        return df
    
def reemplazar_negativo(df, columna):

    return (
        df
        .with_columns(pl.when(pl.col(columna)<0).then(pl.lit(0)).otherwise(pl.col(columna)).alias(columna))
    )



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from prince import MCA


def preprocess_data(df, target_column):
    # Identify categorical columns
    cat_columns = df.select_dtypes(include=['object', 'category']).columns
    cat_columns = [col for col in cat_columns if col != target_column]
    
    # Handle missing values
    for col in cat_columns:
        df[col].fillna('Unknown', inplace=True)
    
    print("Preprocessing completed.")
    return df, cat_columns


def analyze_categorical(df, target_column, cat_columns):
    results = {}
    for col in cat_columns:
        # Frequency distribution
        freq = df[col].value_counts(normalize=True)
        
        # Target rate analysis
        target_rates = df.groupby(col)[target_column].mean()
        # Chi-square test
        contingency_table = pd.crosstab(df[col], df[target_column])
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
                
        # Information Value
        iv = calculate_iv(df, col, target_column)
        
        results[col] = {
            'frequency': freq,
            'target_rates': target_rates,
            'chi2_p_value': p_value,
            'iv': iv
        }
    
    return results


def calculate_iv(df, feature, target):
    eps = 1e-10  # Small value to prevent division by zero
    grouped = df.groupby(feature)[target].agg(['count', 'sum'])
    grouped['non_event'] = grouped['count'] - grouped['sum']
    grouped['f_perc'] = grouped['count'] / grouped['count'].sum()
    grouped['non_event_perc'] = (grouped['non_event'] + eps) / (grouped['non_event'].sum() + eps)
    grouped['event_perc'] = (grouped['sum'] + eps) / (grouped['sum'].sum() + eps)
    grouped['woe'] = np.log(grouped['event_perc'] / grouped['non_event_perc'])
    grouped['iv'] = (grouped['event_perc'] - grouped['non_event_perc']) * grouped['woe']
    iv = grouped['iv'].sum()
    return iv if np.isfinite(iv) else None

def visualize_results(df, analysis_results, target_column):
    for col, result in analysis_results.items():
        plt.figure(figsize=(16, 6))
        
        # Frequency plot
        plt.subplot(131)
        result['frequency'].plot(kind='bar')
        plt.title(f'Frequency Distribution - {col}')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha='right')
        
        # Target rate plot
        plt.subplot(132)
        result['target_rates'].plot(kind='bar')
        plt.title(f'Target Rate - {col}')
        plt.ylabel('Target Rate')
        plt.xticks(rotation=45, ha='right')
        
        # Mosaic plot
        plt.subplot(133)
        contingency_table = pd.crosstab(df[col], df[target_column])
        sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlGnBu')
        plt.title(f'Mosaic Plot - {col} vs Target')
        plt.ylabel(col)
        plt.xlabel('Target')
        
        plt.tight_layout()
        plt.show()

def feature_importance(df, target_column, cat_columns):
    le = LabelEncoder()
    X = df[cat_columns].apply(le.fit_transform)
    y = df[target_column]

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=cat_columns).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    importances.plot(kind='bar')
    plt.title('Feature Importance')
    plt.ylabel('Importance')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    return importances

def mutual_information(df, target_column, cat_columns):
    le = LabelEncoder()
    X = df[cat_columns].apply(le.fit_transform)
    y = df[target_column]

    mi_scores = mutual_info_classif(X, y)
    mi_scores = pd.Series(mi_scores, index=cat_columns).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    mi_scores.plot(kind='bar')
    plt.title('Mutual Information Scores')
    plt.ylabel('Mutual Information')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    return mi_scores

def correlation_analysis(df, cat_columns):
    # One-hot encode categorical variables
    #onehot = OneHotEncoder()
    #encoded = onehot.fit_transform(df[cat_columns])
    #encoded_df = pd.DataFrame(encoded, columns=onehot.get_feature_names_out(cat_columns))
    encoded_df = pd.get_dummies(df, cat_columns)
    
    # Calculate correlation matrix
    corr_matrix = encoded_df.corr()
    
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap of Encoded Categorical Variables')
    plt.tight_layout()
    plt.show()

def target_encoding_performance(df, target_column, cat_columns):
    # Perform target encoding
    encoded_df = df.copy()
    for col in cat_columns:
        target_mean = df.groupby(col)[target_column].mean()
        encoded_df[f'{col}_encoded'] = df[col].map(target_mean)
    
    # Prepare data for modeling
    X = encoded_df[[f'{col}_encoded' for col in cat_columns]]
    y = encoded_df[target_column]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train logistic regression
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate performance
    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    print(f"Logistic Regression performance after target encoding:")
    print(f"Train AUC: {train_auc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    
    # Feature importance based on logistic regression coefficients
    feature_importance = pd.Series(np.abs(model.coef_[0]), index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    feature_importance.plot(kind='bar')
    plt.title('Feature Importance after Target Encoding')
    plt.ylabel('Absolute Coefficient Value')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def mca_analysis(df, cat_columns):
    # Perform MCA
    mca = MCA(n_components=2, random_state=42)
    mca_coords = mca.fit_transform(df[cat_columns])
    
    # Plot results
    plt.figure(figsize=(10, 8))
    plt.scatter(mca_coords.iloc[:, 0], mca_coords.iloc[:, 1], alpha=0.7)
    plt.title('Multiple Correspondence Analysis')
    plt.xlabel('First Component')
    plt.ylabel('Second Component')
    plt.tight_layout()
    plt.show()

def analyze_interactions(df, target_column, cat_columns):
    for i, col1 in enumerate(cat_columns):
        for col2 in cat_columns[i+1:]:
            plt.figure(figsize=(12, 8))
            interaction_data = df.groupby([col1, col2])[target_column].mean().unstack()
            sns.heatmap(interaction_data, annot=True, fmt='.2f', cmap='YlGnBu')
            plt.title(f'Interaction Effect: {col1} vs {col2}')
            plt.tight_layout()
            plt.show()
            
def create_summary_table(analysis_results, importances, mi_scores):
    summary = []
    for col, result in analysis_results.items():
        summary.append({
            'Variable': col,
            'Chi-square p-value': result['chi2_p_value'],
            'Information Value': result['iv'],
            'Feature Importance': importances.get(col, 0),
            'Mutual Information': mi_scores.get(col, 0)
        })
    
    summary_df = pd.DataFrame(summary)
    
    # Sort by a composite score (you can adjust the weights as needed)
    summary_df['Composite Score'] = (
        summary_df['Information Value'] * 0.3 +
        summary_df['Feature Importance'] * 0.3 +
        summary_df['Mutual Information'] * 0.2 +
        (1 - summary_df['Chi-square p-value']) * 0.2
    )
    
    summary_df = summary_df.sort_values('Composite Score', ascending=False).reset_index(drop=True)
    
    return summary_df

def visualize_top_variables(summary_df, top_n=10):
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Composite Score', y='Variable', data=summary_df.head(top_n))
    plt.title(f'Top {top_n} Categorical Variables by Composite Score')
    plt.tight_layout()
    plt.show()

def main(df, target_column):
    # Load and preprocess data
    df, cat_columns = preprocess_data(df, target_column)
    
    # Perform categorical variable analysis
    analysis_results = analyze_categorical(df, target_column, cat_columns)
    
    # Visualize results
    visualize_results(df, analysis_results, target_column)
    
    # Feature importance
    importances = feature_importance(df, target_column, cat_columns)
    print("Feature Importance:")
    print(importances)
    
    # Mutual Information
    mi_scores = mutual_information(df, target_column, cat_columns)
    print("\nMutual Information Scores:")
    print(mi_scores)
    
    # Correlation analysis
    correlation_analysis(df, cat_columns)
    
    # Target encoding performance
    target_encoding_performance(df, target_column, cat_columns)
    
    # MCA analysis
    mca_analysis(df, cat_columns)
    
    # Analyze interactions
    # analyze_interactions(df, target_column, cat_columns)
    summary_table = create_summary_table(analysis_results, importances, mi_scores)
    print(summary_table.head(10).to_string(index=False))
    visualize_top_variables(summary_table)
    
    

    # Print summary of findings
    print("\nSummary of Findings:")
    for col, result in analysis_results.items():
        print(f"\nVariable: {col}")
        print(f"Chi-square p-value: {result['chi2_p_value']:.4f}")
        print(f"Information Value: {result['iv']:.4f}")
        print(f"Feature Importance: {importances[col]:.4f}")
        print(f"Mutual Information: {mi_scores[col]:.4f}")
    
    print("\nRecommendations:")
    print("1. Consider keeping variables with low chi-square p-values (< 0.05).")
    print("2. Prioritize variables with higher Cramér's V, Information Value, Feature Importance, and Mutual Information scores.")
    print("3. Be cautious of variables with very high cardinality or strong correlations with other variables.")
    print("4. Review interaction effects for potential feature engineering opportunities.")
    print("5. Consult domain experts to validate these findings and consider business context.")
    
    return summary_table