import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import re


# 1) VERİYİ YÜKLEME
def load_data(path):
    df = pd.read_excel(path)
    df.columns = [col.strip() for col in df.columns]  # kolon adlarını temizle

    # Boş stringleri NaN'a çevir
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].replace(['', 'none', 'nan', 'NaN', 'None'], np.nan)

    return df


# 2) METİN SÜTUNLARINI AYRIŞTIRMA
def preprocess_text_columns(df):
    """
    KronikHastalik, Alerji, Tanilar ve UygulamaYerleri sütunlarını ayrıştırır
    """
    df_copy = df.copy()
    
    # KronikHastalik için işlemler
    if 'KronikHastalik' in df_copy.columns:
        chronic_diseases = df_copy['KronikHastalik'].dropna().str.split(',').explode()
        chronic_diseases = chronic_diseases.str.strip()
        top_chronic = chronic_diseases.value_counts().head(8).index.tolist()
        for disease in top_chronic:
            pattern = re.escape(str(disease))
            df_copy[f'Kronik_{disease}'] = df_copy['KronikHastalik'].str.contains(pattern, na=False).astype(int)
        df_copy['Kronik_Hastalik_Sayisi'] = df_copy['KronikHastalik'].apply(
            lambda x: len(x.split(',')) if pd.notna(x) else 0
        )
    
    # Alerji için işlemler
    if 'Alerji' in df_copy.columns:
        allergies = df_copy['Alerji'].dropna().str.split(',').explode()
        allergies = allergies.str.strip()
        top_allergies = allergies.value_counts().head(5).index.tolist()
        for allergy in top_allergies:
            pattern = re.escape(str(allergy))
            df_copy[f'Alerji_{allergy}'] = df_copy['Alerji'].str.contains(pattern, na=False).astype(int)
        df_copy['Alerji_Sayisi'] = df_copy['Alerji'].apply(
            lambda x: len(x.split(',')) if pd.notna(x) else 0
        )
    
    # Tanilar için işlemler
    if 'Tanilar' in df_copy.columns:
        diagnoses = df_copy['Tanilar'].dropna().str.split(',').explode()
        diagnoses = diagnoses.str.strip()
        top_diagnoses = diagnoses.value_counts().head(10).index.tolist()
        for diagnosis in top_diagnoses:
            clean_diagnosis = re.sub(r'[^a-zA-Z0-9]', '_', diagnosis)
            pattern = re.escape(str(diagnosis))
            df_copy[f'Tani_{clean_diagnosis}'] = df_copy['Tanilar'].str.contains(pattern, na=False).astype(int)
        df_copy['Tani_Sayisi'] = df_copy['Tanilar'].apply(
            lambda x: len(x.split(',')) if pd.notna(x) else 0
        )
    
    # UygulamaYerleri için işlemler
    if 'UygulamaYerleri' in df_copy.columns:
        app_areas = df_copy['UygulamaYerleri'].dropna().str.split(',').explode()
        app_areas = app_areas.str.strip()
        top_areas = app_areas.value_counts().head(8).index.tolist()
        for area in top_areas:
            clean_area = re.sub(r'[^a-zA-Z0-9]', '_', str(area))
            pattern = re.escape(str(area))
            df_copy[f'Uygulama_{clean_area}'] = df_copy['UygulamaYerleri'].str.contains(pattern, na=False).astype(int)
        df_copy['Uygulama_Yeri_Sayisi'] = df_copy['UygulamaYerleri'].apply(
            lambda x: len(x.split(',')) if pd.notna(x) else 0
        )
    
    return df_copy




# -------------------------
# 3) EDA
# -------------------------
def eda(df):
    print(df.shape)
    print(df.head(10))
    print(df.info())

    # numeric değerlere dönüşüm
    df["TedaviSuresi"] = df["TedaviSuresi"].str.extract(r'(\d+)').astype(int)
    df["UygulamaSuresi"] = df["UygulamaSuresi"].str.extract(r'(\d+)').astype(int)

    print(df.describe())


    # Tedavi Süresi dağılımı
    plt.figure(figsize=(10,6))
    sns.histplot(df['TedaviSuresi'], bins=30, kde=True, color='skyblue')
    plt.title('Tedavi Süresi Dağılımı')
    plt.xlabel('Tedavi Süresi (dk)')
    plt.ylabel('Frekans')
    plt.savefig('TedaviSuresi_Dagilimi.png', dpi=200)
    plt.show()

    # Boxplot ile outlier analizi
    plt.figure(figsize=(10,6))
    sns.boxplot(x=df['TedaviSuresi'], color='lightgreen')
    plt.title('Tedavi Süresi Boxplot')
    plt.xlabel('Tedavi Süresi (dk)')
    plt.savefig('TedaviSuresi_Boxplot.png', dpi=200)
    plt.show()

        # Tedavi Süresi outlier analizi
    Q1_tedavi = df['TedaviSuresi'].quantile(0.25)
    Q3_tedavi = df['TedaviSuresi'].quantile(0.75)
    IQR_tedavi = Q3_tedavi - Q1_tedavi
    outliers_tedavi = df[(df['TedaviSuresi'] < (Q1_tedavi - 1.5 * IQR_tedavi)) | 
                        (df['TedaviSuresi'] > (Q3_tedavi + 1.5 * IQR_tedavi))]
    print(f"Tedavi Süresi outlier sayısı: {len(outliers_tedavi)}")


    
    # Uygulama Süresi ile karşılaştırma
    plt.figure(figsize=(10,6))
    sns.scatterplot(x='UygulamaSuresi', y='TedaviSuresi', data=df, hue='TedaviAdi', palette='tab10')
    plt.title('Uygulama Süresi vs Tedavi Süresi')
    plt.xlabel('Uygulama Süresi (dk)')
    plt.ylabel('Tedavi Süresi (dk)')
    plt.legend(title='Tedavi Adı', fontsize=8, title_fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('Uygulama_vs_TedaviSuresi.png', dpi=200)
    plt.show()


    # Eksik değer analizi
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    missing_df = pd.DataFrame({
        'missing_count': df.isnull().sum(),
        'missing_percentage': missing_percentage
    })

    missing_df = missing_df[missing_df['missing_count'] > 0]
    print("Eksik değer analizi:")
    print(missing_df.sort_values('missing_percentage', ascending=False))

    # Eksik değer görselleştirme
    plt.figure(figsize=(12,8))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Eksik Değerler Grafiği')
    plt.savefig('Eksik_Değerler_Grafiği.png', dpi=200)
    plt.show()

    # Yaş dağılımı
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Yas'], bins=30, kde=True)
    plt.title('Yaş Dağılımı')
    plt.xlabel('Yaş')
    plt.ylabel('Frekans')
    plt.savefig('Yas_Dagilimi.png', dpi=200)
    plt.show()

    # Outlier analizi
    Q1_yas = df['Yas'].quantile(0.25)
    Q3_yas = df['Yas'].quantile(0.75)
    IQR_yas = Q3_yas - Q1_yas
    outliers_yas = df[(df['Yas'] < (Q1_yas - 1.5 * IQR_yas)) | 
                      (df['Yas'] > (Q3_yas + 1.5 * IQR_yas))]
    print(f"Yaş outlier sayısı: {len(outliers_yas)}")


     # Cinsiyet dağılımı
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Cinsiyet', data=df, palette='pastel')
    plt.title('Cinsiyet Dağılımı')
    plt.xlabel('Cinsiyet')
    plt.ylabel('Frekans')
    plt.savefig('Cinsiyet_Dagilimi.png', dpi=200)
    plt.show()

     # KanGrubu dağılımı
    plt.figure(figsize=(10, 6))
    sns.countplot(x='KanGrubu', data=df, palette='pastel')
    plt.title('KanGrubu Dağılımı')
    plt.xlabel('KanGrubu')
    plt.ylabel('Frekans')
    plt.savefig('KanGrubu_Dagilimi.png', dpi=200)
    plt.show()

    # Uyruk dağılımı
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Uyruk', data=df, palette='pastel')
    plt.title('Uyruk Dağılımı')
    plt.xlabel('Uyruklar')
    plt.ylabel('Frekans')
    plt.savefig('Uyruk_Dagilimi.png', dpi=200)
    plt.show()


    # Korelasyon
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(10,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()



# 4) PREPROCESSING

def preprocess(df):
    # String sütunlardaki boşlukları temizle
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"": np.nan, "none": np.nan, "nan": np.nan, "NaN": np.nan})

    # Hasta bazlı doldurma
    patients_cols = ["Cinsiyet", "KanGrubu", "KronikHastalik", "Alerji"]
    for col in patients_cols :
        if col in df.columns:
            df[col] = df.groupby("HastaNo")[col].transform(lambda x: x.ffill().bfill())
            if col in ["KronikHastalik", "Alerji"]:
                df[col] = df[col].fillna("Bilinmiyor")
            else:
                if df[col].isnull().any():
                    imputer = SimpleImputer(strategy='most_frequent')
                    df[col] = imputer.fit_transform(df[[col]]).ravel()

    # Diğer kolonlar
    other_cols = ["Bolum", "UygulamaYerleri", "Tanilar"]
    for col in other_cols:
        if col in df.columns and df[col].isnull().any():
            imputer = SimpleImputer(strategy='most_frequent')
            df[col] = imputer.fit_transform(df[[col]]).ravel()

    # Kategorik değerleri normalize et
    label_cols = ['TedaviAdi', 'KronikHastalik', 'Alerji']
    for col in label_cols:
        df[col] = df[col].str.lower().str.strip()

    # One-hot encoding
    one_hot_cols = ['Cinsiyet', 'KanGrubu', 'Uyruk', 'Bolum']
    df = pd.get_dummies(df, columns=one_hot_cols , dtype=int)

    # Label encoding
    label_cols = ['TedaviAdi', 'KronikHastalik', 'Alerji', 'UygulamaYerleri', 'Tanilar']
    for col in label_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])


    return df



# 5) KAYDETME

def save_data(df, filename="PusulaAcademy_CaseStudy_Data.xlsx"):
    df.to_excel(filename, index=False, engine='openpyxl')
    print(f" Preprocessed data saved as {filename}")


# 6) MAIN PIPELINE

def main():
    df = load_data("Talent_Academy_Case_DT_2025.xlsx") 

    eda(df)                                           
    
    df = preprocess_text_columns(df) 
    
    df_clean = preprocess(df)                         

    save_data(df_clean)                               


if __name__ == "__main__":
    main()
