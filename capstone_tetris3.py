
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pickle

#from wordcloud import WordCloud
#from numerize import numerize

st.set_page_config(layout='wide')

github_url = "https://github.com/snrhlz01/category_predict/raw/main/combine_table_withzero.csv"

# Membaca dataset dari URL
df = pd.read_csv(github_url)

#print(df.head())
# Sidebar menu
st.sidebar.header("Menu")
selected_menu = st.sidebar.radio("Selected Menu :", ["Home", "Data Visualization", "Rank Companies by Stars", "Category Prediction (Machine Learning)"])

# Konten utama berdasarkan menu yang dipilih
if selected_menu == "Home":
    # Image
    st.title(r"Job Market Explorer: Unveiling Opportunities in the Data Science Domain")
    st.write("[Github](https://github.com/snrhlz01)")
    st.write("[LinkedIn](https://id.linkedin.com/in/salma-nurhaliza-pribadi-456a63195)")
    st.write("[Instagram](https://www.instagram.com/snrhlz/)")
    from PIL import Image

    # image_path = "C:/Users/Salma/Downloads/jobstreet logo.png"
    # image = Image.open(image_path)
    # st.image(image, width=500)

    st.subheader("Problem and User")
    with st.expander("Problem"):
        st.markdown(
    "80 persen mahasiswa di Indonesia bekerja tidak sesuai dengan jurusan kuliahnya diungkapkan oleh [Menteri Pendidikan, Kebudayaan, Riset dan Teknologi (Mendikbud Ristek) Nadiem Makarim](https://www.kompas.com/edu/read/2022/11/07/090548371/80-persen-mahasiswa-bekerja-tidak-sesuai-jurusan-ini-4-alasannya?page=all) sebagaimana mereka memiliki passion yang berbeda dengan jurusan kuliahnya dan pekerjaan yang dimilikinya. Dari [Monika dkk.](https://lintar.untar.ac.id/repository/pengabdian/buktiabdi_10709001_2B050722154011.pdf), ketika seseorang menjalankan pekerjaannya atau aktivitas yang ditekuninya sesuai dengan passion-nya, maka orang tersebut akan merasakan dorongan yang besar untuk terus berusaha dan berkarya dengan lebih baik lagi. Dengan demikian, hal ini penting dipahami oleh setiap orang, khususnya seseorang yang sedang merencanakan karir ke depan. Sebuah tantangan besar dalam mencocokkan kandidat yang tepat dengan peluang pekerjaan yang sesuai di sebuah bidang data, yang semakin penting seiring dengan pertumbuhan industri dan peningkatan permintaan akan data scientist yang terampil diambil dari [World Economic Forum](https://www3.weforum.org/docs/WEF_Data_Science_In_the_New_Economy.pdf). Proses ini kompleks karena pencari kerja perlu memahami dinamika perusahaan dan persyaratan pekerjaan yang spesifik, sementara perusahaan perlu menilai keterampilan dan kualifikasi calon karyawan, namun solusi yang memfasilitasi proses ini dan memberikan platform eksplorasi yang komprehensif bagi pencari kerja data scientist akan sangat bermanfaat dalam mengatasi masalah dunia nyata ini. Oleh karena itu, proyek ini bertujuan untuk menyediakan platform eksplorasi yang komprehensif bagi para pencari kerja Data Scientist."
    )

        
    with st.expander("User"):
        st.markdown(
            """
            1. Pencari Kerja di Bidang Data Science: Mereka yang mencari pekerjaan atau ingin mengeksplorasi peluang di bidang Data Science.
            2. Perusahaan: Untuk mendapatkan wawasan tentang bagaimana mereka dinilai oleh para pencari kerja dan bagaimana mereka dapat meningkatkan daya tarik mereka sebagai tempat kerja.
""" )

    with st.expander("Flowchart"):
        image_path = "C:/Users/Salma/Downloads/tetris_flow.png"
        image = Image.open(image_path)
        st.image(image, width=800)
        
    st.subheader("Data Collection")
    st.markdown(
    "Kumpulan data yang digunakan dalam analisis ini dikumpulkan dari [JobStreet API](https://jobstreet.com/). Data tersebut mencakup daftar pekerjaan yang terkait dengan 'Data Scientist' dengan parameter Job ID, title, requirements, location, category, subcategory, date upload, job type, salary, company, dan company rating"
    )

    st.subheader("Data Visualization")
    st.markdown(
        "Dalam pengalaman visualisasi data ini, menempatkan fokus utama pada analisis mendalam dari sepuluh judul pekerjaan teratas, sepuluh perusahaan teratas, serta distribusi kategori, subkategori, dan lokasi dalam dataset pekerjaan. Dengan mengeksplorasi informasi ini secara rinci, Anda akan dapat memperoleh wawasan yang lebih mendalam dan relevan terhadap tren dalam dunia pekerjaan Data Scientist saat ini."
        )


    st.subheader("Rank Companies by Star")

    st.markdown(
        "Wawasan tentang kinerja perusahaan melalui fitur 'Rank Companies by Star'. Evaluasi dan bandingkan perusahaan berdasarkan rating bintang mereka, yang berasal dari kriteria khusus dan metrik kinerja. Fitur ini menyederhanakan proses penelitian perusahaan, memberikan gambaran cepat tentang perusahaan terbaik. Buat keputusan yang terinformasi tentang calon perusahaan dan temukan perusahaan yang paling cocok untuk aspirasi karier di bidang Data Scientist.")

    st.subheader("Category Prediction from Job Requirements using Machine Learning")

    st.markdown(
        "Fitur 'Category Prediction from Job Requirements' yang didukung oleh machine learning dari model logistic regression dan random forest. Masukkan kriteria yang sekiranya anda miliki, dan biarkan model machine learning kami memprediksi kategori pekerjaan yang paling relevan. Fitur ini menyederhanakan proses kategorisasi pekerjaan, memberikan klasifikasi yang cepat dan cukup akurat."
        )

# VISUALISASI DATA
elif selected_menu == "Data Visualization":


    def plot_pie_chart(data, text, figure_size=(8, 8), text_size=12):
        st.write(f"### {text}")
        fig = px.pie(data, names=data.index, values=data.values, 
                    labels={'values': 'Count'}, hole=0.4, 
                    height=figure_size[1] * 100, width=figure_size[0] * 100)
        fig.update_traces(textinfo='percent+label', pull=[0.1] * len(data), textfont_size=text_size)
        st.plotly_chart(fig)

    # Custom function for autopct to display both percentage arrows and values
    def func(pct, allvalues):
        absolute = int(pct / 100.*sum(allvalues))
        return f"{pct:.1f}%\n({absolute})"


    import plotly.express as px

    # Function to plot bar chart
    def plot_bar_chart(data, title, x_label, y_label):
        st.write(f"### {title}")
        fig = px.bar(data, x=data.index, y=data.values, labels={'y': y_label})
        fig.update_traces(marker_color='skyblue', hoverinfo='y+text', text=data.values)
        fig.update_layout(xaxis_title=x_label, yaxis_title=y_label, xaxis=dict(tickangle=-45))
        st.plotly_chart(fig)

    # Function to plot histogram
    def plot_histogram(data, title, x_label, y_label):
        st.write(f"### {title}")
        fig = px.histogram(data.str.len(), nbins=20, labels={'value': x_label}, color_discrete_sequence=['skyblue'])
        fig.update_layout(xaxis_title=x_label, yaxis_title=y_label)
        st.plotly_chart(fig)

    st.title("Top 10 Company and Job Distributions")
    st.write("Sumber: [Jobstreet](https://jobstreet.com/)")
    # Top 10 Job Titles
    top_job_titles = df['job_title'].value_counts()[:10]
    plot_bar_chart(top_job_titles, "Top 10 pekerjaan yang memiliki jumlah lowongan kerja terbanyak", "Job Title", "Count")
    st.write("Data analyst merupakan pekerjaan yang paling dibutuhkan karena memiliki jumlah lowongan kerja terbanyak")

    # Top 10 Companies
    top_companies = df['company'].value_counts()[:10]
    plot_bar_chart(top_companies, "Top 10 perusahaan yang membuka lowongan kerja terbanyak", "Company Name", "Count")
    st.write("PT. Merah Cipta Media memiliki jumlah lowongan kerja tertinggi dari total 18 lowongan kerja")

    # Job Description Length Histogram
    #plot_histogram(df['requirements'], "Job Description Length", "Job Description Length", "No. of Jobs")


    st.title("Top 5 Data Distributions")

    # Top 5 Category Distribution
    top_categories = df['category'].value_counts().head(5)
    plot_pie_chart(top_categories, "Distribusi kategori lowongan kerja terbanyak", figure_size=(8, 8), text_size=12)
    st.write("Science & technology dan information & communication technology adalah dua kategori terbanyak yang muncul dalam distribusi tersebut.")
    
    # Top 5 Subcategories Distribution
    top_subcategories = df['subcategory'].value_counts().head(5)
    plot_pie_chart(top_subcategories, "Distribusi subkategori lowongan kerja terbanyak", figure_size=(8, 8), text_size=12)
    st.write("Mathematics, Statistics, & Information Sciences adalah kategori terbanyak yang muncul dalam distribusi subkategori lowongan kerja terbanyak")
    
    # Top 5 Location Distribution
    top_locations = df['location'].value_counts().head(5)
    plot_pie_chart(top_locations, "Distribusi lokasi lowongan kerja terbanyak", figure_size=(8, 8), text_size=12)
    st.write("Secara garis besar, Jakarta adalah lokasi terbanyak pekerjaan untuk lowongan Data Scientist")



elif selected_menu == "Rank Companies by Stars":
    ## REMOVE THE 0 VALUE
    # Function to calculate review statistics
    def get_statistics(df, measure):
        reviews = [5 for i in range(int(df['5']))] \
                + [4 for i in range(int(df['4']))] \
                + [3 for i in range(int(df['3']))] \
                + [2 for i in range(int(df['2']))] \
                + [1 for i in range(int(df['1']))]

        if measure == 'mean':
            return np.round(np.mean(np.array(reviews)), 3)
        elif measure == 'std':
            return np.round(np.std(np.array(reviews), ddof=1), 3)
        elif measure == 'sum':
            return df['5'] + df['4'] + df['3'] + df['2'] + df['1']
        else:
            raise Exception('Not Implemented')

    # Calculate review statistics
    df['sum_review'] = df.apply(lambda x: get_statistics(x, 'sum'), axis=1)
    df['mean_review'] = df.apply(lambda x: get_statistics(x, 'mean'), axis=1)
    df['std_review'] = df.apply(lambda x: get_statistics(x, 'std'), axis=1)

    # Replace NULL values with 0
    df[['mean_review', 'std_review']] = df[['mean_review', 'std_review']].fillna(0)

    # Remove rows where both 'mean_review' and 'std_review' are 0
    df = df[(df['mean_review'] != 0) & (df['std_review'] != 0)]

    # WEIGHTED MEAN REVIEW
    def weighted_mean_review(df):
        R = df['mean_review']
        v = df['sum_review']
        m = 5
        C = 4.318
        return (v / (v+m)) * R + (m / (v+m)) * C

    df['weighted_mean_review'] = df.apply(lambda x: weighted_mean_review(x), axis=1)

    # Mengganti nilai NULL dengan 0
    df[['weighted_mean_review']] = df[['weighted_mean_review']].fillna(0)
    df = df[(df['weighted_mean_review'] != 0)]

    # Show duplicate rows based on the 'Company' column
    duplicate_rows = df[df.duplicated(subset='company', keep=False)]
    # print("Duplicate Rows:")
    # print(duplicate_rows)

    # Print the total count of duplicate rows
    total_duplicates = duplicate_rows.shape[0]
    # print(f"Total Duplicate Rows: {total_duplicates}")

    # Remove duplicate rows based on the 'company' column
    df_no_duplicates = df.drop_duplicates(subset='company', keep='first')


    # Sort the DataFrame by the desired metrics
    top_sum_review = df_no_duplicates.nlargest(10, 'sum_review')
    top_mean_review = df_no_duplicates.nlargest(10, 'mean_review')
    top_std_review = df_no_duplicates.nlargest(10, 'std_review')
    top_weighted_mean_review = df_no_duplicates.nlargest(10, 'weighted_mean_review')

    st.title("Top Company Review Analysis")

    # Calculate the total count for each star level
    total_5 = df['5'].sum()
    total_4 = df['4'].sum()
    total_3 = df['3'].sum()
    total_2 = df['2'].sum()
    total_1 = df['1'].sum()

    st.subheader("Total Star Ratings Given")

    total_5_star, total_4_star, total_3_star, total_2_star, total_1_star = st.columns(5)

    # Display total count for each star level as metrics
    with total_5_star:
        st.metric("Total 5 Stars", value=total_5)
    with total_4_star:
        st.metric("Total 4 Stars", value=total_4)
    with total_3_star:
        st.metric("Total 3 Stars", value=total_3)
    with total_2_star:
        st.metric("Total 2 Stars", value=total_2)
    with total_1_star:
        st.metric("Total 1 Star", value=total_1)


    # Allow users to select the metric
    selected_metric = st.selectbox("Select a Review Analysis", ["Sum Review", "Mean Review", "Standard Deviation Review", "Weighted Mean Review"])

    # Visualize the top companies with a larger figure size
    fig, ax = plt.subplots(figsize=(8, 8))

    # Bar plot for the selected metric
    if selected_metric == "Sum Review":
        # Display Sum Review Formula with explanation
        st.subheader("Sum Review Formula:")
        st.markdown("The sum review is calculated by multiplying each rating count by its corresponding rating value and summing them up:")
        st.latex(r'\text{sum\_review} = 5 \cdot \text{count\_5} + 4 \cdot \text{count\_4} + 3 \cdot \text{count\_3} + 2 \cdot \text{count\_2} + 1 \cdot \text{count\_1}')

        st.write("Keterangan:")
        st.write("- count_5: Jumlah review dengan rating 5")
        st.write("- count_4: Jumlah review dengan rating 4")
        st.write("- count_3: Jumlah review dengan rating 3")
        st.write("- count_2: Jumlah review dengan rating 2")
        st.write("- count_1: Jumlah review dengan rating 1")
        bars = ax.bar(top_sum_review['company'], top_sum_review['sum_review'], color='gold')

    elif selected_metric == "Mean Review":
        # Display Mean Review Formula with explanation
        st.subheader("Mean Review Formula:")
        st.markdown("The mean review is the average rating across all reviews. Here, N represents the total number of reviews:")
        st.latex(r'\text{mean\_review} = \frac{1}{N} \sum_{i=1}^{N} \text{reviews}_i')
        
        st.write("Keterangan:")
        st.write(r"- $\text{mean\_review}$: Nilai rata-rata dari semua ulasan.")
        st.write(r"- $\frac{1}{N}$: Pembagian satu dengan jumlah total data ($N$), yang merupakan jumlah ulasan yang dianalisis.")
        st.write(r"- $\sum_{i=1}^{N}$: Penjumlahan dari ulasan pertama hingga ulasan ke-$N$.")
        st.write(r"- $\text{reviews}_i$: Ulasan untuk setiap data ke-$i$.")

        bars = ax.bar(top_mean_review['company'], top_mean_review['mean_review'], color='skyblue')

    elif selected_metric == "Standard Deviation Review":
        # Display Standard Deviation Review Formula with explanation
        st.subheader("Standard Deviation Review Formula:")
        st.markdown("The standard deviation review measures the dispersion of ratings around the mean review. Here, N represents the total number of reviews:")
        st.latex(r'\text{std\_review} = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N} (\text{reviews}_i - \text{mean\_review})^2}')

        st.write("Keterangan:")
        st.write(r"- $\text{std\_review}$: Standar deviasi dari semua ulasan.")
        st.write(r"- $\sqrt{\cdot}$: Simbol akar kuadrat, menghitung akar kuadrat dari nilai dalam tanda kurung.")
        st.write(r"- $\frac{1}{N-1}$: Pembagian satu dengan jumlah total data ($N$) dikurangi satu, yang merupakan derajat kebebasan dalam menghitung standar deviasi sample.")
        st.write(r"- $\sum_{i=1}^{N}$: Penjumlahan dari ulasan pertama hingga ulasan ke-$N$.")
        st.write(r"- $(\text{reviews}_i - \text{mean\_review})^2$: Kuadrat dari selisih antara setiap ulasan dan nilai rata-rata ulasan.")

        bars = ax.bar(top_std_review['company'], top_std_review['std_review'], color='darkblue')

    elif selected_metric == "Weighted Mean Review":
        # Display Weighted Mean Review Formula with explanation
        st.subheader("Weighted Mean Review Formula:")
        st.markdown("The weighted mean review considers both the average rating and a predefined constant C. Here, N represents the total number of reviews:")
        st.latex(r'\text{weighted\_mean\_review} = \frac{v}{v+m} \cdot R + \frac{m}{v+m} \cdot C')


        st.write("Keterangan:")
        st.write(r'$R$ adalah nilai rata-rata ulasan')
        st.write(r'$v$ adalah jumlah nilai ulasan')
        st.write(r'$m$ adalah konstan (misalnya, 5)')
        st.write(r'$C$ adalah konstan yang telah ditentukan sebelumnya (misalnya, 4.318)')

        bars = ax.bar(top_weighted_mean_review['company'], top_weighted_mean_review['weighted_mean_review'], color='lightgreen')

    # Highlight the top 1 company
    def is_top_company(df, metric, company):
        top_company = df[df['company'] == company]
        return top_company.index[0] == df.index[0]

    # Highlight the top company
    top_company_name = top_sum_review['company'].iloc[0]  # You can use any metric for the top company
    for bar, company in zip(bars, top_sum_review['company']):
        if is_top_company(top_sum_review, selected_metric, company):
            bar.set_color('green')  # Change color for the top 1
            break
    
    ax.set_title(f'Plot - {selected_metric}')
    ax.tick_params(axis='x', rotation=90)  # Rotate x-axis labels

    # Show the plot
    st.pyplot(fig)

    # Print the top companies for the selected metric
    st.subheader(f"Top 10 Companies - {selected_metric}:")
    if selected_metric == "Sum Review":
        st.dataframe(top_sum_review[['company', 'sum_review']])
    elif selected_metric == "Mean Review":
        st.dataframe(top_mean_review[['company', 'mean_review']])
    elif selected_metric == "Standard Deviation Review":
        st.dataframe(top_std_review[['company', 'std_review']])
    elif selected_metric == "Weighted Mean Review":
        st.dataframe(top_weighted_mean_review[['company', 'weighted_mean_review']])

    

    

elif selected_menu == "Category Prediction (Machine Learning)":

    # Load pre-trained models
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Load pre-trained models
    with open('./rf_model.pkl', 'rb') as rf_file:
        rf_model = pickle.load(rf_file)

    with open('./logistic_model.pkl', 'rb') as logistic_file:
        logistic_model = pickle.load(logistic_file)

    # Load the vectorizer used during training
    with open('./tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        tfidf_vectorizer = pickle.load(vectorizer_file)

    # Streamlit app
    def main():
        st.title("Category Prediction with job requirements")

        # Input text area for user
        user_input = st.text_area("Enter the job requirements text:", "")

        if st.button("Predict Category Encoded"):
            # Preprocess user input using the same vectorizer
            cleaned_input = tfidf_vectorizer.transform([user_input])

            # Make predictions
            rf_prediction = rf_model.predict(cleaned_input)
            logistic_prediction = logistic_model.predict(cleaned_input)

            # Display predictions
            st.write("Random Forest Prediction:", rf_prediction[0])
            st.write("Logistic Regression Prediction:", logistic_prediction[0])

    
    if __name__ == "__main__":
        main()

    category_mapping = {
        'Accounting': 0,
        'Administration & Office Support': 1,
        'Consulting & Strategy': 2,
        'Information & Communication Technology': 3,
        'Insurance & Superannuation': 4,
        'Manufacturing, Transport & Logistics': 5,
        'Mining, Resources & Energy': 6,
        'Retail & Consumer Products': 7,
        'Sales': 8,
        'Science & Technology': 9
    }

    st.subheader("Category Explanations:")
    for category, code in category_mapping.items():
        st.write(f"{category}: {code}")
        

    # Visualize total count of each category
    st.title("Category Counts")
    category_counts = df['category'].value_counts()
    #st.bar_chart(category_counts)

    # Interactive bar chart using Plotly Express
    fig = px.bar(x=category_counts.index, y=category_counts.values, labels={'x': 'Category', 'y': 'Count'},
                title='Category Counts')
    st.plotly_chart(fig)

    # REQUIREMENTS
    # Create a DataFrame with requirement counts
    requirements_counts = df['requirements'].value_counts().reset_index()
    requirements_counts.columns = ['requirements', 'count']

    # Visualize the bar chart using Streamlit
    st.title("Requirements Frequency")
    st.bar_chart(requirements_counts.set_index('requirements'))