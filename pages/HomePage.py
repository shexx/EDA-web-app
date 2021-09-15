#import enum
from operator import index
from matplotlib.pyplot import axis, margins
from numpy import float64, int64
from numpy.core import numeric
from pkg_resources import normalize_path
from scipy.stats import contingency
try:
    # import os
    import streamlit as st
    # import textwrap
    # import sys
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import io
    # import plotly as p
    import plotly.express as px
    # import pandas_profiling as pf
    import scipy.stats as stats
    from sklearn import datasets
    from sklearn.experimental import enable_iterative_imputer
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from sklearn import preprocessing
    # from sklearn.preprocessing import StandardScaler 
    from sklearn.impute import IterativeImputer
    from sklearn.impute import SimpleImputer
    from sklearn.impute import KNNImputer
    from scipy.stats import chi2_contingency
    #from scipy.stats import chi2
    print("All modules loaded")
    sns.set_style(style= 'whitegrid')
except Exception as e:
    print("Some Modules are missing: {} ".format(e))


STYLE = """
<style>
img{
}
</style>
"""

def app():
    """ Run this function to display the streamlit app"""
    #st.info(__doc__)
    
    st.write('This is the `home page` of this multi-page app.')

    # Add a sidebar
    st.sidebar.subheader("App Settings")
    st.markdown(STYLE, unsafe_allow_html=True)

    file =  st.sidebar.file_uploader(label="Upload data", type=["csv", "png", "jpg", "xlsx"])
    show_file = st.empty()
    


    if not file:
        show_file.info("Please upload file: {}".format(' '.join(["csv", "png", "jpg", "xlsx"])))
        return
    content = file.getvalue()
    
    @st.cache
    def load_data():
        data = pd.read_csv(file)
        return data

    if isinstance(file, io.BytesIO):
        df = load_data()
        st.subheader("Data")
        responseVar = st.sidebar.multiselect("Enter Response Variable",df.columns)

           
        with st.beta_expander('Show/Hide DataFrame'):
                st.dataframe(df)
                
        if st.checkbox("Header only"):
            st.write("###### Enter number of rows to view")
            rows = st.number_input("", min_value=0, value=6)

            if rows > 0:
                st.dataframe(df.head(rows))
        

        #Descriptive statistic of numerical/categorical variables
        
        st.subheader("EDA 1: descriptive analysis")
        
        with st.beta_expander("Show descriptive analysis"):
            # dim = st.radio("Select a dimension", ("Rows","Columns","All"))
            col1,col2,col3 = st.beta_columns(3)
            
            #Show dimensions
            col1.text("Total Rows:")
            col1.write(df.shape[0])
            
        
            col2.text("Total Columns:")
            col2.write(df.shape[1])
        
            col3.text("Shape of dataset:")
            col3.write(df.shape)

            st.text("Types of variables")
            st.write(df.dtypes)
            
            # change data type for each column
            
            st.text("Data description (Categorcal and Numerical)")
            st.write(df.describe(include='O'))


            st.text("Data description (Numerical)")
            st.write(df.describe())


        # Non-graphical univariate analysis -  checking for unique values, checking for null values
        st.subheader("EDA 2: Univariate Analysis")
        with st.beta_expander("Non-graphical Univariate analysis"):
            
            null_data =  df.columns[df.isnull().any()].tolist()

            coly, colx = st.beta_columns(2)
            colx.text("Columns with missing values")
            colx.table(null_data)

            coly.text("Missing value count per column")
            coly.write(df.isnull().sum())

            st.subheader('Missing Value Imputation options')
            st.write("""  """) #Adding a gap of one line

            imputation = ['Choose One', 'Remove Null', 'Single Imputation-Mean', 'Single Imputation-Median', 
            'Single Imputation-Most Frequent', 'Multiple Imputation', 'Nearest Neighbour'] #types of imputation
            imputation_selectbox = st.selectbox("Choose imputation type",imputation)
            
            cat_columns = []
            num_columns = []
            for c in df.columns:
                if df[c].map(type).eq(str).any(): #check if there are strings in column
                    cat_columns.append(c)
                else:
                    num_columns.append(c)

            #create two dataframes for each data type
            df_numeric = df[num_columns]
            df_categorical = pd.DataFrame(df[cat_columns])

            if imputation_selectbox == imputation[1]:
                df = df.dropna()
                st.write(df.isna().sum())

            elif imputation_selectbox == imputation[2]:
                df = df.replace(r'^\s*$', np.nan, regex=True)
                S_imp = SimpleImputer(missing_values=np.NaN, strategy='mean')
                df_numeric = pd.DataFrame(S_imp.fit_transform(df_numeric), columns=df_numeric.columns)
                
                 #assign the same column names


            elif imputation_selectbox == imputation[3]:
                df = df.replace(r'^\s*$', np.nan, regex=True)
                S_imp = SimpleImputer(missing_values=np.NaN, strategy='median')
                df_numeric = pd.DataFrame(S_imp.fit_transform(df_numeric), columns=df_numeric.columns)


            elif imputation_selectbox == imputation[4]:
                df = df.replace(r'^\s*$', np.nan, regex=True)
                S_imp = SimpleImputer(missing_values=np.NaN, strategy='most_frequent')
                df_categorical = pd.DataFrame(S_imp.fit_transform(df_categorical), columns=df_categorical.columns)
                #assign the same column names


            elif imputation_selectbox == imputation[5]:
                df = df.replace(r'^\s*$', np.nan, regex=True)
                M_imp = IterativeImputer(missing_values=np.NaN, max_iter=10, random_state=0)
                df = pd.DataFrame(M_imp.fit_transform(df))
                df.columns = df.columns #assign the same column names


            elif imputation_selectbox == imputation[6]:
                df = df.replace(r'^\s*$', np.nan, regex=True)
                knn_imp = KNNImputer(missing_values=np.NaN, n_neighbors=5, weights="uniform")
                df_numeric = pd.DataFrame(knn_imp.fit_transform(df_numeric), columns=df_numeric.columns)

            #for categorical, Simple Imputer uses 'most_frequent' & 'constant' imputation for dtype = 'category'
            df_joined = pd.concat([df_numeric,df_categorical], axis = 1)
            if st.button("Show Imputed dataframe"):   
                
                st.write(df_joined.head())
                if sum(df_joined.isnull().sum()) == 0:
                    st.text("There are no more missing values in the dataframe") 
                else:
                    st.text("Missing Values are present")
                st.write("---")
            
            
        
            

        with st.beta_expander("Check for outliers"):
            numericData = df.select_dtypes(exclude = [object]).columns
            selectVar = st.multiselect("Enter a variable", df[numericData].columns, key=1)
            
            if len(selectVar)>1:
                st.text("You can only select one variable")
            elif len(selectVar)==0:
                st.text("Please enter a variable")
            else:
                fig, ax = plt.subplots(figsize=(10,8))
                ax.boxplot(df[selectVar])
                st.pyplot(fig)

        
        with st.beta_expander("Graphical Univariate Analysis"):
                #catData = df.select_dtypes(exclude = [int64]).columns
                catData2 = [col for col in df.columns if len(df[col].unique()) >= 2 and len(df[col].unique()) <= 4]
                catTable = df.filter(catData2, axis=1)

                #count plots
                
                fig = plt.figure(figsize=(75,25))
                fig.subplots_adjust(wspace=0.5, hspace=1.5)
                
                Y = catTable.columns
                Ychoice = st.selectbox('Select Y variable', Y)
                X = catTable.columns
                Xchoice = st.selectbox('Select X variable', X)
            
                fig  = px.bar(df, x=Xchoice, y=Ychoice, barmode="relative")
                st.plotly_chart(fig)

        
        st.subheader("EDA 3: Bivariate Analysis")
        with st.beta_expander("Point-Biserial Correlation"):
        
        
            selectD = responseVar
            #st.multiselect("Enter variable to dummify",df.columns)
            # dummify variable if categorical
            dummify = pd.get_dummies(df, columns=selectD, drop_first=True)
            if st.button("Dummify the response variable"):
                st.text("Dummy resposne variable is the last column on the dataset ")
                st.write(dummify.head(6))
            st.write('---')
                
            selectY = st.multiselect("Enter X variable",dummify.columns)


            selectNum = st.multiselect("Enter Y variable", dummify.columns, key=2)
            if st.button("show biserial correlation"):
                biserialCor = stats.pointbiserialr(dummify[selectY].squeeze(), dummify[selectNum].squeeze())
                st.write(biserialCor)


        st.subheader("EDA 4: Multicolinearity")
        pd.options.display.float_format = "{:,.1f}".format

        makeDf = st.multiselect("Select variables", df.columns, key=3)
        
        newDf = df[makeDf]
        
        with st.beta_expander("Correlation Matrix"):
            
            if st.button("Calculate correlation matrix"):
                fig,ax = plt.subplots()
                sns.heatmap(newDf.corr(), annot=True, cmap=plt.cm.Blues, ax=ax)
                st.write(fig)

        st.write("""  """) #Adding a gap of one line

        with st.beta_expander("Variance Inflation Factor"):
            #standardise numerical variables
            vif = pd.DataFrame()
            if st.checkbox("show VIF calculation (non-standardised)"):
                # newDf = preprocessing.scale(newDf) #standardising only predictors
                # newDf = pd.DataFrame(newDf_std)
                #newDf_std = newDf_std.rename(columns=makeDf)
                st.write(newDf.head(3))
                vif["VIF Factor"] = [variance_inflation_factor(newDf.values, i)
                for i in range(newDf.shape[1])]

                vif["features"] = newDf.columns
                st.write(vif.round(1))
            else:
                st.text("Waiting for predictor variables...")
        
            if st.checkbox("show VIF calculation (Standardised)"):
                newDf_std = preprocessing.scale(newDf) #standardising only predictors
                newDf_std = pd.DataFrame(newDf_std)
                #newDf_std = newDf_std.rename(columns=makeDf)
                st.write(newDf_std.head(3))
                vif["VIF Factor"] = [variance_inflation_factor(newDf_std.values, i)
                for i in range(newDf_std.shape[1])]

                vif["features"] = newDf_std.columns
                st.write(vif.round(1))
            else:
                st.text("Waiting for predictor variables...")       
        st.write("""  """) #Adding a gap of one line

        #Chi-sq test
        st.subheader("EDA 5: Relationship between Predictor and Response variables")
        with st.beta_expander("Chi-square testing"):

            
            chisqX = st.multiselect("Enter Predictor Variable", df_categorical.columns)
            if len(chisqX)>1:
                st.text("You can only select one variable")
            
            elif st.checkbox("show Chi-sq calculation"):
                chiY = responseVar[0]
                chiX = chisqX[0]
                
                # see chi-sq calc for one variable
                def chisqtest(x, y=df_categorical[chiY]):
                    chisqtest = pd.crosstab(y,x, margins=True)
                    value = np.array([chisqtest.iloc[0][0:5].values, chisqtest.iloc[1][0:5].values]) 
                    chi, p, dof = chi2_contingency(value)[0:3]
                    format(chi,'.10f'),
                    format(p,'.10f'),
                    format(dof,'.10f')
                    chiResult = [chi,"{:e}".format(p),dof]
                    chidf = pd.DataFrame(chiResult, index=["Chi", "P-value", "Degree of Freedom"], columns=[chiX])
                    
                    return st.table(chidf)
                chisqtest(df_categorical[chiX])

                # see chi-sq calc for all variables
                data = [col for col in df.columns if len(df[col].unique()) >= 2 and len(df[col].unique()) <= 4 
                     and col != responseVar[0]
                     and col != 'SubscriptionStatus'
                     ]

                st.write(data)
                def chisqtestAll(x, y=df_categorical[chiY]):
                    chisqtest = pd.crosstab(y,x, margins=True)
                    value = np.array([chisqtest.iloc[0][0:5].values, chisqtest.iloc[1][0:5].values]) 
                    chi, p, dof = chi2_contingency(value)[0:3]
                    format(chi,'.10f'),
                    format(p,'.10f'),
                    format(dof,'.10f')
                    chiResult = (chi,"{:e}".format(p),dof)
                 
                
                #chisqtestAll(data)
                #chiRes =  df_categorical.apply(lambda column: chisqtestAll(df_categorical[column]))
                

                
            
            



        

            




        



    
    



    


