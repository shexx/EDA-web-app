try:
    import streamlit as st
    import textwrap
    from multiapp import MultiApp
    from pages import HomePage, Instruction
    from PIL import Image
    import base64
    from pathlib import Path
    print("All modules loaded")
except Exception as e:
    print("Some Modules are missing :{} ".format(e))

app = MultiApp()
# st.title("""
# Exploratory Data Analysis App
# """)
st.set_page_config(layout="centered")
def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded
# header = "https://www.dropbox.com/s/ema92dcf7epihid/PointEstimationLogo.PNG?dl=0"

# header_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
#     img_to_bytes(header)
# )
# st.markdown(
#     header_html, unsafe_allow_html=True,
# )

st.write("""

<h1 style="color:LightSeaGreen"><b>Exploratory Data Analysis App</b></h1>

*In this app, we will be building a exploratory data analysis. The app will aim to run a dataset through all the possible EDA steps.*

""", unsafe_allow_html=True)

string = """EDA aims to understand the dataset, identify the missing values & outliers if any using visual and quantitative methods to get a sense of the story it tells"""

username = st.text_input("Enter your username", type="password")
password = st.text_input("Enter a password", type="password")

if password == '' and username == '':
    st.write("Type your login details")
elif password == '':
    st.write("Type your password")
elif username == '':
    st.write("Type your username")
elif (password != 'password1'):
    st.write("Wrong Password")
elif (username != 'poweruser'):
    st.write("Wrong Username")
elif (username == 'poweruser' and password == 'password1'):
    st.write('---')

    # wrapper = textwrap.TextWrapper(width=85)
    # wordString = wrapper.fill(text=string)

    # st.write(wordString)


    app.add_app("Instruction", Instruction.app)
    app.add_app("Home", HomePage.app)


    #The main app
    app.run()