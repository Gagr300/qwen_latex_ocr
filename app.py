import streamlit as st
from PIL import Image
from inference import MyModel

model = MyModel()
st.header('LaTeX math formula from image to text')
st.write('### Prompt')

uploaded_file = st.file_uploader(
    "**Choose an image**",
    type=['png', 'jpg', 'jpeg'],
    accept_multiple_files=False
)

user_prompt = '''Write the correct LaTeX expression for the formula on the image.'''

user_prompt_input = st.text_area("**Change the base prompt if you want to**", user_prompt, height=200, )

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.button("Get latex-formula"):
        image = Image.open(uploaded_file)
        ans = model.infenence(image)
        st.write('### Answer')
        st.write('#### in text format:')
        st.text(ans)
        st.write('#### in latex format:')
        st.latex(ans)
