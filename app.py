import tensorflow as tf
import streamlit as st
import numpy as np

st.set_option('deprecation.showfileUploaderEncoding', False)
model=tf.keras.models.load_model('final.h5')

# title
st.title('Brain Hemmorahage Detection')

# Taking input image
fil=st.file_uploader('',type=['jpg','png','jepg'])


def func():
    #displaying image back on webapp
    '''file_bytes = np.asarray(bytearray(fil.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    st.image(opencv_image, channels="RGB")'''

    # Making prediction
    image=tf.keras.preprocessing.image.load_img(fil,target_size=(130,130))

    img=tf.keras.preprocessing.image.img_to_array(image)
    img=np.expand_dims(img,[0])
    img=img/255

    ans=model.predict(img)

    return ans

if fil is not None:
    result=func()
    if result[0][0]>0.5:
        st.write('**Positive : **',round(result[0][0],2))
    else:
        st.write('Negative')

else:
    st.write('Please Upload an Image')