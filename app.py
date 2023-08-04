import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Define the predict_caption function
def predict_caption(filename):
    # Your code for predicting the caption goes here
    gru_state = tf.zeros((1, ATTENTION_DIM))

    img = tf.image.decode_jpeg(tf.io.read_file(filename), channels=IMG_CHANNELS)
    img = tf.image.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img = img / 255

    features = encoder(tf.expand_dims(img, axis=0))
    dec_input = tf.expand_dims([word_to_index("<start>")], 1)
    result = []
    for i in range(MAX_CAPTION_LEN):
        predictions, gru_state = decoder_pred_model(
            [dec_input, gru_state, features]
        )

        # draws from log distribution given by predictions
        top_probs, top_idxs = tf.math.top_k(
            input=predictions[0][0], k=10, sorted=False
        )
        chosen_id = tf.random.categorical([top_probs], 1)[0].numpy()
        predicted_id = top_idxs.numpy()[chosen_id][0]

        result.append(tokenizer.get_vocabulary()[predicted_id])

        if predicted_id == word_to_index("<end>"):
            return img, result

        dec_input = tf.expand_dims([predicted_id], 1)

    return img, result

# Create the Streamlit app
def main():
    st.title("Image Captioning App")
    
    # Add an image upload option
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = tf.image.decode_jpeg(uploaded_file.read(), channels=3)
        plt.imshow(image)
        plt.axis("off")
        st.pyplot()
        
        # Add a button to generate the caption
        if st.button("Generate Caption"):
            # Call the predict_caption function with the uploaded image
            caption = predict_caption(uploaded_file)
            
            # Display the generated caption
            st.write("Generated Caption:", caption)
    
    # Add a button to upload a new image
    if st.button("Upload New Image"):
        uploaded_file = None

if __name__ == "__main__":
    main()
