import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Next Word AI",
    page_icon="🧠",
    layout="centered"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.html("""
<style>

.stApp {
    background: #0e1117;
}

.block-container {
    max-width: 850px;
    padding-top: 3rem;
    padding-bottom: 3rem;
}


/* ================= HERO ================= */

.hero {
    text-align: center;
    padding: 20px 0 35px 0;
}

.hero-icon {
    font-size: 60px;
}

.hero-title {
    font-size: 48px;
    font-weight: 800;
    margin-bottom: 8px;
    color: white;
}

.hero-subtitle {
    font-size: 18px;
    color: #a7adb8;
    margin-bottom: 20px;
}


/* ================= INPUT ================= */

.input-title {
    font-size: 18px;
    font-weight: 600;
    color: white;
    margin-bottom: 8px;
}


/* ================= PREDICTION CARD ================= */

.prediction-card {
    background: linear-gradient(
        135deg,
        #132b22,
        #163d2e
    );

    border: 1px solid #2d8a62;
    border-radius: 18px;

    padding: 28px;
    margin-top: 25px;

    text-align: center;

    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
}

.prediction-label {
    color: #9be7c1;
    font-size: 16px;
    margin-bottom: 8px;
}

.prediction-word {
    color: #4ade80;
    font-size: 38px;
    font-weight: 800;
}

.confidence {
    color: #b7c4bd;
    margin-top: 8px;
    font-size: 14px;
}


/* ================= SECTION HEADINGS ================= */

.section-title {
    font-size: 20px;
    font-weight: 700;
    color: white;
    margin-top: 30px;
    margin-bottom: 12px;
}


/* ================= INFO CARDS ================= */

.info-card {
    background: #171b23;
    border: 1px solid #292f3a;

    border-radius: 14px;

    padding: 18px;

    color: #c7ccd5;

    margin-bottom: 10px;
}


/* ================= FOOTER ================= */

.footer {
    text-align: center;

    color: #737b88;

    font-size: 13px;

    margin-top: 50px;
}

</style>
""")


# =========================================================
# LOAD MODEL + TOKENIZER
# =========================================================

@st.cache_resource
def load_resources():

    model = load_model("lstm_model.h5")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)

    return model, tokenizer, max_len


model, tokenizer, max_len = load_resources()


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_next_word(text):

    # -----------------------------------------------------
    # 1. Convert text → token IDs
    # -----------------------------------------------------

    sequence = tokenizer.texts_to_sequences([text])[0]

    # -----------------------------------------------------
    # 2. Padding
    # -----------------------------------------------------

    sequence = pad_sequences(
        [sequence],
        maxlen=max_len - 1,
        padding="pre"
    )

    # -----------------------------------------------------
    # 3. Model prediction
    # -----------------------------------------------------

    preds = model.predict(
        sequence,
        verbose=0
    )

    # -----------------------------------------------------
    # 4. Probability vector
    # -----------------------------------------------------

    probabilities = preds[0]

    # -----------------------------------------------------
    # 5. Best prediction
    # -----------------------------------------------------

    predicted_index = np.argmax(probabilities)

    # -----------------------------------------------------
    # 6. Confidence
    # -----------------------------------------------------

    confidence = probabilities[predicted_index]

    # -----------------------------------------------------
    # 7. Convert index → word
    # -----------------------------------------------------

    reverse_word_index = {
        index: word
        for word, index in tokenizer.word_index.items()
    }

    predicted_word = reverse_word_index.get(
        predicted_index,
        ""
    )

    # -----------------------------------------------------
    # 8. Top 5 predictions
    # -----------------------------------------------------

    top_indices = np.argsort(
        probabilities
    )[-5:][::-1]

    top_predictions = []

    for index in top_indices:

        if index in reverse_word_index:

            word = reverse_word_index[index]

            probability = probabilities[index]

            top_predictions.append(
                (word, probability)
            )

    return (
        predicted_word,
        confidence,
        top_predictions
    )


# =========================================================
# HERO SECTION
# =========================================================

st.html("""
<div class="hero">

    <div class="hero-icon">
        🧠
    </div>

    <div class="hero-title">
        Next Word AI
    </div>

    <div class="hero-subtitle">
        LSTM-powered next word prediction
    </div>

</div>
""")


# =========================================================
# INPUT SECTION
# =========================================================

st.html("""
<div class="input-title">
    ✍️ Write something...
</div>
""")


user_input = st.text_area(
    "",
    placeholder="Example: I think you are...",
    height=120
)


# =========================================================
# EXAMPLE INPUTS
# =========================================================

st.html("""
<div class="section-title">
    💡 Try an example
</div>
""")


col1, col2, col3 = st.columns(3)


with col1:

    example1 = st.button(
        "How are you",
        use_container_width=True
    )


with col2:

    example2 = st.button(
        "I think you are",
        use_container_width=True
    )


with col3:

    example3 = st.button(
        "I love",
        use_container_width=True
    )


# Example button actions

if example1:

    user_input = "How are you"


if example2:

    user_input = "I think you are"


if example3:

    user_input = "I love"


# =========================================================
# ACTION BUTTONS
# =========================================================

col1, col2 = st.columns([3, 1])


with col1:

    predict_button = st.button(
        "🔮 Predict Next Word",
        use_container_width=True,
        type="primary"
    )


with col2:

    clear_button = st.button(
        "🧹 Clear",
        use_container_width=True
    )


# Clear

if clear_button:

    st.rerun()


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    if user_input.strip() == "":

        st.warning(
            "Please enter some text first."
        )

    else:

        # Loading animation

        with st.spinner(
            "🧠 LSTM is thinking..."
        ):

            (
                next_word,
                confidence,
                top_predictions
            ) = predict_next_word(
                user_input
            )


        # =================================================
        # PREDICTION CARD
        # =================================================

        st.html(f"""
        <div class="prediction-card">

            <div class="prediction-label">
                ✨ Predicted Next Word
            </div>

            <div class="prediction-word">
                {next_word}
            </div>

            <div class="confidence">
                Model confidence:
                {confidence * 100:.2f}%
            </div>

        </div>
        """)


        # =================================================
        # TOP PREDICTIONS
        # =================================================

        st.html("""
        <div class="section-title">
            📊 Other likely words
        </div>
        """)


        for word, probability in top_predictions:

            st.write(
                f"**{word}** — "
                f"{probability * 100:.2f}%"
            )

            st.progress(
                float(probability)
            )


# =========================================================
# HOW IT WORKS
# =========================================================

st.html("""
<div class="section-title">
    ⚙️ How it works
</div>


<div class="info-card">

    <b>1️⃣ Tokenization</b>

    <br><br>

    Your sentence is converted into numerical token IDs.

</div>


<div class="info-card">

    <b>2️⃣ LSTM Processing</b>

    <br><br>

    The LSTM processes the sequence and uses the
    previous context to predict the next word.

</div>


<div class="info-card">

    <b>3️⃣ Probability Prediction</b>

    <br><br>

    The model calculates probabilities for possible
    next words.

</div>


<div class="info-card">

    <b>4️⃣ Final Prediction</b>

    <br><br>

    The word with the highest probability is selected.

</div>
""")


# =========================================================
# FOOTER
# =========================================================

st.html("""
<div class="footer">

    Built with ❤️ using TensorFlow, LSTM & Streamlit

    <br><br>

    🤖 Next Word Prediction Model

</div>
""")