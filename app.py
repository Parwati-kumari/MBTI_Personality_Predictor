import streamlit as st
import joblib
import re
import string
import matplotlib.pyplot as plt
import numpy as np
import nltk

# Ensure stopwords are downloaded
nltk.download('stopwords')

from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words("english"))
# Optional Lottie animation import
try:
    from streamlit_lottie import st_lottie
    import requests

    def load_lottie(url):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()

except Exception:
    def st_lottie(*args, **kwargs):
        return None
    def load_lottie(url): return None

# ------------------------------
# Load Model, Vectorizer, and Encoder
# ------------------------------
clf = joblib.load("personality_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
le = joblib.load("label_encoder.pkl")

STOPWORDS = set(stopwords.words("english"))

# ------------------------------
# MBTI Personality Info
# ------------------------------
mbti_info = {
    "INTJ": ("The Architect", "Imaginative, strategic thinkers with a plan for everything.", "Mastermind logo", "🧠", "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQQrBHrqnLOpm1a2rqVzp0CZCYvaQZHvUE6fw&s", ["Strategic", "Innovative", "Analytical"]),
    "INTP": ("The Logician", "Innovative inventors with an unquenchable thirst for knowledge.", "Thinker logo", "🔬", "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQU77nhwMwYB3bI-6x1yrFiQQ_Ft_8bPMZ7mXjd_-seheUpAk3h2dOyLK3iW7-G5A3g3iE&usqp=CAU", ["Analytical", "Curious", "Intellectual"]),
    "ENTJ": ("The Commander", "Bold, imaginative, and strong-willed leaders.", "Leader logo", "⚔️", "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTZs5ETg7-JPZ9ywP8KdjoJ4A2Z3TCu24Jalw&s", ["Confident", "Decisive", "Organized"]),
    "ENTP": ("The Debater", "Smart and curious thinkers who love intellectual challenges.", "Debater logo", "💡", "https://i.pinimg.com/474x/70/61/92/706192240fc2aac137dacbfd8c2a6e5f.jpg", ["Curious", "Energetic", "Innovative"]),
    "INFJ": ("The Advocate", "Quiet and mystical, yet inspiring and tireless idealists.", "Counselor logo", "🌿", "https://p7.hiclipart.com/preview/396/737/277/infj-personality-type-myers-briggs-type-indicator-personality-test-infj.jpg", ["Insightful", "Altruistic", "Idealistic"]),
    "INFP": ("The Mediator", "Poetic, kind, and altruistic, always eager to help a good cause.", "Healer logo", "💖", "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTp1D2KcUNw6DhmEanbej_7iDDcM-cefU6aRg&s", ["Empathetic", "Creative", "Idealistic"]),
    "ENFJ": ("The Protagonist", "Charismatic and inspiring leaders.", "Teacher logo", "🌟", "https://i.pinimg.com/474x/9e/fe/ee/9efeee00dd3de2f08464f2dd0e080df3.jpg", ["Charismatic", "Organized", "Empathetic"]),
    "ENFP": ("The Campaigner", "Enthusiastic, creative, and sociable free spirits.", "Champion logo", "🎨", "https://i.pinimg.com/236x/6a/5a/d0/6a5ad07acfec36e2df85195ff47afb00.jpg", ["Energetic", "Curious", "Imaginative"]),
    "ISTJ": ("The Logistician", "Practical and reliable individuals.", "Inspector logo", "📚", "https://i.pinimg.com/236x/3e/ff/d4/3effd4a1437af491adefa54936d611d7.jpg", ["Responsible", "Organized", "Practical"]),
    "ISFJ": ("The Defender", "Dedicated and warm protectors.", "Protector logo", "🏛️", "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSx39RFnut1MLvr2P1YSBsxCeDHg_7BPFm9eg&s", ["Caring", "Reliable", "Supportive"]),
    "ESTJ": ("The Executive", "Excellent administrators and organizers.", "Supervisor logo", "🛡️", "https://i.pinimg.com/474x/03/67/9e/03679e33c1dfc9cefd29d93f467541df.jpg", ["Efficient", "Organized", "Direct"]),
    "ESFJ": ("The Consul", "Caring and popular social personalities.", "Provider logo", "🤝", "https://i.pinimg.com/474x/5a/aa/6e/5aaa6e5ec42d3ad6872e35bffd5b0f36.jpg", ["Friendly", "Helpful", "Loyal"]),
    "ISTP": ("The Virtuoso", "Bold and practical experimenters.", "Craftsman logo", "🛠️", "https://ih1.redbubble.net/image.3927653212.1423/st,small,507x507-pad,600x600,f8f8f8.u2.jpg", ["Practical", "Observant", "Spontaneous"]),
    "ISFP": ("The Adventurer", "Flexible and charming artists.", "Artist logo", "🎸", "https://i.pinimg.com/474x/81/89/d0/8189d0e91ec4045d05384347a13a757b.jpg", ["Creative", "Adaptable", "Gentle"]),
    "ESTP": ("The Entrepreneur", "Energetic and perceptive risk-takers.", "Dynamo logo", "🏎️", "https://i.redd.it/5yt9n1xoff6f1.png", ["Bold", "Energetic", "Pragmatic"]),
    "ESFP": ("The Entertainer", "Spontaneous and enthusiastic people.", "Performer logo", "🎭", "https://i.pinimg.com/474x/a3/d1/e7/a3d1e71bbe3dba8fff471a9963643333.jpg", ["Outgoing", "Energetic", "Fun-loving"]),
}

# ------------------------------
# Helper Functions
# ------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in text.split() if t.isalpha() and t not in STOPWORDS]
    return " ".join(tokens)

def predict_personality(post):
    clean = clean_text(post)
    X = tfidf.transform([clean])
    probs = clf.predict_proba(X)[0]
    classes = le.classes_
    results = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
    return results

# ------------------------------
# Streamlit Page Configuration
# ------------------------------
st.set_page_config(page_title="MBTI Personality Predictor 🧠", page_icon="🧩", layout="wide")

# ------------------------------
# Custom CSS with larger flip-cards & hover scale
# ------------------------------
st.markdown("""
<style>
body, .stApp { background-color: #0D1321; color: #F0EBD8; }
h1,h2,h3,h4,h5 { color:#748CAB; }
.stTextArea textarea { background-color:#1D2D44;color:#F0EBD8;border-radius:10px;border:1px solid #748CAB; }
.stButton>button { background:linear-gradient(to right,#1D2D44,#748CAB);color:#F0EBD8;border-radius:10px;border:none;padding:0.6em 1.5em; transition:0.3s ease-in-out; }
.stButton>button:hover { background:linear-gradient(to right,#748CAB,#1D2D44); transform:scale(1.05); box-shadow:0px 0px 15px #748CAB;}
.personality-card { background-color:#1D2D44;border-radius:15px;padding:15px;margin-bottom:15px;border:1px solid #748CAB; box-shadow:0px 0px 10px rgba(240,235,216,0.2); color:#F0EBD8;}
.personality-card:hover { transform:scale(1.03); box-shadow:0px 0px 20px rgba(240,235,216,0.5); border:1px solid #F0EBD8; }

.flip-card { background-color: transparent; width: 240px; height: 340px; perspective: 1000px; margin-bottom:20px; transition: transform 0.3s; }
.flip-card-inner { position: relative; width: 100%; height: 100%; text-align: center; transition: transform 0.8s; transform-style: preserve-3d; }
.flip-card:hover { transform: scale(1.08); }
.flip-card:hover .flip-card-inner { transform: rotateY(180deg); }
.flip-card-front, .flip-card-back { position: absolute; width: 100%; height: 100%; backface-visibility: hidden; border-radius:15px; border:1px solid #748CAB; box-shadow:0px 0px 12px rgba(240,235,216,0.2); }
.flip-card-front { background-color:#F0EBD8; color:#0D1321; display:flex; flex-direction:column; align-items:center; justify-content:center; padding:10px; }
.flip-card-front img { width:160px; border-radius:10px; margin-bottom:10px; }
.flip-card-back { background-color:#1D2D44; color:#F0EBD8; transform: rotateY(180deg); padding:15px; display:flex; flex-direction:column; justify-content:flex-start; align-items:flex-start; text-align:left; font-size:14px;}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# UI Layout
# ------------------------------
col1, col2 = st.columns([1,1])
with col1:
    st.title("🧠 MBTI Personality Predictor")
    st.markdown("### Discover your MBTI personality type through your words.")
    st.write("Type something about yourself — your thoughts, goals, or feelings — and let AI predict your MBTI personality type!")

with col2:
    lottie_ai = load_lottie("https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json")
    if lottie_ai:
        st_lottie(lottie_ai,height=250,key="ai")

st.markdown("---")

# Text Input
user_input = st.text_area("✍️ Write about yourself below:", height=150, placeholder="Example: I enjoy deep conversations and exploring new ideas...")

# Predict Button
if st.button("🔮 Predict Personality"):
    if user_input.strip():
        preds = predict_personality(user_input)
        top3 = preds[:3]

        st.subheader("🌟 Top 3 Personality Predictions:")
        for mbti, prob in top3:
            st.markdown(f"""
                <div class='personality-card'>
                    <h3>✨ {mbti} — {mbti_info[mbti][0]}</h3>
                    <p>{mbti_info[mbti][1]}</p>
                    <b>Confidence:</b> {prob*100:.2f}%
                </div>
            """, unsafe_allow_html=True)
            st.progress(float(prob))

        # Bar chart with percentages
        st.subheader("📊 Confidence of Top 3 Personalities")
        fig, ax = plt.subplots(figsize=(7,4))
        types = [t[0] for t in top3]
        probabilities = [p[1]*100 for p in top3]
        colors = ["#748CAB","#A3B9C9","#F0EBD8"]
        bars = ax.bar(types, probabilities, color=colors, edgecolor="#F0EBD8", linewidth=1.5)
        ax.set_ylabel("Probability (%)", fontsize=12, color="#F0EBD8")
        ax.set_xlabel("MBTI Type", fontsize=12, color="#F0EBD8")
        ax.set_title("Top 3 MBTI Confidence", color="#748CAB", fontsize=14)
        ax.tick_params(colors="#F0EBD8")
        fig.patch.set_facecolor("#0D1321")
        ax.set_facecolor("#1D2D44")
        for bar, val in zip(bars, probabilities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 1, f'{val:.2f}%', ha='center', color='#F0EBD8', fontsize=10)
        st.pyplot(fig)

        # Flip-card section for all 16 personalities
        with st.expander("🧩 Show All 16 Personality Types ▼"):
            cols = st.columns(4)
            for i, (mbti, (title, desc, logo, emoji, img, traits)) in enumerate(mbti_info.items()):
                with cols[i%4]:
                    trait_text = ", ".join(traits)
                    st.markdown(f"""
                        <div class="flip-card">
                          <div class="flip-card-inner">
                            <div class="flip-card-front">
                              <img src="{img}" alt="{mbti}">
                              <h4>{mbti} — {title}</h4>
                            </div>
                            <div class="flip-card-back">
                              <p><b>Logo/Icon:</b> {logo} {emoji}</p>
                              <p>{desc}</p>
                              <p><b>Key Traits:</b> {trait_text}</p>
                            </div>
                          </div>
                        </div>
                    """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please type something before prediction.")

st.markdown("---")
