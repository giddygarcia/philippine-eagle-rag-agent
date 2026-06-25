import streamlit as st


def render_header():
    st.title("🦅 Philippine Eagle Research Chatbot")
    st.markdown(
        """
        The **great Philippine Eagle** is one of the largest and rarest forest raptors in the world. It is the national bird of the Philippines and is endemic to the forests of the country, currently facing declining populations as a **critically endangered** species — learn more, chat about these species and how to help today:
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_header():
    with st.sidebar:
        st.image("visuals/eagle_picture.jpeg", width="stretch")
        st.markdown("## 📚️ Learn More:")
        st.markdown("""
- [🌐 Philippine Eagle Foundation](https://www.philippineeaglefoundation.org/philippine-eagle)
- [📋 IUCN Statistics / BirdLife Fact Sheet](https://datazone.birdlife.org/species/factsheet/philippine-eagle-pithecophaga-jefferyi)
- [📖 Biodiversity Management Bureau](https://www.bmb.gov.ph)
            """)


def apply_styles():
    st.markdown(
        """
        <style>
        .stSidebar a {
            color: #CDB30C !important;
            text-decoration: underline !important;
        }

        .stSidebar a:hover {
            text-decoration: underline !important;
        }
        div[data-testid="stPopover"] button {
            border-radius: 50%;
            width: 44px;
            height: 44px;
            font-size: 18px;
        }

        [data-testid="stAlertContainer"] {
            background-color: #2d4a3e !important;
            color: #FFFFFF !important;
        }

        [data-testid="stChatMessageContent"] h1 { font-size: 1.1rem !important; }
        [data-testid="stChatMessageContent"] h2 { font-size: 1rem !important; }
        [data-testid="stChatMessageContent"] h3 { font-size: 0.95rem !important; }
        [data-testid="stChatMessageContent"] h4,
        [data-testid="stChatMessageContent"] h5,
        [data-testid="stChatMessageContent"] h6 { font-size: 0.9rem !important; }

        [data-testid="stChatMessageContent"] h1,
        [data-testid="stChatMessageContent"] h2,
        [data-testid="stChatMessageContent"] h3 { font-weight: 600 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
