# ==========================================
# App Header & College Branding (Custom Styled with Logo)
# ==========================================
import os

logo_path = "logo_pjc.png"

col_title, col_logo = st.columns([4, 1])

with col_title:
    st.markdown(
        """
        <div style='text-align: left; margin-bottom: 15px;'>
            <h1 style='color: #0288d1; font-size: 2.3em; margin-bottom: 5px;'>🎓 Universal AI Virtual Classroom</h1>
            <p style='color: #0288d1; font-size: 1.25em; font-weight: 500; margin-top: 0; margin-bottom: 10px;'>One platform, endless e‑learning possibilities</p>
            <p style='color: #555; font-size: 0.9em; margin: 0;'><b>Maintained by:</b> Prabhu Jagatbandhu College, Andul-Mouri, Howrah, Pin- 711302</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_logo:
    if os.path.exists(logo_path):
        st.image(logo_path, width=110)
    else:
        st.markdown("<p style='color: #888; font-size: 0.8em;'>(Logo missing)</p>", unsafe_allow_html=True)
