import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import base64

# 🎨 Function to set background image
def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            color: white;
        }}
        h1, h2, h3 {{
            color: #ffcc00 !important;
            text-shadow: 1px 1px 2px black;
        }}
        div.stButton > button {{
            background: linear-gradient(90deg, #ff6a00, #ee0979);
            color: white;
            font-size: 18px;
            border-radius: 12px;
            border: none;
            padding: 0.6em 1.2em;
        }}
        div.stButton > button:hover {{
            background: linear-gradient(90deg, #ffb347, #ffcc33);
            color: black;
        }}
        section[data-testid="stSidebar"] {{
            background: rgba(0,0,0,0.7);
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# حط صورتك هنا
add_bg_from_local("hotel_bg.jpg")

@st.cache_data
def load_data():
    df = pd.read_csv("Hotel_Reservations_.csv")
    
    numerical_columns = df.select_dtypes(include=['number']).columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns

    for col in categorical_columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    for col in numerical_columns:
        df[col] = df[col].fillna(df[col].mean())

    le = LabelEncoder()
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col].astype(str))
    return df

df = load_data()

X = df.drop(["Booking_ID", "booking_status"], axis=1)
y = df["booking_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Navigation
st.sidebar.title("📌 Hotel Reservations")
page = st.sidebar.radio("🚀choose page:", ["📊 Visualizations", "🤖 Model Prediction"])

# ------------------ : Visualizations ------------------
if page == "📊 Visualizations":
    st.title("📊 Hotel Reservations Visualizations")

    st.subheader("📝 Data Sample (head)")
    st.dataframe(df.head())

    # Countplot arrival_year
    st.subheader("📅 Count of Bookings by Arrival Year")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x="arrival_year", data=df, ax=ax, palette="viridis")
    st.pyplot(fig)

    # Pie chart arrival_month
    st.subheader("📅 Distribution by Arrival Month")
    fig, ax = plt.subplots()
    df['arrival_month'].value_counts().plot.pie(
        autopct='%.0f%%', ax=ax, figsize=(6,6), colors=sns.color_palette("coolwarm")
    )
    st.pyplot(fig)

    # # Line plot lead_time vs booking_status
    # st.subheader("⏳ Lead Time vs Booking Status (Line Plot)")
    # fig, ax = plt.subplots(figsize=(10, 5))
    # lead_status = df.groupby("lead_time")["booking_status"].sum()
    # plt.plot(lead_status.index, lead_status.values, marker="o", linestyle="-", color="#ffcc00")
    # plt.grid(True, alpha=0.3)
    # st.pyplot(fig)

    # Barplot repeated_guest vs booking_status
    st.subheader("👥 Repeated Guest vs Booking Status")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df, x="repeated_guest", y="booking_status", estimator=sum, ci=None, ax=ax, palette="magma")
    st.pyplot(fig)

    # Countplot meal plan
    st.subheader("🍽️ Type of Meal Plan")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x="type_of_meal_plan", data=df, ax=ax, palette="Set2")
    st.pyplot(fig)

    # Avg price per month
    st.subheader("💰 Average Price per Room by Month")
    monthly_avg = df.groupby("arrival_month")["avg_price_per_room"].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    monthly_avg.plot(kind="line", marker="o", ax=ax, color="cyan")
    st.pyplot(fig)

    # Market segment vs booking status
    st.subheader("📊 Market Segment vs Booking Status")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x="market_segment_type", hue="booking_status", ax=ax, palette="cool")
    plt.xticks(rotation=30)
    st.pyplot(fig)

    # Room type vs booking status
    st.subheader("🏨 Room Type vs Booking Status")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x="room_type_reserved", hue="booking_status", ax=ax, palette="Spectral")
    plt.xticks(rotation=30)
    st.pyplot(fig)

    # Booking status pie
    st.subheader("📊 Booking Status Distribution")
    fig, ax = plt.subplots()
    df["booking_status"].value_counts().plot.pie(
        labels=df["booking_status"].value_counts().index,
        autopct="%1.1f%%",
        colors=sns.color_palette("pastel"),
        ax=ax
    )
    st.pyplot(fig)

    # Heatmap correlation
    st.subheader("🔥 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(16, 12))  # 👈 أكبر حجم
    sns.heatmap(
        df.corr(numeric_only=True), 
        annot=True,               # 👈 نعرض الأرقام
        fmt=".2f",                # 👈 رقم عشري من 2 خانات
        cmap="magma", 
        ax=ax
    )
    st.pyplot(fig)

    # Scatterplot
    st.subheader("🔍 Lead Time vs Avg Price per Room (by Status)")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x="lead_time", y="avg_price_per_room", hue="booking_status", palette="cool", ax=ax)
    st.pyplot(fig)

# ------------------ : Model ------------------
elif page == "🤖 Model Prediction":
    st.title("🤖 Booking Cancellation Prediction")

    st.subheader(f"📈 Model Accuracy: {acc:.3f}")

    st.subheader("🧮 Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="plasma",
                xticklabels=["Canceled", "Not_Canceled"],
                yticklabels=["Canceled", "Not_Canceled"], ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)

    st.subheader("📝 Enter Booking Details")

    user_input = {}
    for col in X.columns:
        if df[col].nunique() < 10:
            options = sorted(df[col].unique())
            user_input[col] = st.selectbox(f"{col}", options, index=0)
        else:
            user_input[col] = st.number_input(f"{col}", value=float(df[col].mean()))

    input_df = pd.DataFrame([user_input])

    if st.button("🔮 Predict"):
        prediction = rf.predict(input_df)[0]
        st.subheader("🔮 Prediction Result:")
        if prediction == 1:
            st.success("✅ Not_Canceled")
        else:
            st.error("❌ Canceled")
