import clip
import os
import json
import torch
import streamlit as st
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
from torch.nn.functional import cosine_similarity
from geopy.distance import geodesic
import folium
from streamlit_folium import st_folium
from base64 import b64encode

# === Наши данные ===
DATASET_DIR = "/Users/lolovolkova/Desktop/pet_app_airi/id_pets" # путь к датасету с картинками, где папки с фото
LOCATIONS_PATH = "/Users/lolovolkova/Desktop/pet_app_airi/animal_locations.json" # метаданные с геолокацией
TOP_N = 10

# === Устройство ===
device = "cuda" if torch.cuda.is_available() else "cpu" # ели macos с m2 вместо cuda можно mps  

# === Модель и трансформации ===
@st.cache_resource

# Здесь можно запускать любую модель получения эмбеддингов
def load_clip_model():
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess

model_clip, preprocess_clip = load_clip_model()

# Получение эмбеддингов входного изображения (можно заменить на любую другую модель)
def get_clip_embedding(image: Image.Image):
    image_tensor = preprocess_clip(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model_clip.encode_image(image_tensor)
    return embedding.squeeze(0).cpu()

def get_first_image(animal_id):
    folder = os.path.join(DATASET_DIR, animal_id)
    for file in os.listdir(folder):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            return os.path.join(folder, file)
    return None

def image_to_base64(path):
    with open(path, "rb") as img_file:
        return b64encode(img_file.read()).decode()

# === Использование рассчитаных эмбеддингов с нашей моделью (если меняете модель, сначала необходимо пересоздатть базу эмбеддингов для новой модели) ===

EMBEDDINGS_CACHE_DIR = "/Users/lolovolkova/Desktop/pet_app_airi/embeddings_cache_clip" # тут указываем путь, где храниться json с эмбеддингами для каждого изображения внутри папки-животного из базы данных

def load_cached_embeddings(animal_id):
    path = os.path.join(EMBEDDINGS_CACHE_DIR, f"{animal_id}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

# === Расчет схожести изображений ===

def find_similar_animals(uploaded_images, lat, lon):
    similarity_by_animal = {}

    target_embeddings = [get_clip_embedding(img) for img in uploaded_images]

    for animal_id in os.listdir(DATASET_DIR):
        animal_path = os.path.join(DATASET_DIR, animal_id)
        if not os.path.isdir(animal_path):
            continue

        cached_embeddings = load_cached_embeddings(animal_id)
        if cached_embeddings is None:
            continue

        max_sim = -1
        for _, emb_list in cached_embeddings.items():
            try:
                emb_tensor = torch.tensor(emb_list)
                for user_emb in target_embeddings:
                    sim = cosine_similarity(user_emb.unsqueeze(0), emb_tensor.unsqueeze(0)).item()
                    if sim > max_sim:
                        max_sim = sim
            except Exception as e:
                print(f"⚠️ Ошибка с {animal_id}: {e}")
                continue

         # === Трешхолд схожести - можно менять в зависимости от нашей модели (это параметр можно задавать вручную)===
        if max_sim > 0.73:
            similarity_by_animal[animal_id] = max_sim

    # === Работа с координатами ===
    with open(LOCATIONS_PATH, "r", encoding="utf-8") as f:
        locations = json.load(f)

    top_animals = sorted(similarity_by_animal.items(), key=lambda x: x[1], reverse=True)[:TOP_N]
    nearby, faraway, others = [], [], []

    for animal_id, sim in top_animals:
        if animal_id in locations:
            coords = locations[animal_id]
            dist = geodesic((lat, lon), (coords["lat"], coords["lon"])).km
            first_img = get_first_image(animal_id)
            info = (animal_id, coords, sim, dist, first_img)
            if dist <= 1.0:
                nearby.append(info)
            else:
                faraway.append(info)

    others_ids = set(similarity_by_animal.keys()) - set([a[0] for a in top_animals])
    for animal_id in others_ids:
        if animal_id in locations:
            coords = locations[animal_id]
            dist = geodesic((lat, lon), (coords["lat"], coords["lon"])).km
            first_img = get_first_image(animal_id)
            others.append((animal_id, coords, similarity_by_animal[animal_id], dist, first_img))

    return nearby, faraway, others

# === Streamlit UI ===
st.set_page_config(
    page_title="Поиск животного", 
    layout="wide",
    page_icon="🐾"
)

# === Основные стили из сайта лето AIRI подворовываем :) ===
st.markdown("""
    <style>
    :root {
        --primary: #2F80ED;
        --secondary: #56CCF2;
        --accent: #F2C94C;
        --dark: #333333;
        --light: #F8F9FA;
        --success: #27AE60;
        --danger: #EB5757;
        --warning: #F2994A;
    }
    
    body {
        font-family: 'Inter', sans-serif;
        color: var(--dark);
        background-color: white;
    }
    
    .stApp {
        background: linear-gradient(180deg, #F8F9FA 0%, #FFFFFF 100%);
    }
    
    .stButton>button {
        background-color: var(--primary);
        color: white;
        border-radius: 8px;
        padding: 8px 16px;
        border: none;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #1A6FD9;
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(47, 128, 237, 0.3);
    }
    
    .stFileUploader {
        border: 2px dashed var(--secondary);
        border-radius: 12px;
        padding: 20px;
        background-color: rgba(86, 204, 242, 0.05);
    }
    
    .stMarkdown h1 {
        color: var(--primary);
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .stMarkdown h2 {
        color: var(--dark);
        font-weight: 600;
        border-bottom: 2px solid var(--secondary);
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
    }
    
    .scroll-container {
        max-height: 600px;
        overflow-y: auto;
        padding-right: 10px;
        background: white;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        padding: 1rem;
    }
    
    .card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        transition: all 0.3s;
        border-left: 4px solid var(--primary);
    }
    
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15);
    }
    
    .nearby {
        border-left: 4px solid var(--success) !important;
    }
    
    .faraway {
        border-left: 4px solid var(--warning) !important;
    }
    
    .others {
        border-left: 4px solid var(--secondary) !important;
    }
    
    .map-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        width: 100%;
        height: 400px;
        position: relative;
    }
    
    .stSpinner>div {
        color: var(--primary) !important;
    }
    </style>
""", unsafe_allow_html=True)

# Заголовок с иконкой и градиентом
st.markdown("""
    <div style="background: linear-gradient(90deg, #2F80ED 0%, #56CCF2 100%); 
                padding: 2rem; 
                border-radius: 12px; 
                color: white; 
                margin-bottom: 2rem;">
        <h1 style="color: white; margin: 0; display: flex; align-items: center; gap: 1rem;">
            <span style="font-size: 2.5rem;">🔍</span>
            <span>Поиск пропавшего животного по фото</span>
        </h1>
        <p style="opacity: 0.9; margin: 0.5rem 0 0;">Загрузите фотографии и укажите место, где животное видели в последний раз</p>
    </div>
""", unsafe_allow_html=True)

for key in ["map", "coords", "confirmed_coords", "image", "center_on", "nearby", "faraway", "others", "results", "page"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "page" else 0

# === Секция загрузки фото прльзователем ===
with st.container():
    st.markdown("### 📸 Загрузите фотографии животного")
    st.markdown("<p style='color: #666; margin-top: -1rem;'>Можно загрузить до 10 фотографий для более точного поиска</p>", unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Перетащите файлы сюда или нажмите для выбора", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

if uploaded_files:
    uploaded_files = uploaded_files[:10]
    images = [Image.open(file) for file in uploaded_files]
    st.session_state['image'] = images

    # === Галерея загруженных фото ===
    st.markdown("#### Ваши фотографии")
    cols = st.columns(5)
    for i, img in enumerate(images):
        col = cols[i % 5]
        with col:
            # st.image(img, caption=f"Фото {i+1}", use_column_width=True)
            st.image(img, caption=f"Фото {i+1}", use_container_width=True)

    # === Секция карты ===
    st.markdown("### 🗺️ Укажите место, где животное видели в последний раз")
    st.markdown("<p style='color: #666; margin-top: -1rem;'>Кликните на карте, чтобы отметить точку, затем подтвердите выбор</p>", unsafe_allow_html=True)
    
    default_loc = [55.75, 37.62]
    col_map, col_button = st.columns([4, 1])

    with col_map:
        with st.container():
            pick_map = folium.Map(location=default_loc, zoom_start=12)
            pick_map.add_child(folium.LatLngPopup())
            pick_data = st_folium(pick_map, width=700, height=500)

    with col_button:
        if pick_data and pick_data.get("last_clicked"):
            lat = pick_data["last_clicked"]["lat"]
            lon = pick_data["last_clicked"]["lng"]
            st.session_state['coords'] = (lat, lon)

            if st.button("📍 Подтвердить точку", use_container_width=True):
                st.session_state['confirmed_coords'] = (lat, lon)

                preview_map = folium.Map(location=[lat, lon], zoom_start=13)
                folium.Marker(
                    [lat, lon], 
                    popup="📍 Место пропажи", 
                    icon=folium.Icon(color="red", icon="heart", prefix="fa")
                ).add_to(preview_map)

                with open(LOCATIONS_PATH, "r", encoding="utf-8") as f:
                    all_locations = json.load(f)

                for animal_id, coords in all_locations.items():
                    img_path = get_first_image(animal_id)
                    if img_path:
                        try:
                            sim = get_clip_embedding(Image.open(img_path))
                            for user_img in images:
                                user_emb = get_clip_embedding(user_img)
                                s = cosine_similarity(user_emb.unsqueeze(0), sim.unsqueeze(0)).item()
                                if s > 0.6:
                                    encoded = image_to_base64(img_path)
                                    html = f"""
                                    <div style="font-family: Arial, sans-serif;">
                                        <h4 style="color: var(--primary); margin-bottom: 0.5rem;">ID: {animal_id}</h4>
                                        <img src='data:image/jpeg;base64,{encoded}' style='width: 150px; border-radius: 8px;'>
                                        <p style="margin-top: 0.5rem;">Сходство: {s:.2f}</p>
                                    </div>
                                    """
                                    folium.Marker(
                                        location=[coords["lat"], coords["lon"]],
                                        popup=folium.Popup(html, max_width=250),
                                        icon=folium.Icon(color="orange", icon="paw", prefix="fa")
                                    ).add_to(preview_map)
                                    break
                        except:
                            continue

                st.success("✅ Координаты подтверждены! Теперь вы можете начать поиск.")
            elif st.session_state['confirmed_coords']:
                st.success(f"Выбранная точка: {st.session_state['confirmed_coords']}")
            

# === Кнопка поиска ===
if st.session_state['confirmed_coords'] and st.session_state['image']:
    if st.button("🔍 Начать поиск похожих животных", use_container_width=True):
        with st.spinner("🔍 Идёт поиск... Пожалуйста, подождите"):
            lat, lon = st.session_state['confirmed_coords']
            images = st.session_state['image']
            nearby, faraway, others = find_similar_animals(images, lat, lon)
            st.session_state.update({"nearby": nearby, "faraway": faraway, "others": others, "center_on": None})
            all_animals = nearby + faraway + others
            st.session_state['results'] = sorted(all_animals, key=lambda x: x[2], reverse=True)

# === Отображение результатов ===
if st.session_state['results']:
    st.markdown("""
        <div style="background: linear-gradient(90deg, #2F80ED 0%, #56CCF2 100%); 
                    padding: 1rem; 
                    border-radius: 12px; 
                    color: white; 
                    margin-bottom: 1.5rem;">
            <h2 style="color: white; margin: 0;">🗺️ Найденные животные</h2>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])

    with col1:
        # Карта с результатами (остается без изменений)
        st.markdown("### Карта поиска")
        lat, lon = st.session_state['confirmed_coords']
        center = st.session_state.get("center_on")
        updated_map = folium.Map(location=center or [lat, lon], zoom_start=15 if center else 13)
        updated_map.add_child(folium.LatLngPopup())

        # Группы животных с разными цветами
        groups = [
            (st.session_state['nearby'], "green", "home", "Рядом (до 1 км)"),
            (st.session_state['faraway'], "orange", "map-marker-alt", "Недалеко (более 1 км)"),
            (st.session_state['others'], "blue", "search", "Другие совпадения")
        ]

        for group, color, icon, label in groups:
            for aid, coords, sim, dist, img_path in group:
                if img_path:
                    encoded = image_to_base64(img_path)
                    html = f"""
                    <div style="font-family: Arial, sans-serif;">
                        <h4 style="color: var(--primary); margin-bottom: 0.5rem;">ID: {aid}</h4>
                        <img src='data:image/jpeg;base64,{encoded}' style='width: 150px; border-radius: 8px;'>
                        <p style="margin-top: 0.5rem;"><b>Сходство:</b> {sim:.2f}</p>
                        <p><b>Расстояние:</b> {dist:.2f} км</p>
                    </div>
                    """
                    folium.Marker(
                        location=[coords["lat"], coords["lon"]],
                        popup=folium.Popup(html, max_width=250),
                        icon=folium.Icon(color=color, icon=icon, prefix="fa")
                    ).add_to(updated_map)

        # Основная точка - место пропажи
        folium.Marker(
            location=[lat, lon],
            popup="📍 Место пропажи",
            icon=folium.Icon(color="red", icon="heart", prefix="fa")
        ).add_to(updated_map)

        # Область карты с тенью
        st.markdown('<div class="map-container">', unsafe_allow_html=True)
        st_folium(updated_map, width=600, height=500)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # Список карточек животных (только топ-10 - можно в начале поменять этот параметр, но при прокрутке карты у нас увеличивается число - тут чтобы сайт не подвисал пока отображено небольшое число)
        st.markdown("### 📋 Топ-10 наиболее похожих животных")
        st.markdown('<div class="scroll-container">', unsafe_allow_html=True)

        def display_card(aid, coords, sim, dist, img_path, group):
            group_class = {
                "nearby": "nearby",
                "faraway": "faraway",
                "others": "others"
            }.get(group, "")
            
            st.markdown(f"""
                <div class="card {group_class}">
                    <img src="data:image/jpeg;base64,{image_to_base64(img_path)}" 
                        style="width: 100px; height: auto; border-radius: 6px; margin-bottom: 0.3rem;">
                    <p><b>ID:</b> {aid}</p>
                    <p><b>Сходство:</b> {sim:.2f}</p>
                    <p><b>Расстояние:</b> {dist:.2f} км</p>
                    <button onclick="window.streamlitScriptHost.requestCustomMessage('center_on', '{aid}')" 
                            style="background: var(--primary); color: white; border: none; border-radius: 6px; padding: 6px 12px; cursor: pointer; width: 100%;">
                        📍 Показать на карте
                    </button>
                </div>
            """, unsafe_allow_html=True)

        # Получаем все результаты и сортируем по сходству
        all_results = []
        if st.session_state['nearby']:
            all_results.extend([(*animal, "nearby") for animal in st.session_state['nearby']])
        if st.session_state['faraway']:
            all_results.extend([(*animal, "faraway") for animal in st.session_state['faraway']])
        if st.session_state['others']:
            all_results.extend([(*animal, "others") for animal in st.session_state['others']])
        
        # Сортируем по сходству (cosine simularity) и берем топ-10
        top_10 = sorted(all_results, key=lambda x: x[2], reverse=True)[:10]

        # Отображаем только топ-10
        for animal in top_10:
            display_card(*animal)

        st.markdown('</div>', unsafe_allow_html=True)

# Футер
st.markdown("""
    <div style="margin-top: 3rem; padding: 1.5rem; text-align: center; color: #666; border-top: 1px solid #eee;">
        <p>Сервис поиска пропавших животных • AIRI Pet Finder</p>
        <p style="font-size: 0.9rem;">Использует технологии компьютерного зрения для поиска похожих животных</p>
    </div>
""", unsafe_allow_html=True)
