import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image

# ========================
# キャッシュ対応: 代表色を取得する関数
# ========================
@st.cache_data
def get_dominant_color(region, k=1):
    data = region.reshape((-1, 3))
    data = data[np.any(data != [255, 255, 255], axis=1)]
    if len(data) == 0:
        return (255, 255, 255)
    kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(data)
    return tuple(map(int, kmeans.cluster_centers_[0]))

# ========================
# 色の組み合わせ判定関数
# ========================
@st.cache_data
def color_combination_level_improved(color1_bgr, color2_bgr):
    def bgr_to_hsv(bgr):
        hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)
        return hsv[0][0]
    hsv1, hsv2 = bgr_to_hsv(color1_bgr), bgr_to_hsv(color2_bgr)
    h1, s1, v1 = int(hsv1[0]), int(hsv1[1]), int(hsv1[2])
    h2, s2, v2 = int(hsv2[0]), int(hsv2[1]), int(hsv2[2])
    h_diff, s_diff, v_diff = abs(h1 - h2), abs(s1 - s2), abs(v1 - v2)
    h_diff = min(h_diff, 180 - h_diff)
    s_avg, v_avg = (s1 + s2) / 2, (v1 + v2) / 2

    if h_diff < 20 and s_avg > 120 and v_avg > 180: return "❗️ 奇抜で浮く可能性（鮮やかなワントーン）"
    if s_avg < 25:
        if v_avg < 130: return "✅ 無難 (無彩色)"
        elif v_diff > 120: return "🟦 控えめにおしゃれ (モノトーン)"
        else: return "✅ 無難 (無彩色)"
    if s_avg > 180 and v_avg > 180: return "❗️ 奇抜で浮く可能性 (ネオン系の組み合わせ)"
    if v_diff > 120 and s_diff > 100: return "❗️ 奇抜で浮く可能性(高コントラスト)"
    if h_diff < 30:
        if v_diff > 80 or s_diff > 80: return "🟦 控えめにおしゃれ (トーンオントーン)"
        if s_avg < 100: return "✅ 無難 (類似色)"
        else: return "❗️ 奇抜で浮く可能性 (鮮やかな類似色)"
    if h_diff < 75:
        if s_avg < 90: return "🟦 控えめにおしゃれ (中差色・低彩度)"
        else: return "🟨 目立つけど許容範囲 (中差色・高彩度)"
    if h_diff >= 75:
        if s_avg < 100: return "🟨 目立つけど許容範囲 (補色系・低彩度)"
        else: return "❗️ 奇抜で浮く可能性 (補色系・高彩度)"
    return "❗️ 奇抜で浮く可能性"

# ========================
# 総合スコアを算出する関数
# ========================
@st.cache_data
def calculate_detailed_score(color1_bgr, color2_bgr):
    hsv1 = cv2.cvtColor(np.uint8([[color1_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    hsv2 = cv2.cvtColor(np.uint8([[color2_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    h1, s1, v1 = int(hsv1[0]), int(hsv1[1]), int(hsv1[2])
    h2, s2, v2 = int(hsv2[0]), int(hsv2[1]), int(hsv2[2])

    h_diff = abs(h1 - h2)
    h_diff = min(h_diff, 180 - h_diff)
    s_diff = abs(s1 - s2)
    v_diff = abs(v1 - v2)
    s_avg = (s1 + s2) / 2
    v_avg = (v1 + v2) / 2

    judgment = color_combination_level_improved(color1_bgr, color2_bgr)

    if "無難" in judgment:
        base_min, base_max = 85, 100
    elif "控えめ" in judgment:
        base_min, base_max = 70, 84
    elif "許容範囲" in judgment:
        base_min, base_max = 50, 69
    elif "奇抜" in judgment:
        base_min, base_max = 0, 49
    else:
        base_min, base_max = 50, 69

    base_score = (base_min + base_max) / 2
    offset = 0

    if v_diff > 80 and s_diff < 80:
        offset += 5
    elif v_diff < 20 and 40 < s_avg < 160:
        offset -= 5

    if s_avg > 200 and v_avg > 220:
        offset -= 5

    if h_diff < 30 and v_diff > 50 and s_diff > 50:
        offset += 5

    final_score = base_score + offset

    return int(np.clip(final_score, base_min, base_max))

# ========================
# アドバイスを返す関数
# ========================
def get_advice(judgment):
    if "奇抜" in judgment:
        return "💡 アドバイス: 派手な印象を和らげたい場合は、どちらかを中間色や低彩度に変えてみましょう。"
    return "👍 特に問題のない組み合わせです。自信を持ってコーディネートを楽しみましょう！"

# ========================
# 季節のカラーパレット
# ========================
season_palettes = {
    "春": [(25, 60, 240), (35, 90, 230), (90, 40, 220), (160, 70, 240), (20, 40, 210), (0, 0, 250), (120, 30, 200)],
    "夏": [(130, 40, 230), (95, 50, 220), (160, 60, 250), (0, 0, 245), (115, 30, 210), (140, 50, 200), (30, 50, 200)],
    "秋": [(15, 180, 160), (25, 150, 140), (10, 80, 80), (40, 130, 110), (0, 100, 180), (0, 0, 180), (100, 60, 130)],
    "冬": [(120, 200, 90), (140, 180, 80), (0, 0, 255), (0, 0, 30), (160, 160, 255), (110, 120, 70), (0, 150, 250)]
}

# ========================
# 差し色を提案する関数
# ========================
@st.cache_data
def suggest_accent_color(color1_bgr, color2_bgr):
    hsv1 = cv2.cvtColor(np.uint8([[color1_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    hsv2 = cv2.cvtColor(np.uint8([[color2_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    avg_h = (int(hsv1[0]) + int(hsv2[0])) / 2
    accent_h1 = int((avg_h + 60) % 180)
    accent_h2 = int((avg_h + 120) % 180)
    accent_s, accent_v = 200, 230
    accent_hsv1, accent_hsv2 = (accent_h1, accent_s, accent_v), (accent_h2, accent_s, accent_v)
    accent_bgr1 = tuple(int(c) for c in cv2.cvtColor(np.uint8([[accent_hsv1]]), cv2.COLOR_HSV2BGR)[0][0])
    accent_bgr2 = tuple(int(c) for c in cv2.cvtColor(np.uint8([[accent_hsv2]]), cv2.COLOR_HSV2BGR)[0][0])
    return [accent_bgr1, accent_bgr2]

# ========================
# 代替カラーを提案する関数
# ========================
@st.cache_data
def generate_alternative_colors(fixed_color_bgr, season, is_top):
    suggestions, candidate_hsvs = [], []
    h, s, v = cv2.cvtColor(np.uint8([[fixed_color_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    base_colors_bgr = get_dynamic_base_colors(v)
    candidate_hsvs.extend([tuple(cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]) for bgr in base_colors_bgr])
    
    if season == "選択なし":
        allowed_keywords = ["無難", "控えめ"]
        for delta_h in [-90, -45, -20, 20, 45, 90]:
            for delta_s in [-50, 0, 50]:
                for delta_v in [-50, 0, 50]:
                    nh, ns, nv = (int(h) + delta_h) % 180, np.clip(int(s) + delta_s, 30, 255), np.clip(int(v) + delta_v, 30, 255)
                    candidate_hsvs.append((nh, ns, nv))
        comp_h = (int(h) + 90) % 180
        candidate_hsvs.extend([(comp_h, np.clip(s, 100, 200), np.clip(v, 100, 200)), (comp_h, 80, 150)])
    else:
        allowed_keywords = ["無難", "控えめ", "許容範囲"]
        for base_hsv in season_palettes[season]:
            for delta_v in [-40, 0, 40]:
                nh, ns, nv = base_hsv
                candidate_hsvs.append((nh, ns, np.clip(nv + delta_v, 30, 255)))

    for hsv in set(candidate_hsvs):
        new_bgr_tuple = tuple(int(c) for c in cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2BGR)[0][0])
        top_color, bottom_color = (new_bgr_tuple, fixed_color_bgr) if is_top else (fixed_color_bgr, new_bgr_tuple)
        judgment = color_combination_level_improved(top_color, bottom_color)
        if any(word in judgment for word in allowed_keywords):
            suggestions.append((new_bgr_tuple, judgment))
            
    # 色が重複するものを排除し5件まで
    return list({s[0]: s for s in suggestions}.values())[:5]

# ========================
# 動的にベースカラーを生成する関数
# ========================
def get_dynamic_base_colors(v_value):
    staple_colors = [(205, 220, 235), (130, 70, 20)]
    if v_value > 170:
        dynamic_neutrals = [(128, 128, 128), (80, 40, 0), (50, 50, 50)]
    elif v_value < 85:
        dynamic_neutrals = [(245, 245, 245), (210, 210, 210)]
    else:
        dynamic_neutrals = [(245, 245, 245), (128, 128, 128), (50, 50, 50)]
    return staple_colors + dynamic_neutrals


# ========================
# カラー表示用の丸い色チップHTML
# ========================
def create_color_chip_html(bgr_color, size=30):
    rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])
    return f"""
    <div style='
        display:inline-block; 
        width:{size}px; height:{size}px; 
        background-color: rgb{rgb_color};
        border-radius: 50%;
        box-shadow: 0 0 8px rgba(0,0,0,0.25);
        margin-right:12px;
        vertical-align:middle;
        border: 2px solid #eee;
    '></div>
    """

# ========================
# Streamlit アプリ本体
# ========================
st.set_page_config(page_title="コーディネートはこーでねーと", layout="centered")
st.title("🎨コーディネートはこーでねーと")

season = st.selectbox("季節を選んでください (提案される色が変わります)", ["選択なし", "春", "夏", "秋", "冬"])
uploaded_file = st.file_uploader("服装画像をアップロードしてください(全身が写っている画像を推奨します)", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    with mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        h, w, _ = img_bgr.shape
        result = pose.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        if result.pose_landmarks:
            lm = result.pose_landmarks.landmark
            def to_pixel(p): return int(p.x * w), int(p.y * h)
            sL, sR = to_pixel(lm[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]), to_pixel(lm[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER])
            hL, hR = to_pixel(lm[mp.solutions.pose.PoseLandmark.LEFT_HIP]), to_pixel(lm[mp.solutions.pose.PoseLandmark.RIGHT_HIP])
            kL, kR = to_pixel(lm[mp.solutions.pose.PoseLandmark.LEFT_KNEE]), to_pixel(lm[mp.solutions.pose.PoseLandmark.RIGHT_KNEE])
            x1, y1 = min(sL[0], sR[0]), min(sL[1], sR[1])
            x2, y2 = max(hL[0], hR[0]), max(hL[1], hR[1])
            y3 = max(kL[1], kR[1])
            y1, y2, y3 = max(0, y1), max(y1 + 10, y2), max(y2 + 10, y3)
            x1, x2 = max(0, x1), max(x1 + 10, x2)
            top_region, bottom_region = img_bgr[y1:y2, x1:x2], img_bgr[y2:y3, x1:x2]

            if top_region.size == 0 or bottom_region.size == 0:
                st.error("⚠️ 服装の領域を検出できませんでした。別の画像をお試しください。")
            else:
                top_color, bottom_color = get_dominant_color(top_region), get_dominant_color(bottom_region)
                
                st.image(image, caption="アップロード画像", use_column_width=True)

                st.markdown(f"""
                <div style='
                    display:flex; 
                    align-items:center; 
                    font-size:18px; 
                    margin-top:20px; 
                    margin-bottom:15px;
                    font-weight:bold;
                    color:#333;
                '>
                    {create_color_chip_html(top_color, 40)} トップスの代表色
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div style='
                    display:flex; 
                    align-items:center; 
                    font-size:18px; 
                    margin-bottom:25px;
                    font-weight:bold;
                    color:#333;
                '>
                    {create_color_chip_html(bottom_color, 40)} ボトムスの代表色
                </div>
                """, unsafe_allow_html=True)

                # スコア＆判定カード風ボックス
                score = calculate_detailed_score(top_color, bottom_color)
                judgment = color_combination_level_improved(top_color, bottom_color)
                
                st.markdown(f"""
                <div style='
                    background-color:#f5faff; 
                    padding:30px; 
                    border-radius:25px; 
                    box-shadow: 0 10px 25px rgba(30,144,255,0.2);
                    max-width:500px;
                    margin-bottom:40px;
                    font-family:"Helvetica Neue", Arial, sans-serif;
                '>
                    <h2 style='color:#1E90FF; margin-bottom:10px; font-weight: bold; font-size:26px;'>コーデスコア: <span style='font-size:50px; color:#FF4500; font-weight:bold;'>{score} 点</span></h2> 
                    <p style='font-weight:bold; font-size:22px; margin:12px 0; color:#333;'>判定: {judgment}</p>
                    <p style='font-style:italic; color:#555; font-size:18px; margin-top:14px;'>{get_advice(judgment)}</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("---")


                # 差し色提案
                with st.expander("💡 別のコーディネート提案を見る"):
                    st.markdown("<h4 style='color:#0078D7; margin-bottom:12px;'>👜 小物やアクセサリーで差し色を加えるなら？</h4>", unsafe_allow_html=True)
                    accent_colors = suggest_accent_color(top_color, bottom_color)
                    cols = st.columns(len(accent_colors))
                    for i, ac_color in enumerate(accent_colors):
                        with cols[i]:
                            st.markdown(create_color_chip_html(ac_color, 40), unsafe_allow_html=True)
                            st.markdown(f"<small style='color:#555;'>差し色案 {i+1}</small>", unsafe_allow_html=True)

                    # 代替カラー提案（トップス・ボトムス）
                    
                    st.markdown("<h4 style='color:#0078D7; margin-top:25px; margin-bottom:12px;'>👕 トップスの色を変えたい場合の提案</h4>", unsafe_allow_html=True)
                    alt_tops = generate_alternative_colors(bottom_color, season, is_top=True)
                    cols = st.columns(len(alt_tops))
                    for i, (color, judgment_alt) in enumerate(alt_tops):
                        with cols[i]:
                            st.markdown(create_color_chip_html(color, 40), unsafe_allow_html=True)
                            st.markdown(f"<small style='color:#555;'>{judgment_alt}</small>", unsafe_allow_html=True)

                    
                    st.markdown("<h4 style='color:#0078D7; margin-top:25px; margin-bottom:12px;'>👖 ボトムスの色を変えたい場合の提案</h4>", unsafe_allow_html=True)
                    alt_bottoms = generate_alternative_colors(top_color, season, is_top=False)
                    cols = st.columns(len(alt_bottoms))
                    for i, (color, judgment_alt) in enumerate(alt_bottoms):
                        with cols[i]:
                            st.markdown(create_color_chip_html(color, 40), unsafe_allow_html=True)
                            st.markdown(f"<small style='color:#555;'>{judgment_alt}</small>", unsafe_allow_html=True)

        else:
            st.error("⚠️ 画像から姿勢ランドマークを検出できませんでした。もっとはっきりした画像をアップロードしてください。")

else:
    st.info("画像をアップロードすると、トップスとボトムスの代表色を判定してコーディネートを評価します。")

