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
# キャッシュ対応: 色の組み合わせを判定する関数
# ========================
@st.cache_data
def color_combination_level_improved(color1_bgr, color2_bgr):
    # ... (内容は変更なし) ...
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
# 【新機能】複雑なロジックで総合スコアを算出する関数
# ========================
@st.cache_data
def calculate_detailed_score(color1_bgr, color2_bgr):
    """
    色の組み合わせのHSV値を多角的に分析し、スコアを算出する。
    """
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

    score = 70  # ベーススコア

    # 1. 彩度に基づく評価 (S: Saturation)
    if s_avg < 25:  # 無彩色コーデは非常に合わせやすい
        score += 20
    elif s_avg < 80:  # 低彩度で落ち着いている
        score += 10
    if s_avg > 180 and v_avg > 180: # ネオンカラーの組み合わせは難しい
        score -= 40

    # 2. 明度差に基づく評価 (V: Value)
    if v_diff > 70:  # コントラストが明確でおしゃれに見えやすい
        score += 10
    elif v_diff < 20 and 40 < s_avg < 150: # 中彩度で明度差が少ないと地味に見えがち
        score -= 10

    # 3. 色相差に基づく評価 (H: Hue)
    if h_diff < 30:  # 類似色・同系色
        score += 5
        if v_diff > 80 or s_diff > 80: # さらにトーンオントーンなら高評価
            score += 10
    elif h_diff >= 75: # 補色系
        if s_avg < 100: # 彩度が低ければ、上級者向けのおしゃれな配色
            score += 15
        else: # 彩度が高い補色系は非常に奇抜
            score -= 30
    
    # スコアを0-100の範囲に収める
    return int(np.clip(score, 0, 100))

# ========================
# アドバイスを返す関数 (変更なし)
# ========================
def get_advice(judgment):
    if "奇抜" in judgment: return "💡 アドバイス: 派手な印象を和らげたい場合は、どちらかを中間色や低彩度に変えてみましょう。"
    return "👍 特に問題のない組み合わせです。自信を持ってコーディネートを楽しみましょう！"

# (他の関数は変更なし)
# ... season_palettes, suggest_accent_color, generate_alternative_colors, get_dynamic_base_colors ...
# ========================
# 季節のカラーパレット (変更なし)
# ========================
season_palettes = {
    "春": [(25, 60, 240), (35, 90, 230), (90, 40, 220), (160, 70, 240), (20, 40, 210), (0, 0, 250), (120, 30, 200)],
    "夏": [(130, 40, 230), (95, 50, 220), (160, 60, 250), (0, 0, 245), (115, 30, 210), (140, 50, 200), (30, 50, 200)],
    "秋": [(15, 180, 160), (25, 150, 140), (10, 80, 80), (40, 130, 110), (0, 100, 180), (0, 0, 180), (100, 60, 130)],
    "冬": [(120, 200, 90), (140, 180, 80), (0, 0, 255), (0, 0, 30), (160, 160, 255), (110, 120, 70), (0, 150, 250)]
}

# ========================
# 差し色を提案する関数 (変更なし)
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
# 代替カラーを提案する関数 (変更なし)
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
            
    return list({s[0]: s for s in suggestions}.values())[:5]

# ========================
# 動的にベースカラーを生成する関数 (変更なし)
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
# Streamlit アプリ本体
# ========================
st.set_page_config(page_title="コーディネートはこーでねーと", layout="centered")
st.title("👕👖 コーディネートはこーでねーと")

season = st.selectbox("季節を選んでください (提案される色が変わります)", ["選択なし", "春", "夏", "秋", "冬"])
uploaded_file = st.file_uploader("服装画像をアップロードしてください", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    with mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        h, w, _ = img_bgr.shape
        result = pose.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        if result.pose_landmarks:
            # (ランドマーク検出部分は変更なし)
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
                def create_color_chip_html(bgr_color):
                    rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])
                    return f"<div style='display:inline-block; width:20px; height:20px; background-color:rgb{rgb_color}; border:1px solid #ccc; margin-right:8px; vertical-align:middle;'></div>"

                st.markdown(f"{create_color_chip_html(top_color)} **トップスの代表色**", unsafe_allow_html=True)
                st.markdown(f"{create_color_chip_html(bottom_color)} **ボトムスの代表色**", unsafe_allow_html=True)
                
                # --- 【UI変更】新しいスコア算出関数を呼び出し ---
                score = calculate_detailed_score(top_color, bottom_color)
                st.metric("コーデスコア", f"{score} 点")

                # --- 判定とアドバイス表示 ---
                judgment = color_combination_level_improved(top_color, bottom_color)
                st.markdown(f"**判定**: {judgment}")
                st.markdown(f"**アドバイス**: {get_advice(judgment)}")
                
                st.markdown("---")

                # --- 提案機能 (変更なし) ---
                with st.expander("💡 別のコーディネート提案を見る"):
                    st.markdown("##### 🎨 小物で差し色を加えるなら？")
                    accent_colors = suggest_accent_color(top_color, bottom_color)
                    cols = st.columns(len(accent_colors))
                    for i, color in enumerate(accent_colors):
                        with cols[i]:
                            st.markdown(create_color_chip_html(color), unsafe_allow_html=True)
                    st.markdown("---")
                    
                    top_suggestions = generate_alternative_colors(bottom_color, season, is_top=True)
                    if top_suggestions:
                        st.markdown("##### 👕 今のボトムスに合わせるなら？")
                        for color, j in top_suggestions:
                            st.markdown(f"{create_color_chip_html(bottom_color)} + {create_color_chip_html(color)} &rarr; {j}", unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    bottom_suggestions = generate_alternative_colors(top_color, season, is_top=False)
                    if bottom_suggestions:
                        st.markdown("##### 👖 今のトップスに合わせるなら？")
                        for color, j in bottom_suggestions:
                            st.markdown(f"{create_color_chip_html(top_color)} + {create_color_chip_html(color)} &rarr; {j}", unsafe_allow_html=True)
        else:
            st.error("⚠️ 人物が検出できませんでした。全身が写っている画像をお試しください。")