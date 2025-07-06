import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image

# ========================
# 代表色を取得する関数
# ========================
def get_dominant_color(region, k=1):
    data = region.reshape((-1, 3))
    data = data[np.any(data != [255, 255, 255], axis=1)]
    if len(data) == 0:
        return (255, 255, 255)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    return tuple(map(int, kmeans.cluster_centers_[0]))

# ========================
# 色の組み合わせを判定する関数
# ========================
def color_combination_level_improved(color1_bgr, color2_bgr):
    def bgr_to_hsv(bgr):
        hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)
        return hsv[0][0]

    hsv1 = bgr_to_hsv(color1_bgr)
    hsv2 = bgr_to_hsv(color2_bgr)

    h1, s1, v1 = int(hsv1[0]), int(hsv1[1]), int(hsv1[2])
    h2, s2, v2 = int(hsv2[0]), int(hsv2[1]), int(hsv2[2])

    h_diff = abs(h1 - h2)
    h_diff = min(h_diff, 180 - h_diff)
    s_diff = abs(s1 - s2)
    v_diff = abs(v1 - v2)
    s_avg = (s1 + s2) / 2
    v_avg = (v1 + v2) / 2

    if h_diff < 20 and s_avg > 120 and v_avg > 180:
        return "❗️ 奇抜で浮く可能性（鮮やかなワントーン）"
    if s_avg < 25:
        if v_avg < 130:
            return "✅ 無難 (無彩色)"
        elif v_diff > 120:
            return "🟦 控えめにおしゃれ (モノトーン)"
        else:
            return "✅ 無難 (無彩色)"
    if s_avg > 180 and v_avg > 180:
        return "❗️ 奇抜で浮く可能性 (ネオン系の組み合わせ)"
    if v_diff > 120 and s_diff > 100:
        return "❗️ 奇抜で浮く可能性(高コントラスト)"
    if h_diff < 30:
        if v_diff > 80 or s_diff > 80:
            return "🟦 控えめにおしゃれ (トーンオントーン)"
        if s_avg < 100:
            return "✅ 無難 (類似色)"
        else:
            return "❗️ 奇抜で浮く可能性 (鮮やかな類似色)"
    if h_diff < 75:
        if s_avg < 90:
            return "🟦 控えめにおしゃれ (中差色・低彩度)"
        else:
            return "🟨 目立つけど許容範囲 (中差色・高彩度)"
    if h_diff >= 75:
        if s_avg < 100:
            return "🟨 目立つけど許容範囲 (補色系・低彩度)"
        else:
            return "❗️ 奇抜で浮く可能性 (補色系・高彩度)"

    return "❗️ 奇抜で浮く可能性"

# ========================
# アドバイスを返す関数
# ========================
def get_advice(judgment):
    if "奇抜" in judgment:
        return (
            "💡 アドバイス: 色がかなり目立つので、落ち着いた色味のアクセサリーや小物を合わせるとバランスが取れます。\n"
            "または、どちらか一方の色を抑えめの中間色にすると良いでしょう。"
        )
    else:
        return "👍 特に問題のない組み合わせです。自信を持って出かけましょう！"

# ========================
# 動的に代替色を提案する関数
# ========================
def generate_alternative_colors(fixed_color_bgr, change_target="top"):
    fixed_hsv = cv2.cvtColor(np.uint8([[fixed_color_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = int(fixed_hsv[0]), int(fixed_hsv[1]), int(fixed_hsv[2])

    variations = [
        ((h + 15) % 180, s, v),
        ((h - 15) % 180, s, v),
        ((h + 90) % 180, max(50, s - 50), max(50, v - 50)),
        ((h + 30) % 180, s, min(255, v + 30)),
        ((h + 60) % 180, min(255, s + 30), v),
    ]

    suggestions = []
    for nh, ns, nv in variations:
        hsv = np.uint8([[[nh, ns, nv]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        bgr_tuple = tuple(int(x) for x in bgr)

        if change_target == "top":
            judgment = color_combination_level_improved(bgr_tuple, fixed_color_bgr)
        else:
            judgment = color_combination_level_improved(fixed_color_bgr, bgr_tuple)

        if "無難" in judgment or "控えめ" in judgment:
            suggestions.append((bgr_tuple, judgment))

    return suggestions[:2]

# ========================
# Streamlit アプリ本体
# ========================
st.set_page_config(page_title="コーディネート診断", layout="centered")
st.title("👕👖 コーディネートはこーでねーと")

uploaded_file = st.file_uploader("服装画像をアップロードしてください(全身が写っているものがいいです)", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True) as pose:
        h, w, _ = img_bgr.shape
        result = pose.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        if result.pose_landmarks:
            lm = result.pose_landmarks.landmark

            def to_pixel(lm): return int(lm.x * w), int(lm.y * h)

            sL = to_pixel(lm[mp_pose.PoseLandmark.LEFT_SHOULDER])
            sR = to_pixel(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER])
            hL = to_pixel(lm[mp_pose.PoseLandmark.LEFT_HIP])
            hR = to_pixel(lm[mp_pose.PoseLandmark.RIGHT_HIP])
            kL = to_pixel(lm[mp_pose.PoseLandmark.LEFT_KNEE])
            kR = to_pixel(lm[mp_pose.PoseLandmark.RIGHT_KNEE])

            x1, y1 = min(sL[0], sR[0]), min(sL[1], sR[1])
            x2, y2 = max(hL[0], hR[0]), max(hL[1], hR[1])
            y3 = max(kL[1], kR[1])

            top_region = img_bgr[y1:y2, x1:x2]
            bottom_region = img_bgr[y2:y3, x1:x2]

            top_color = get_dominant_color(top_region)
            bottom_color = get_dominant_color(bottom_region)

            top_rgb = (top_color[2], top_color[1], top_color[0])
            bottom_rgb = (bottom_color[2], bottom_color[1], bottom_color[0])

            judgment = color_combination_level_improved(top_color, bottom_color)
            advice = get_advice(judgment)

            # 表示
            st.image(image, caption="アップロード画像", use_column_width=True)
            st.markdown(
                f"<div style='display:inline-block; width:20px; height:20px; background-color:rgb{top_rgb}; border:1px solid #000; margin-right:8px;'></div>"
                f"**トップスの代表色**: {top_color}",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='display:inline-block; width:20px; height:20px; background-color:rgb{bottom_rgb}; border:1px solid #000; margin-right:8px;'></div>"
                f"**ボトムスの代表色**: {bottom_color}",
                unsafe_allow_html=True,
            )
            st.markdown(f"### 🎨 判定結果:\n{judgment}")
            st.markdown(f"### 💬 アドバイス:\n{advice}")

            # ======= 代替コーディネート提案 =======
            st.markdown("### 🧩 代替コーディネートの提案")

            top_suggestions = generate_alternative_colors(bottom_color, change_target="top")
            bottom_suggestions = generate_alternative_colors(top_color, change_target="bottom")

            if top_suggestions:
                st.markdown("#### 👕 トップスを変えるなら？")
                for color, judgment in top_suggestions:
                    rgb = (color[2], color[1], color[0])
                    st.markdown(
                        f"<div style='display:inline-block; width:20px; height:20px; background-color:rgb{rgb}; border:1px solid #000; margin-right:8px;'></div>"
                        f"**提案色** - {judgment}",
                        unsafe_allow_html=True,
                    )

            if bottom_suggestions:
                st.markdown("#### 👖 ボトムスを変えるなら？")
                for color, judgment in bottom_suggestions:
                    rgb = (color[2], color[1], color[0])
                    st.markdown(
                        f"<div style='display:inline-block; width:20px; height:20px; background-color:rgb{rgb}; border:1px solid #000; margin-right:8px;'></div>"
                        f"**提案色** - {judgment}",
                        unsafe_allow_html=True,
                    )
        else:
            st.error("⚠️ 人物が検出できませんでした。上半身が明確に写っている画像をアップロードしてください。")
