import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image

def get_dominant_color(region, k=1):
    data = region.reshape((-1, 3))
    data = data[np.any(data != [255, 255, 255], axis=1)]
    if len(data) == 0:
        return (255, 255, 255)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    return tuple(map(int, kmeans.cluster_centers_[0]))

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

def get_advice(judgment):
    if "奇抜" in judgment:
        return (
            "💡 アドバイス: 色がかなり目立つので、落ち着いた色味のアクセサリーや小物を合わせるとバランスが取れます。\n"
            "または、どちらか一方の色を抑えめの中間色にすると良いでしょう。"
        )
    else:
        return "👍 特に問題のない組み合わせです。自信を持って出かけましょう！"


season_palettes = {
    "春": [(30, 80, 220), (150, 50, 230), (20, 70, 210)],
    "夏": [(90, 150, 200), (110, 120, 240), (0, 0, 255)],
    "秋": [(15, 180, 150), (25, 170, 130), (10, 200, 120)],
    "冬": [(120, 200, 80), (0, 0, 50), (140, 190, 60)],
}

def generate_alternative_colors(fixed_color_bgr, season=None, change_target="top"):
    fixed_hsv = cv2.cvtColor(np.uint8([[fixed_color_bgr]]), cv2.COLOR_BGR2HSV)[0][0]

    suggestions = []

    if season is None:  # 季節なしは幅広い色変化で生成
        for delta_h in [-60, -30, -15, 15, 30, 60, 90, 120]:
            for delta_s in [-60, -30, 0, 30]:
                for delta_v in [-60, -30, 0, 30]:
                    nh = (h + delta_h) % 180
                    ns = np.clip(s + delta_s, 30, 255)
                    nv = np.clip(v + delta_v, 30, 255)

                    new_hsv = np.uint8([[[nh, ns, nv]]])
                    new_bgr = cv2.cvtColor(new_hsv, cv2.COLOR_HSV2BGR)[0][0]
                    new_bgr_tuple = tuple(int(x) for x in new_bgr)

                    if change_target == "top":
                        judgment = color_combination_level_improved(new_bgr_tuple, fixed_color_bgr)
                    else:
                        judgment = color_combination_level_improved(fixed_color_bgr, new_bgr_tuple)

                    if any(word in judgment for word in ["無難", "控えめ", "許容範囲"]):
                        suggestions.append((new_bgr_tuple, judgment))
        return suggestions[:5]

    # 季節指定ありの場合（従来の季節パレットに寄せる方式）
    palette_hsv = [np.uint8([[[h, s, v]]]) for (h, s, v) in season_palettes[season]]
    palette_bgr = [cv2.cvtColor(c, cv2.COLOR_HSV2BGR)[0][0] for c in palette_hsv]

    for base_bgr in palette_bgr:
        base_hsv = cv2.cvtColor(np.uint8([[base_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        for delta_h in [-15, 0, 15]:
            nh = (int(base_hsv[0]) + delta_h) % 180
            ns = np.clip(int(base_hsv[1]) + 10, 30, 255)
            nv = np.clip(int(base_hsv[2]) + 10, 30, 255)

            new_hsv = np.uint8([[[nh, ns, nv]]])
            new_bgr = cv2.cvtColor(new_hsv, cv2.COLOR_HSV2BGR)[0][0]
            new_bgr_tuple = tuple(int(x) for x in new_bgr)

            if change_target == "top":
                judgment = color_combination_level_improved(new_bgr_tuple, fixed_color_bgr)
            else:
                judgment = color_combination_level_improved(fixed_color_bgr, new_bgr_tuple)

            if any(word in judgment for word in ["無難", "控えめ", "許容範囲"]):
                suggestions.append((new_bgr_tuple, judgment))

    return suggestions[:3]

# Streamlit UI
st.set_page_config(page_title="コーディネートはこーでねーと", layout="centered")
st.title("👕👖 コーディネートはこーでねーと")

season = st.selectbox("季節を選んでください", ["選択なし", "春", "夏", "秋", "冬"])

uploaded_file = st.file_uploader("服装画像をアップロードしてください", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True) as pose:
        h, w, _ = img_bgr.shape
        result = pose.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        if result.pose_landmarks:
            lm = result.pose_landmarks.landmark
            def to_pixel(p): return int(p.x * w), int(p.y * h)

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

            st.image(image, caption="アップロード画像", use_column_width=True)
            st.markdown(f"<div style='background-color:rgb{top_rgb}; width:20px; height:20px; display:inline-block; border:1px solid #000; margin-right:5px;'></div> **トップスの代表色**: {top_color}", unsafe_allow_html=True)
            st.markdown(f"<div style='background-color:rgb{bottom_rgb}; width:20px; height:20px; display:inline-block; border:1px solid #000; margin-right:5px;'></div> **ボトムスの代表色**: {bottom_color}", unsafe_allow_html=True)

            st.markdown(f"### 🎨 判定結果:\n{judgment}")
            st.markdown(f"### 💬 アドバイス:\n{advice}")

            st.markdown("### 🧩 代替コーディネート提案")

            # 季節選択が「選択なし」の場合は None を渡す
            season_arg = None if season == "選択なし" else season

            top_suggestions = generate_alternative_colors(bottom_color, season_arg, change_target="top")
            bottom_suggestions = generate_alternative_colors(top_color, season_arg, change_target="bottom")

            if top_suggestions:
                st.markdown("#### 👕 トップスを変えるなら？")
                for color, judgment in top_suggestions:
                    rgb = (color[2], color[1], color[0])
                    st.markdown(
                        f"<div style='background-color:rgb{rgb}; width:20px; height:20px; display:inline-block; border:1px solid #000; margin-right:5px;'></div> 提案色: {color} - {judgment}",
                        unsafe_allow_html=True,
                    )

            if bottom_suggestions:
                st.markdown("#### 👖 ボトムスを変えるなら？")
                for color, judgment in bottom_suggestions:
                    rgb = (color[2], color[1], color[0])
                    st.markdown(
                        f"<div style='background-color:rgb{rgb}; width:20px; height:20px; display:inline-block; border:1px solid #000; margin-right:5px;'></div> 提案色: {color} - {judgment}",
                        unsafe_allow_html=True,
                    )
        else:
            st.error("⚠️ 人物が検出できませんでした。画像を確認してください。")
