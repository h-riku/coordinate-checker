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
    # n_init='auto' を指定して将来的な警告(FutureWarning)を抑制
    kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(data)
    return tuple(map(int, kmeans.cluster_centers_[0]))

# ========================
# 色の組み合わせを判定する関数 (変更なし)
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
# アドバイスを返す関数 (変更なし)
# ========================
def get_advice(judgment):
    if "奇抜" in judgment:
        return "💡 アドバイス: 派手な印象を和らげたい場合は、どちらかを中間色や低彩度に変えてみましょう。"
    return "👍 特に問題のない組み合わせです。自信を持ってコーディネートを楽しみましょう！"

# ========================
# 季節のカラーパレット (HSV形式)
# ========================
season_palettes = {
    "春": [  # 明るく柔らかい色・黄味系が多め
        (25, 60, 240),   # ペールアプリコット
        (35, 90, 230),   # コーラル
        (90, 40, 220),   # パステルグリーン
        (160, 70, 240),  # チェリーピンク
        (20, 40, 210),   # ピーチベージュ
        (0, 0, 250),     # アイボリー
        (120, 30, 200)   # スカイブルー
    ],
    "夏": [  # くすみ系・寒色寄り・透明感重視
        (130, 40, 230),  # ラベンダー
        (95, 50, 220),   # ミントグリーン
        (160, 60, 250),  # フューシャピンク
        (0, 0, 245),     # ホワイトグレー
        (115, 30, 210),  # アイスブルー
        (140, 50, 200),  # スモーキーパープル
        (30, 50, 200)    # シャンパンベージュ
    ],
    "秋": [  # 温かみのある濃い色・黄〜赤系が主軸
        (15, 180, 160),  # テラコッタ
        (25, 150, 140),  # マスタード
        (10, 80, 80),    # オリーブカーキ
        (40, 130, 110),  # キャメルブラウン
        (0, 100, 180),   # 焦げ赤
        (0, 0, 180),     # ウォームグレー
        (100, 60, 130)   # ディープグリーン
    ],
    "冬": [  # 高彩度・コントラスト強め・寒色とモノトーン
        (120, 200, 90),  # ロイヤルブルー
        (140, 180, 80),  # バーガンディ
        (0, 0, 255),     # ピュアホワイト
        (0, 0, 30),      # ブラック
        (160, 160, 255), # ビビッドパープル
        (110, 120, 70),  # ダークグリーン
        (0, 150, 250)    # ビビッドレッド
    ]
}
# ========================
# 代替カラーを提案する関数
# ========================
def generate_alternative_colors(fixed_color_bgr, season, is_top):
    """
    固定色と季節に基づき、相性の良い代替色を生成する。
    入力色に応じてベースカラーを動的に変更し、提案の多様性と質を向上させる。
    :param fixed_color_bgr: 基準となる色 (BGR)
    :param season: "春", "夏", "秋", "冬" または "選択なし"
    :param is_top: Trueならトップス、Falseならボトムスの色を提案
    :return: (提案色BGR, 判定結果) のタプルのリスト
    """
    suggestions = []
    h, s, v = cv2.cvtColor(np.uint8([[fixed_color_bgr]]), cv2.COLOR_BGR2HSV)[0][0]

    # ===== 変更点: 固定色の明るさに応じてベースカラーを動的に取得 =====
    base_colors_bgr = get_dynamic_base_colors(v)
    
    # BGRからHSVに変換して候補リストの初期値とする
    candidate_hsvs = [
        tuple(cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]) 
        for bgr in base_colors_bgr
    ]
    
    # 提案に含める判定結果を場合分け
    if season == "選択なし":
        allowed_keywords = ["無難", "控えめ"]
        
        # 1. 固定色に近い色を探索
        for delta_h in [-120, -90, -60, -30, 0, 30, 60, 90, 120]:
            for delta_s in [-120, -90, -60, -30, 0, 30, 60, 90, 120]:
                for delta_v in [-120, -90, -60, -30, 0, 30, 60, 90, 120]:
                    nh = (int(h) + delta_h) % 180
                    ns = np.clip(int(s) + delta_s, 30, 255)
                    nv = np.clip(int(v) + delta_v, 30, 255)
                    candidate_hsvs.append((nh, ns, nv))

        # 2. 補色（反対色）を候補に追加
        comp_h = (int(h) + 90) % 180
        candidate_hsvs.append((comp_h, np.clip(s, 100, 200), np.clip(v, 100, 200)))
        candidate_hsvs.append((comp_h, 80, 150))
        
    else: # 季節選択時
        allowed_keywords = ["無難", "控えめ", "許容範囲"]
        for base_hsv in season_palettes[season]:
            for delta_v in [-40, 0, 40]:
                nh, ns, nv = base_hsv
                nv = np.clip(nv + delta_v, 30, 255)
                candidate_hsvs.append((nh, ns, nv))

    # 候補色と固定色の組み合わせを判定
    for hsv in set(candidate_hsvs):
        new_bgr_tuple = tuple(int(c) for c in cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2BGR)[0][0])
        
        top_color = new_bgr_tuple if is_top else fixed_color_bgr
        bottom_color = fixed_color_bgr if is_top else new_bgr_tuple
        
        judgment = color_combination_level_improved(top_color, bottom_color)

        if any(word in judgment for word in allowed_keywords):
            suggestions.append((new_bgr_tuple, judgment))

    # 重複を除き、最大5件を返す
    unique_suggestions = list({s[0]: s for s in suggestions}.values())
    return unique_suggestions[:5]# ========================
# 【NEW】動的にベースカラーを生成する関数
# ========================
def get_dynamic_base_colors(v_value):
    """
    入力色の明度(v_value)に基づき、相性の良いベースカラーのリストを動的に生成する。
    :param v_value: 入力色の明度 (0-255)
    :return: BGR色のタプルのリスト
    """
    # 明るさに関わらず使いやすい定番色
    staple_colors = [
        (205, 220, 235),  # ベージュ
        (130, 70, 20),    # デニムブルー
    ]
    
    # 入力色が明るい場合 (v_value > 170) は、暗い色をベースにする
    if v_value > 170:
        dynamic_neutrals = [
            (128, 128, 128),  # ミドルグレー
            (80, 40, 0),      # ネイビー
            (50, 50, 50),     # チャコールグレー
        ]
    # 入力色が暗い場合 (v_value < 85) は、明るい色をベースにする
    elif v_value < 85:
        dynamic_neutrals = [
            (245, 245, 245),  # オフホワイト
            (210, 210, 210),  # ライトグレー
        ]
    # 中間の明るさの場合は、標準的な組み合わせ
    else:
        dynamic_neutrals = [
            (245, 245, 245),  # オフホワイト
            (128, 128, 128),  # ミドルグレー
            (50, 50, 50),     # チャコールグレー
        ]
        
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

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        h, w, _ = img_bgr.shape
        result = pose.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        if result.pose_landmarks:
            lm = result.pose_landmarks.landmark
            def to_pixel(p): return int(p.x * w), int(p.y * h)

            sL, sR = to_pixel(lm[mp_pose.PoseLandmark.LEFT_SHOULDER]), to_pixel(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER])
            hL, hR = to_pixel(lm[mp_pose.PoseLandmark.LEFT_HIP]), to_pixel(lm[mp_pose.PoseLandmark.RIGHT_HIP])
            kL, kR = to_pixel(lm[mp_pose.PoseLandmark.LEFT_KNEE]), to_pixel(lm[mp_pose.PoseLandmark.RIGHT_KNEE])

            x1, y1 = min(sL[0], sR[0]), min(sL[1], sR[1])
            x2, y2 = max(hL[0], hR[0]), max(hL[1], hR[1])
            y3 = max(kL[1], kR[1])

            # 領域の座標がマイナスになったり、高さが0にならないように調整
            y1, y2, y3 = max(0, y1), max(y1 + 10, y2), max(y2 + 10, y3)
            x1, x2 = max(0, x1), max(x1 + 10, x2)

            top_region = img_bgr[y1:y2, x1:x2]
            bottom_region = img_bgr[y2:y3, x1:x2]

            if top_region.size == 0 or bottom_region.size == 0:
                st.error("⚠️ 服装の領域を検出できませんでした。別の画像をお試しください。")
            else:
                top_color = get_dominant_color(top_region)
                bottom_color = get_dominant_color(bottom_region)
                
                # --- 結果表示 ---
                st.image(image, caption="アップロード画像", use_column_width=True)

                def create_color_chip_html(bgr_color):
                    rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])
                    return f"<div style='display:inline-block; width:20px; height:20px; background-color:rgb{rgb_color}; border:1px solid #ccc; margin-right:8px; vertical-align:middle;'></div>"

                st.markdown(f"{create_color_chip_html(top_color)} **トップスの代表色**", unsafe_allow_html=True)
                st.markdown(f"{create_color_chip_html(bottom_color)} **ボトムスの代表色**", unsafe_allow_html=True)
                
                judgment = color_combination_level_improved(top_color, bottom_color)
                st.markdown(f"### 🎨 判定結果\n{judgment}")
                st.markdown(f"### 💬 アドバイス\n{get_advice(judgment)}")
                
                st.markdown("---")

                # --- 代替案の表示 ---
                with st.expander("💡 別のコーディネート提案を見る"):
                    # トップス提案
                    top_suggestions = generate_alternative_colors(bottom_color, season, is_top=True)
                    if top_suggestions:
                        st.markdown("##### 👕 今のボトムスに合わせるなら？ (トップスの提案)")
                        for color, j in top_suggestions:
                            html = f"{create_color_chip_html(bottom_color)} + {create_color_chip_html(color)} &rarr; {j}"
                            st.markdown(html, unsafe_allow_html=True)
                    else:
                        st.markdown("##### 👕 今のボトムスに合うトップスの提案は見つかりませんでした。")
                    
                    st.markdown("---")

                    # ボトムス提案
                    bottom_suggestions = generate_alternative_colors(top_color, season, is_top=False)
                    if bottom_suggestions:
                        st.markdown("##### 👖 今のトップスに合わせるなら？ (ボトムスの提案)")
                        for color, j in bottom_suggestions:
                            html = f"{create_color_chip_html(top_color)} + {create_color_chip_html(color)} &rarr; {j}"
                            st.markdown(html, unsafe_allow_html=True)
                    else:
                        st.markdown("##### 👖 今のトップスに合うボトムスの提案は見つかりませんでした。")

        else:
            st.error("⚠️ 人物が検出できませんでした。全身が写っている画像をお試しください。")