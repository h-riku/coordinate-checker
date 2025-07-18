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
    # 白以外のピクセルを抽出
    data = data[np.any(data != [255, 255, 255], axis=1)]
    if len(data) == 0:
        # 白以外のピクセルがない場合は白を返す
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
    h_diff = min(h_diff, 180 - h_diff) # 色相差は180度を超えないように調整
    s_avg, v_avg = (s1 + s2) / 2, (v1 + v2) / 2

    # 以下、色の組み合わせ判定ロジック
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

    # 判定に基づいてスコア範囲を設定
    if "無難" in judgment:
        base_min, base_max = 85, 100
    elif "控えめ" in judgment:
        base_min, base_max = 70, 84
    elif "許容範囲" in judgment:
        base_min, base_max = 50, 69
    elif "奇抜" in judgment:
        base_min, base_max = 0, 49
    else: # 想定外の判定
        base_min, base_max = 50, 69

    # 基本スコアは範囲の中央値ではなく、より柔軟に設定
    base_score = (base_min + base_max) / 2
    offset = 0

    # スコア微調整ロジックをより詳細かつ多様に
    # 1. コントラストのバランス
    # 明度差が大きいほど、彩度差が小さい方がバランスが良い
    if v_diff > 100 and s_diff < 50: # 高コントラストで彩度控えめは好印象
        offset += 8
    elif v_diff < 30 and s_diff > 70: # 明度差が小さく、彩度差が大きいとまとまりにくい
        offset -= 8
    
    # 2. 彩度の調和
    # 似たような彩度レベルだと統一感が出やすい
    if s_diff < 20 and s_avg > 50: # 彩度が近く、ある程度彩度がある組み合わせは良い
        offset += 6
    elif s_avg > 180 and s_diff > 50: # 両方鮮やかで彩度差が大きいと派手すぎる
        offset -= 10

    # 3. 色相の調和
    if h_diff < 15: # 非常に近い色相は統一感がある
        offset += 4
    elif 60 < h_diff < 120 and (s_avg < 80 or v_avg < 80): # 補色に近いが彩度か明度が低いと落ち着く
        offset += 3
    elif 60 < h_diff < 120 and (s_avg > 150 and v_avg > 150): # 補色に近いかつ両方鮮やかだと奇抜
        offset -= 15

    # 4. 全体的な明るさ・暗さのバランス
    if (v1 > 200 and v2 < 50) or (v2 > 200 and v1 < 50): # 明るい色と暗い色の組み合わせはメリハリ
        offset += 7
    elif v_avg < 70 and s_avg < 70: # 両方暗く彩度も低いと地味すぎる場合がある
        offset -= 5

    # 5. 特定の「奇抜」判定に対する追加の減点
    if "ネオン系の組み合わせ" in judgment:
        offset -= 15
    if "高コントラスト" in judgment: # 高コントラストでもバランスが悪い場合
        offset -= 10
    if "鮮やかな類似色" in judgment:
        offset -= 8

    final_score = base_score + offset

    # スコアが設定範囲に収まるようにクリップ
    # クリップする際に、base_minとbase_maxの中間値に限定されるのではなく、
    # 範囲全体でより多様なスコアが出るように調整します。
    # final_scoreがbase_minとbase_maxの範囲をはみ出た場合にのみクリップし、
    # その範囲内であればそのままのスコアを維持します。
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
    "春": [
        (25, 60, 240),    # 明るいイエローグリーン
        (35, 90, 230),    # コーラルピンク
        (90, 40, 220),    # ライトピーチ
        (160, 70, 240),   # ライトミント
        (20, 40, 210),    # クリームイエロー
        (0, 0, 250),      # スカイブルー
        (120, 30, 200),   # ラベンダー
        (50, 80, 255),    # フレッシュグリーン  
        (10, 100, 255)    # タンポポイエロー  
    ],
    "夏": [
        (130, 40, 230),  # アクアブルー
        (95, 50, 220),   # ラベンダーブルー
        (160, 60, 250),  # クリアピンク
        (0, 0, 245),     # ロイヤルブルー
        (115, 30, 210),  # グレイッシュパープル
        (140, 50, 200),  # ミントグリーン
        (30, 50, 200),   # アイスグレー
        (200, 10, 200),  # マリンブルー  
        (100, 20, 240)   # クールレッド  
    ],
    "秋": [
        (15, 180, 160),  # テラコッタ
        (25, 150, 140),  # マスタード
        (10, 80, 80),    # オリーブグリーン
        (40, 130, 110),  # ダークブラウン
        (0, 100, 180),   # バーガンディ
        (0, 0, 180),     # フォレストグリーン
        (100, 60, 130),  # ディープパープル
        (0, 160, 100),   # パンプキンオレンジ  
        (70, 120, 150)   # モスグリーン  
    ],
    "冬": [
        (120, 200, 90),  # ロイヤルパープル
        (140, 180, 80),  # エメラルドグリーン
        (0, 0, 255),     # ピュアホワイト
        (0, 0, 30),      # ジェットブラック
        (160, 160, 255), # アイスブルー
        (110, 120, 70),  # グレープ
        (0, 150, 250),   # マゼンタ
        (255, 0, 150),   # ディープネイビー  
        (0, 255, 0)      # ブリリアントレッド  
    ]
}
# ========================
# 差し色を提案する関数
# ========================
@st.cache_data
def suggest_accent_color(color1_bgr, color2_bgr):
    hsv1 = cv2.cvtColor(np.uint8([[color1_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    hsv2 = cv2.cvtColor(np.uint8([[color2_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    avg_h = (int(hsv1[0]) + int(hsv2[0])) / 2
    accent_h1 = int((avg_h + 60) % 180) # 補色的な色相
    accent_h2 = int((avg_h + 120) % 180) # さらに異なる色相
    accent_s, accent_v = 200, 230 # 鮮やかで明るい差し色
    accent_hsv1, accent_hsv2 = (accent_h1, accent_s, accent_v), (accent_h2, accent_s, accent_v)
    accent_bgr1 = tuple(int(c) for c in cv2.cvtColor(np.uint8([[accent_hsv1]]), cv2.COLOR_HSV2BGR)[0][0])
    accent_bgr2 = tuple(int(c) for c in cv2.cvtColor(np.uint8([[accent_hsv2]]), cv2.COLOR_HSV2BGR)[0][0])
    return [accent_bgr1, accent_bgr2]

# ========================
# 2つの色が似すぎているかを判定する関数
# ========================
@st.cache_data
def is_color_too_similar(color1_bgr, color2_bgr, h_threshold=25, s_threshold=35, v_threshold=35):
    """
    2つのBGR色がHSV空間で似すぎているかを判定する。
    :param color1_bgr: 1つ目のBGR色 (タプル)
    :param color2_bgr: 2つ目のBGR色 (タプル)
    :param h_threshold: 色相差のしきい値 (0-180)
    :param s_threshold: 彩度差のしきい値 (0-255)
    :param v_threshold: 明度差のしきい値 (0-255)
    :return: 似すぎている場合はTrue、そうでない場合はFalse
    """
    hsv1 = cv2.cvtColor(np.uint8([[color1_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    hsv2 = cv2.cvtColor(np.uint8([[color2_bgr]]), cv2.COLOR_BGR2HSV)[0][0]

    h1, s1, v1 = int(hsv1[0]), int(hsv1[1]), int(hsv1[2])
    h2, s2, v2 = int(hsv2[0]), int(hsv2[1]), int(hsv2[2])

    h_diff = abs(h1 - h2)
    h_diff = min(h_diff, 180 - h_diff) # 色相は円環状なので、最小差を計算

    s_diff = abs(s1 - s2)
    v_diff = abs(v1 - v2)

    # 全ての差がしきい値以下の場合、似すぎていると判断
    return h_diff < h_threshold and s_diff < s_threshold and v_diff < v_threshold

# ========================
# 代替カラーを提案する関数
# ========================
@st.cache_data
def generate_alternative_colors(fixed_color_bgr, season, is_top):
    suggestions = [] # 最終的な提案リスト
    candidate_hsvs = [] # 探索用のHSV候補リスト
    
    h, s, v = cv2.cvtColor(np.uint8([[fixed_color_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    base_colors_bgr = get_dynamic_base_colors(v)
    candidate_hsvs.extend([tuple(cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]) for bgr in base_colors_bgr])
    
    if season == "選択なし":
        allowed_keywords = ["無難", "控えめ"] # 季節指定なしの場合はより保守的に
        # 元の色相からの変化、彩度・明度の変化を考慮した候補
        # 探索範囲を広げ、より多くのバリエーションを試す
        for delta_h in [-105, -90, -75, -60, -45, -30,-15,15,30, 45, 60, 75, 90, 105]: 
            for delta_s in [-60, -30, 0, 30, 60]:     
                for delta_v in [-60, -30, 0, 30, 60]: 
                    nh, ns, nv = (int(h) + delta_h) % 180, np.clip(int(s) + delta_s, 30, 255), np.clip(int(v) + delta_v, 30, 255)
                    candidate_hsvs.append((nh, ns, nv))
        # 補色系の候補も追加
        comp_h = (int(h) + 90) % 180
        candidate_hsvs.extend([(comp_h, np.clip(s, 100, 200), np.clip(v, 100, 200)), (comp_h, 80, 150)])
    else:
        allowed_keywords = ["無難", "控えめ", "許容範囲"] # 季節指定ありの場合は許容範囲を広げる
        # 季節のパレット色とその明度調整版を候補に追加
        for base_hsv in season_palettes[season]:
            for delta_v in [-40, 0, 40]:
                nh, ns, nv = base_hsv
                candidate_hsvs.append((nh, ns, np.clip(nv + delta_v, 30, 255)))

    # 重複を避け、許容されるキーワードを含む色のみを提案
    # 提案する色同士が似すぎないようにするためのチェックも追加
    for hsv in set(candidate_hsvs):
        new_bgr_tuple = tuple(int(c) for c in cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2BGR)[0][0])
        
        # 1. 元の色と似すぎている場合はスキップ
        if is_color_too_similar(fixed_color_bgr, new_bgr_tuple):
            continue 

        # 2. すでに提案リストにある色と似すぎている場合はスキップ
        # suggestionsリスト内の各色と比較
        is_too_similar_to_existing = False
        for existing_suggestion, _ in suggestions: # (color, judgment)のタプルからcolorだけを取り出す
            if is_color_too_similar(existing_suggestion, new_bgr_tuple):
                is_too_similar_to_existing = True
                break
        if is_too_similar_to_existing:
            continue

        # 新しい色と固定された色の組み合わせで判定
        top_color, bottom_color = (new_bgr_tuple, fixed_color_bgr) if is_top else (fixed_color_bgr, new_bgr_tuple)
        judgment = color_combination_level_improved(top_color, bottom_color)
        
        # 許容されるキーワードを含む場合のみ追加
        if any(word in judgment for word in allowed_keywords):
            suggestions.append((new_bgr_tuple, judgment))
            
        # 提案数が5件に達したらループを終了
        if len(suggestions) >= 5:
            break

    # 色が重複するものを排除し、最大5件まで返す（set()で重複排除済みだが、念のため）
    return list({s[0]: s for s in suggestions}.values())[:5]

# ========================
# 動的にベースカラーを生成する関数
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
# カラー表示用の丸い色チップHTML
# ========================
def create_color_chip_html(bgr_color, size_rem=2.5): # size_remをrem単位で指定 (デフォルト2.5rem = 40px)
    rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])
    return f"""
    <div style='
        display:inline-block;
        width:{size_rem}rem; height:{size_rem}rem;
        background-color: rgb{rgb_color};
        border-radius: 50%;
        box-shadow: 0 0 8px rgba(0,0,0,0.25);
        margin-right:0.75rem; /* 12px相当 */
        vertical-align:middle;
        border: 2px solid #eee;
    '></div>
    """

# ========================
# スコアに基づいて色を返す関数
# ========================
def get_score_color(score):
    if score >= 85:
        return "#28a745"  # Green for excellent scores
    elif score >= 70:
        return "#17a2b8"  # Teal for good scores
    elif score >= 50:
        return "#ffc107"  # Yellow/Orange for acceptable scores
    else:
        return "#dc3545"  # Red for low scores

# ========================
# Streamlit アプリ本体
# ========================
st.set_page_config(page_title="コーディネートはこーでねーと", layout="centered")
st.title("🎨コーディネートはこーでねーと")

# Global CSS for dark mode compatibility and mobile optimization
st.markdown("""
<style>
/* 基本のフォントサイズを設定し、rem単位の基準とする */
html {
    font-size: 16px; /* 1rem = 16px として設定 */
}

/* Base text color for light theme */
body {
    color: #333; /* Default text color */
}

h1 {
    font-size: 2.2rem; /* 35.2px */
}
h2 {
    font-size: 1.8rem; /* 28.8px */
}
h3 {
    font-size: 1.5rem; /* 24px */
}
h4 {
    font-size: 1.25rem; /* 20px */
}
p {
    font-size: 1rem; /* 16px */
}
small {
    font-size: 0.85rem; /* 13.6px */
}


/* Adjustments for dark mode */
@media (prefers-color-scheme: dark) {
    body {
        color: #eee; /* Light text for dark backgrounds */
    }
    h1, h2, h3, h4, h5, h6 {
        color: #eee !important; /* Ensure headings are light in dark mode */
    }
    p {
        color: #ddd !important; /* Ensure paragraphs are light in dark mode */
    }
    /* Specific adjustments for your custom HTML elements */
    .stMarkdown div[style*="font-weight:bold; color:#333;"] {
        color: #eee !important;
    }
    .stMarkdown div[style*="background-color:#f5faff;"] {
        background-color: #2a2a2a !important; /* Darker background for the score card */
        box-shadow: 0 0.625rem 1.5625rem rgba(0,0,0,0.4) !important; /* 10px 25px相当 */
    }
    .stMarkdown h2[style*="color:#1E90FF;"] {
        color: #64B5F6 !important; /* Lighter blue for dark mode headings */
    }
    .stMarkdown p[style*="color:#555;"] {
        color: #bbb !important; /* Lighter grey for advice text */
    }
    .stMarkdown small[style*="color:#555;"] {
        color: #bbb !important; /* Lighter grey for small text */
    }
    .stMarkdown h4[style*="color:#0078D7;"] {
        color: #7EC0EE !important; /* Lighter blue for subheadings */
    }
    .st-emotion-cache-1wvtx9b { /* Target the separator line */
        background-color: #555 !important;
    }
     .st-emotion-cache-10qbe6w p { /* Target info message text */
        color: #eee !important;
     }

}

/* モバイルでの調整 (例: 画面幅が小さい場合) */
@media (max-width: 768px) { /* タブレットやスマホの一般的なブレークポイント */
    h1 {
        font-size: 1.8rem; /* スマホではタイトルを少し小さく */
    }
    h2 {
        font-size: 1.5rem;
    }
    p {
        font-size: 0.95rem; /* パラグラフも少し小さく */
    }
    .stMarkdown div[style*="font-size:50px;"] { /* スコアの数字 */
        font-size: 3rem !important; /* スマホでスコアを小さく */
    }
    .stMarkdown div[style*="font-size:22px;"] { /* 判定テキスト */
        font-size: 1.15rem !important;
    }
    .stMarkdown div[style*="font-size:18px;"] { /* アドバイステキスト */
        font-size: 1rem !important;
    }
    .stMarkdown div[style*="margin-bottom:40px;"] { /* スコアカードのマージン */
        margin-bottom: 2rem !important; /* モバイルでマージンを調整 */
        padding: 1.5rem !important; /* モバイルでパディングを調整 */
    }
}

</style>
""", unsafe_allow_html=True)


season = st.selectbox("季節を選んでください (提案される色が変わります)", ["選択なし", "春", "夏", "秋", "冬"])
uploaded_file = st.file_uploader("服装画像をアップロードしてください(全身が写っている画像を推奨します)", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    with mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        h, w, _ = img_bgr.shape
        result = pose.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        # ランドマークが検出されたかどうかのチェック
        if not result.pose_landmarks:
            st.error("⚠️ 画像から人物を検出できませんでした。もっとはっきりした画像をアップロードしてください。")
            st.stop() # 処理を中断

        lm = result.pose_landmarks.landmark
        def to_pixel(p): return int(p.x * w), int(p.y * h)
        sL, sR = to_pixel(lm[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]), to_pixel(lm[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER])
        hL, hR = to_pixel(lm[mp.solutions.pose.PoseLandmark.LEFT_HIP]), to_pixel(lm[mp.solutions.pose.PoseLandmark.RIGHT_HIP])
        kL, kR = to_pixel(lm[mp.solutions.pose.PoseLandmark.LEFT_KNEE]), to_pixel(lm[mp.solutions.pose.PoseLandmark.RIGHT_KNEE])
        x1, y1 = min(sL[0], sR[0]), min(sL[1], sR[1])
        x2, y2 = max(hL[0], hR[0]), max(hL[1], hR[1])
        y3 = max(kL[1], kR[1])
        
        # 領域が画像範囲内にあるか確認し、最低限のサイズを確保
        y1, y2, y3 = max(0, y1), max(y1 + 10, y2), max(y2 + 10, y3)
        x1, x2 = max(0, x1), max(x1 + 10, x2)
        
        top_region, bottom_region = img_bgr[y1:y2, x1:x2], img_bgr[y2:y3, x1:x2]

        # 検出された領域が空でないかのチェック
        if top_region.size == 0 or bottom_region.size == 0:
            st.error("⚠️ 服装の領域を検出できませんでした。別の画像をお試しください。")
            st.stop() # 処理を中断
        else:
            top_color, bottom_color = get_dominant_color(top_region), get_dominant_color(bottom_region)
            
            st.image(image, caption="アップロード画像", use_container_width=True)

            st.markdown(f"""
            <div style='
                display:flex;
                align-items:center;
                font-size:1.1rem; /* rem単位に変更 */
                margin-top:1.25rem; /* rem単位に変更 */
                margin-bottom:0.9375rem; /* rem単位に変更 */
                font-weight:bold;
                color:#333; /* Default for light theme */
            '>
                {create_color_chip_html(top_color, 2.5)} トップスの代表色
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style='
                display:flex;
                align-items:center;
                font-size:1.1rem; /* rem単位に変更 */
                margin-bottom:1.5625rem; /* rem単位に変更 */
                font-weight:bold;
                color:#333; /* Default for light theme */
            '>
                {create_color_chip_html(bottom_color, 2.5)} ボトムスの代表色
            </div>
            """, unsafe_allow_html=True)

            # スコア＆判定カード風ボックス
            score = calculate_detailed_score(top_color, bottom_color)
            judgment = color_combination_level_improved(top_color, bottom_color)
            score_display_color = get_score_color(score) # ここでスコアの色を取得
            
            st.markdown(f"""
            <div style='
                background-color:#f5faff; /* Default for light theme */
                padding:1.875rem; /* rem単位に変更 */
                border-radius:25px;
                box-shadow: 0 0.625rem 1.5625rem rgba(30,144,255,0.2); /* rem単位に変更 */
                max-width:500px;
                margin-bottom:2.5rem; /* rem単位に変更 */
                font-family:"Helvetica Neue", Arial, sans-serif;
            '>
                <h2 style='color:#1E90FF; margin-bottom:0.625rem; font-weight: bold; font-size:1.625rem;'>コーデスコア: <span style='font-size:3.125rem; color:{score_display_color}; font-weight:bold;'>{score} 点</span></h2>
                <p style='font-weight:bold; font-size:1.375rem; margin:0.75rem 0; color:#333;'>判定: {judgment}</p>
                <p style='font-style:italic; color:#555; font-size:1.125rem; margin-top:0.875rem;'>{get_advice(judgment)}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")


            # 差し色提案
            with st.expander("💡 別のコーディネート提案を見る"):
                st.markdown("<h4 style='color:#0078D7; margin-bottom:0.75rem;'>👜 小物やアクセサリーで差し色を加えるなら？</h4>", unsafe_allow_html=True)
                accent_colors = suggest_accent_color(top_color, bottom_color)
                if accent_colors: # 差し色がある場合のみカラムを作成
                    cols = st.columns(len(accent_colors))
                    for i, ac_color in enumerate(accent_colors):
                        with cols[i]:
                            st.markdown(create_color_chip_html(ac_color, 2.5), unsafe_allow_html=True)
                            st.markdown(f"<small style='color:#555;'>差し色案 {i+1}</small>", unsafe_allow_html=True)
                else:
                    st.info("差し色の提案はありませんでした。(彩度や明度が極端であったりする場合、条件に合う代替色が見つからない場合があります。)")

                # 代替カラー提案（トップス・ボトムス）
                
                st.markdown("<h4 style='color:#0078D7; margin-top:1.5625rem; margin-bottom:0.75rem;'>👕 トップスの色を変えたい場合の提案</h4>", unsafe_allow_html=True)
                alt_tops = generate_alternative_colors(bottom_color, season, is_top=True)
                if alt_tops: # 代替トップスがある場合のみカラムを作成
                    cols = st.columns(len(alt_tops))
                    for i, (color, judgment_alt) in enumerate(alt_tops):
                        with cols[i]:
                            st.markdown(create_color_chip_html(color, 2.5), unsafe_allow_html=True)
                            st.markdown(f"<small style='color:#555;'>{judgment_alt}</small>", unsafe_allow_html=True)
                else:
                    st.info("トップスの代替色の提案はありませんでした。(彩度や明度が極端であったりする場合、条件に合う代替色が見つからない場合があります。)")

                
                st.markdown("<h4 style='color:#0078D7; margin-top:1.5625rem; margin-bottom:0.75rem;'>👖 ボトムスの色を変えたい場合の提案</h4>", unsafe_allow_html=True)
                alt_bottoms = generate_alternative_colors(top_color, season, is_top=False)
                if alt_bottoms: # 代替ボトムスがある場合のみカラムを作成
                    cols = st.columns(len(alt_bottoms))
                    for i, (color, judgment_alt) in enumerate(alt_bottoms):
                        with cols[i]:
                            st.markdown(create_color_chip_html(color, 2.5), unsafe_allow_html=True)
                            st.markdown(f"<small style='color:#555;'>{judgment_alt}</small>", unsafe_allow_html=True)
                else: 
                    st.info("ボトムスの代替色の提案はありませんでした。(彩度や明度が極端であったりする場合、条件に合う代替色が見つからない場合があります。)")