import sys
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

# Machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs, make_classification, make_moons
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

app = FastAPI(title="ML Top 10 Algorithms Interactive Learning API")

# Add CORS Middleware to connect with Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify Next.js origin e.g. ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Schemas ---

class TrainRequest(BaseModel):
    algo_id: str
    params: Dict[str, Any]

class RecommendationRequest(BaseModel):
    labeled: bool
    data_type: str  # "tabular" or "text"
    size: str        # "small" or "large"
    interpretability: bool
    non_linear: bool

class QuizAnswer(BaseModel):
    algo_id: str
    question_idx: int
    selected_option: int

# --- Mock Datasets / Data Generators ---

def generate_classification_data(noise=0.2, random_state=42):
    # Generates a non-linear 2D dataset (moons) for class separation
    X, y = make_moons(n_samples=200, noise=noise, random_state=random_state)
    # Scale feature dimensions
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def generate_linear_data(random_state=42):
    # Generates 1D features for simple linear regression plotting
    np.random.seed(random_state)
    x = np.random.rand(100, 1) * 10
    noise = np.random.normal(0, 1.2, (100, 1))
    y = 2.5 * x + 4.0 + noise
    return x, y.flatten()

def generate_cluster_data(random_state=42):
    # Generates 3 blobs for clustering
    X, y = make_blobs(n_samples=150, centers=3, n_features=2, cluster_std=0.8, random_state=random_state)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def generate_pca_data(random_state=42):
    # Generates highly correlated 2D points to illustrate projection
    np.random.seed(random_state)
    x = np.random.normal(0, 1, 150)
    y = 1.5 * x + np.random.normal(0, 0.4, 150)
    X = np.column_stack((x, y))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

# --- Quiz Database ---

QUIZ_DB = {
    "linear_regression": [
        {
            "question": "線性迴歸的最佳化目標通常是最小化以下哪一項？ (What does Linear Regression optimize?)",
            "options": [
                "最大概似估計 (Maximum Likelihood)",
                "殘差平方和 RSS (Residual Sum of Squares)",
                "交叉熵損失 (Cross-Entropy Loss)",
                "基尼係數 (Gini Impurity)"
            ],
            "correct": 1,
            "explanation": "線性迴歸通常採用最小平方法（OLS），目標是最小化殘差平方和 (RSS)。"
        }
    ],
    "logistic_regression": [
        {
            "question": "邏輯迴歸是如何將連續輸出限制在 (0, 1) 區間以代表機率的？ (How does Logistic Regression output probabilities?)",
            "options": [
                "使用 Step 階躍函數",
                "使用 Softmax 函數",
                "使用 Sigmoid 激活函數",
                "使用 ReLU 修正線性單元"
            ],
            "correct": 2,
            "explanation": "邏輯迴歸透過 Sigmoid 函數 $\\sigma(w^T x) = \\frac{1}{1 + e^{-w^T x}}$ 將線性預測值映射到 (0, 1) 區間。"
        }
    ],
    "knn": [
        {
            "question": "當特徵維度 d 極高時，高維空間點與點的距離趨於相等，這稱為什麼現象？ (What curse affects KNN in high dimensions?)",
            "options": [
                "維度災難 (Curse of Dimensionality)",
                "過擬合 (Overfitting)",
                "偏誤-變異數權衡 (Bias-Variance Tradeoff)",
                "梯度消失 (Gradient Vanishing)"
            ],
            "correct": 0,
            "explanation": "當維度極高時，距離度量會失效，這被稱為「維度災難（Curse of Dimensionality）」，使用 KNN 前需進行降維。"
        }
    ],
    "svm": [
        {
            "question": "當數據在原始低維空間非線性可分時，SVM 使用什麼技術將數據映射至高維特徵空間？ (How does SVM handle non-linear data?)",
            "options": [
                "正規化項 (Regularization)",
                "核函數技術 (Kernel Trick)",
                "自助抽樣法 (Bootstrap)",
                "資訊增益 (Information Gain)"
            ],
            "correct": 1,
            "explanation": "SVM 透過核函數（Kernel Trick，如 RBF 核）將低維非線性數據映射到高維，使其在高維空間線性可分。"
        }
    ],
    "naive_bayes": [
        {
            "question": "樸素貝氏演算法中的「樸素 (Naive)」是指哪一個假設？ (What does 'Naive' stand for?)",
            "options": [
                "假設數據一定是常態分佈 (Normal distribution)",
                "假設所有類別權重相同 (Equal weights)",
                "假設給定類別條件下，所有特徵相互獨立 (Conditional Independence)",
                "假設不需要進行特徵標準化"
            ],
            "correct": 2,
            "explanation": "樸素貝氏作出了極其強烈的假設：在給定類別 Y 的條件下，所有特徵 X 彼此獨立，因此條件機率可拆解為乘積。"
        }
    ],
    "decision_tree": [
        {
            "question": "為了防止決策樹無限制生長導致「過擬合」，最常見的限制參數是？ (How to prevent overfitting in Decision Tree?)",
            "options": [
                "極小化資訊增益",
                "限制樹的最大深度 (max_depth)",
                "加入 L1 正則化",
                "增加樣本隨機性"
            ],
            "correct": 1,
            "explanation": "決策樹極易過擬合，因此需要剪枝或限制最大深度（max_depth）以控制其複雜度。"
        }
    ],
    "random_forest": [
        {
            "question": "隨機森林是基於以下哪種集成學習技術建構的？ (What ensemble method does Random Forest use?)",
            "options": [
                "Boosting (梯度提升)",
                "Stacking (堆疊)",
                "Bagging (自助抽樣集成)",
                "Voting (多數決投票)"
            ],
            "correct": 2,
            "explanation": "隨機森林是基於 Bagging (Bootstrap Aggregating) 技術，並結合了特徵隨機性來降低模型變異數。"
        }
    ],
    "xgboost": [
        {
            "question": "關於 XGBoost 的特性，以下哪項敘述是正確的？ (Which is correct about XGBoost?)",
            "options": [
                "它與隨機森林一樣是平行建立樹木",
                "它的損失函數中加入了一階和二階泰勒展開以及結構正則化項",
                "它只適用於小規模數據集",
                "它是非監督式學習演算法"
            ],
            "correct": 1,
            "explanation": "XGBoost 屬於 Boosting 陣營（序列化擬合殘差），其目標函數加入了二階泰勒展開並顯式引入結構正則化項。"
        }
    ],
    "kmeans": [
        {
            "question": "K-Means 演算法在選擇初始中心點時，為了避免陷入局部最優，推薦使用什麼先進初始化演算法？ (Which initialization is recommended for K-Means?)",
            "options": [
                "隨機選擇 (Random Initialization)",
                "k-means++",
                "PCA 初始化",
                "最大距離初始化"
            ],
            "correct": 1,
            "explanation": "k-means++ 是一種優化初始中心點位置的演算法，能顯著提升收斂速度並避免陷入差的局部最優。"
        }
    ],
    "pca": [
        {
            "question": "PCA (主成分分析) 降維的優化核心是？ (What is the optimization objective of PCA?)",
            "options": [
                "最大化降維後數據的正交方差 (Maximize Variance)",
                "最小化交叉熵",
                "最大化類別間的幾何間隔",
                "完全消除異常值"
            ],
            "correct": 0,
            "explanation": "PCA 的優化核心是在降維後的空間中，最大化數據的正交方差，以保留最核心的資訊量。"
        }
    ]
}

# --- Core Learning Info DB (Module 1 - 5) ---

ALGO_INFO = {
    "linear_regression": {
        "id": "linear_regression",
        "name": "線性迴歸 (Linear Regression)",
        "module": "Module 2: 預測與幾何分類",
        "formula": "\\hat{y} = w_0 + w_1x_1 + w_2x_2 + \\dots + w_dx_d = W^T X",
        "optimization": "最小化殘差平方和 (RSS): J(W) = \\sum_{i=1}^{n} (y_i - W^T x_i)^2",
        "explanation": "線性迴歸假設應變數與自變數之間存在線性關係。優化目標為最小化殘差平方和 (RSS)。當矩陣不可逆時，引入脊迴歸 (L2) 或 Lasso 迴歸 (L1)。",
        "python_code": """import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 建立模擬數據
X = np.random.rand(100, 3)
y = 3.5 * X[:, 0] + 2.0 * X[:, 1] + np.random.normal(0, 0.1, 100)

# 模型宣告與擬合
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

print("權重參數 (Weights):", model.coef_)
print("截距 (Intercept):", model.intercept_)""",
        "use_cases": "房地產估價、財務營收預測、廣告投放 ROI 評估。",
        "pros": ["計算速度極快，適合大規模數據", "解釋性極佳，權重直接代表特徵影響力"],
        "cons": ["無法有效捕捉非線性關係", "對異常值 (Outliers) 極度敏感"]
    },
    "logistic_regression": {
        "id": "logistic_regression",
        "name": "邏輯迴歸 (Logistic Regression)",
        "module": "Module 2: 預測與幾何分類",
        "formula": "P(y=1|x) = \\sigma(W^T X) = \\frac{1}{1 + e^{-W^T X}}",
        "optimization": "交叉熵損失 (Cross-Entropy Loss): J(W) = -\\frac{1}{n} \\sum_{i=1}^{n} [y_i \\log(p_i) + (1-y_i) \\log(1-p_i)]",
        "explanation": "邏輯迴歸本質上是分類演算法。它藉由 Sigmoid 函數將線性迴歸的連續輸出映射至 (0, 1) 區間，用以代表二分類的條件機率。損失函數採用交叉熵損失。",
        "python_code": """from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 初始化邏輯迴歸模型，使用 L2 正規化
log_reg = LogisticRegression(penalty='l2', solver='lbfgs')
log_reg.fit(X_train_scaled, y_train)
y_pred = log_reg.predict(X_test_scaled)

print(classification_report(y_test, y_pred))""",
        "use_cases": "銀行信用違約預測、電商廣告點擊率預測 (CTR)、疾病確診機率評估。",
        "pros": ["輸出具有機率含意，便於設定業務閾值", "易於平行化計算，線上部署極其輕量"],
        "cons": ["特徵空間高度非線性時表現不佳", "容易受到特徵高度共線性的影響"]
    },
    "knn": {
        "id": "knn",
        "name": "K-近鄰演算法 (KNN)",
        "module": "Module 2: 預測與幾何分類",
        "formula": "d(x, q) = \\sqrt{\\sum_{j=1}^{d} (x_j - q_j)^2}",
        "optimization": "Lazy Learning (無顯式優化目標，基於多數決投票)",
        "explanation": "KNN 屬於惰性學習（Lazy Learning），沒有明確的訓練階段。其核心邏輯為「物以類聚」。給定未知樣本，在訓練集中檢索距離最近的 K 個樣本，並進行多數決投票。",
        "python_code": """from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# KNN 對尺度敏感，必須進行特徵標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn.fit(X_train_scaled, y_train)""",
        "use_cases": "地理位置鄰近商圈推薦、電商初期推薦、遺失值插補 (KNN Imputer)。",
        "pros": ["理論簡單，完全無需對數據分佈做任何先驗假設", "自然支持多分類任務"],
        "cons": ["計算複雜度高，每次預測皆需掃描全數據集", "記憶體開銷巨大，易受維度災難影響"]
    },
    "svm": {
        "id": "svm",
        "name": "支援向量機 (Support Vector Machine, SVM)",
        "module": "Module 2: 預測與幾何分類",
        "formula": "K(x, z) = \\exp(-\\gamma ||x - z||^2) \\quad (\\text{RBF Kernel})",
        "optimization": "\\min_{W, b} \\frac{1}{2} ||W||^2 \\quad \\text{subject to } y_i(W^T x_i + b) \\ge 1",
        "explanation": "SVM 的目標是在特徵空間中尋找一個幾何間隔最大化的超平面來區分不同類別。當數據非線性可分時，SVM 透過核函數將數據映射到高維空間。",
        "python_code": """from sklearn.svm import SVC

# 使用高斯核函數 (RBF) 的支援向量機
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train_scaled, y_train)""",
        "use_cases": "手寫字體辨識、文本分類、生物資訊學基因序列分類。",
        "pros": ["在高維、稀疏特徵空間中表現極其優異", "僅依賴支援向量，具備極強的抗噪能力"],
        "cons": ["對大規模數據訓練時間過長", "對核函數與超參數 C 的選擇極度敏感"]
    },
    "naive_bayes": {
        "id": "naive_bayes",
        "name": "樸素貝氏 (Naive Bayes)",
        "module": "Module 2: 預測與幾何分類",
        "formula": "P(X|Y) = P(x_1, x_2, \\dots, x_d|Y) = \\prod_{j=1}^{d} P(x_j|Y)",
        "optimization": "\\arg\\max_Y P(Y|X) = \\arg\\max_Y P(Y) \\prod_{j=1}^{d} P(x_j|Y)",
        "explanation": "樸素貝氏是基於貝氏定理的生成式模型。之所以稱為「樸素（Naive）」，是因為它假設所有特徵在給定類別條件下是完全獨立的。",
        "python_code": """from sklearn.naive_bayes import GaussianNB

# 初始化高斯樸素貝氏模型（適用於連續型特徵）
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)""",
        "use_cases": "垃圾郵件過濾 (Spam Filtering)、文本情感分析、新聞快速分類。",
        "pros": ["訓練與預測速度極快，近乎即時", "在小規模數據上表現出奇良好"],
        "cons": ["特徵獨立性假設若嚴重違背，分類效果會大幅下降", "容易受到零機率問題影響（需引入拉普拉斯平滑）"]
    },
    "decision_tree": {
        "id": "decision_tree",
        "name": "決策樹 (Decision Tree)",
        "module": "Module 3: 樹狀與集成架構",
        "formula": "\\text{Gini}(D) = 1 - \\sum_{k=1}^{K} p_k^2",
        "optimization": "\\text{Maximize Gini Gain (Gini}_D - \\text{Gini}_{\\text{split}})",
        "explanation": "決策樹透過遞迴地將特徵空間劃分為不相交的子區域。每次劃分的標準是極大化資訊增益或降低不純度（如 Gini 不純度）。",
        "python_code": """from sklearn.tree import DecisionTreeClassifier, export_text

# 限制最大深度 (max_depth) 防止過擬合
dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_model.fit(X_train, y_train)

# 列印樹狀結構
tree_rules = export_text(dt_model)
print(tree_rules[:300])""",
        "use_cases": "銀行自動審批業務流、電商規則引擎、醫學臨床診斷樹。",
        "pros": ["白盒模型，具備無與倫比的可視化與解釋性", "完全不需要對數據進行標準化或歸一化"],
        "cons": ["極度容易過擬合", "數據微小的變動會導致整棵樹結構劇烈改變（高變異數）"]
    },
    "random_forest": {
        "id": "random_forest",
        "name": "隨機森林 (Random Forest)",
        "module": "Module 3: 樹狀與集成架構",
        "formula": "\\text{Ensemble Voting: } H(x) = \\arg\\max_Y \\sum_{t=1}^{T} I(h_t(x) = Y)",
        "optimization": "Bagging + Feature Subsampling (隨機特徵選擇)",
        "explanation": "隨機森林是基於 Bagging 技術的集成模型。它透過「樣本隨機性」（自助抽樣）與「特徵隨機性」建構多棵獨立決策樹，採取多數決或平均值輸出，能大幅降低變異數。",
        "python_code": """from sklearn.ensemble import RandomForestClassifier

# 宣告隨機森林，內含 100 棵決策樹
rf_model = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=42)
rf_model.fit(X_train, y_train)

# 獲取特徵重要性
importances = rf_model.feature_importances_
print("特徵重要性分數:", importances)""",
        "use_cases": "電商推薦系統、製造業設備故障預警（預測性維護）、特徵篩選。",
        "pros": ["泛化誤差極低，幾乎不需要繁瑣的調參即可獲得高準確度", "能有效處理高維度、具缺失值的非線性數據"],
        "cons": ["失去單一決策樹的直觀解釋性（轉為黑盒模型）", "樹木數量多時，儲存空間與預測延遲會增加"]
    },
    "xgboost": {
        "id": "xgboost",
        "name": "XGBoost (Gradient Boosting)",
        "module": "Module 3: 樹狀與集成架構",
        "formula": "\\text{Obj}^{(t)} \\approx \\sum_{i=1}^{n} [g_i f_t(x_i) + \\frac{1}{2} h_i f_t^2(x_i)] + \\gamma T + \\frac{1}{2}\\lambda \\sum_{j=1}^{T} w_j^2",
        "optimization": "Boosting (序列擬合殘差) + 泰勒二階展開 + 正則化 penalty",
        "explanation": "XGBoost 對梯度提升機 (GBDT) 進行了革命性優化。每一棵新樹都在擬合前一棵樹留下的殘差。目標函數加入了二階泰勒展開，並顯式引入結構正則化項以控制複雜度。",
        "python_code": """# 需安裝 xgboost 庫: pip install xgboost
import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label=y_train)
params = {
    'max_depth': 4,
    'eta': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': 'auc'
}
bst = xgb.train(params, dtrain, num_boost_round=50)""",
        "use_cases": "金融反欺詐、廣告點擊率精準預測、各式 Kaggle 表格數據競賽大殺器。",
        "pros": ["業界公認表格數據中準確度最高、效能最強的演算法", "內建缺失值處理與並行運算優化"],
        "cons": ["超參數繁多，調參難度與時間成本高", "容易受異常噪聲干擾而過擬合"]
    },
    "kmeans": {
        "id": "kmeans",
        "name": "K-Means 分群演算法 (K-Means Clustering)",
        "module": "Module 4: 非監督式探索與統計結構抽取",
        "formula": "\\arg\\min_{S} \\sum_{k=1}^{K} \\sum_{x \\in S_k} ||x - \\mu_k||^2",
        "optimization": "EM 迭代 (Assignment 步驟與 Update 步驟)",
        "explanation": "K-Means 旨在將樣本劃分為 K 個非重疊的簇，使得每個樣本到其所屬簇中心的群內平方和 (WCSS) 最小化。核心步驟為：分配階段 (Assignment) 與更新階段 (Update)。",
        "python_code": """from sklearn.cluster import KMeans

# 宣告 KMeans 模型，使用 k-means++ 初始化優化中心點
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_train_scaled)

print("聚類中心座標:\\n", kmeans.cluster_centers_)""",
        "use_cases": "市場客戶細分 (Customer Segmentation)、影像顏色量化壓縮、異常賬戶偵測基線。",
        "pros": ["演算法時間複雜度與數據量呈線性關係，計算速度極快", "邏輯直觀，易於向業務團隊解說"],
        "cons": ["必須預先指定分群數 K (通常需配合 Elbow Method 評估)", "對初始中心點敏感，且只能識別凸形（圓形）簇"]
    },
    "pca": {
        "id": "pca",
        "name": "主成分分析 (Principal Component Analysis, PCA)",
        "module": "Module 4: 非監督式探索與統計結構抽取",
        "formula": "\\Sigma v = \\lambda v \\quad (\\text{Eigenvalue Decomposition})",
        "optimization": "最大化投影空間中的正交方差 (Variance)",
        "explanation": "PCA 是一種正交線性轉換，用於將高維數據投影到低維空間。其優化核心是：在降維後的空間中，最大化數據的正交方差，以保留最核心的資訊量。",
        "python_code": """from sklearn.decomposition import PCA

# 降維至 2 個主成分以利視覺化展示
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

print("各主成分解釋方差比例:", pca.explained_variance_ratio_)""",
        "use_cases": "萬維多特徵數據降維、高維數據二維/三維視覺化展示、深度學習前置特徵降噪。",
        "pros": ["能有效消除特徵間的共線性 (Multicollinearity)", "降低下游模型的計算與儲存開銷"],
        "cons": ["降維後的新主成分 (PC) 失去了原始特徵的業務字面含意", "無法捕捉非線性的流形結構 (Manifold Structure)"]
    }
}

# --- API Route Handlers ---

@app.get("/api/algorithms")
def get_algorithms():
    return list(ALGO_INFO.values())

@app.get("/api/quiz/{algo_id}")
def get_quiz(algo_id: str):
    if algo_id not in QUIZ_DB:
        raise HTTPException(status_code=404, detail="Algorithm quiz not found")
    # Return questions without the correct index (for cheating prevention)
    questions = []
    for idx, q in enumerate(QUIZ_DB[algo_id]):
        questions.append({
            "question_idx": idx,
            "question": q["question"],
            "options": q["options"]
        })
    return questions

@app.post("/api/quiz/submit")
def submit_quiz_answer(ans: QuizAnswer):
    algo_id = ans.algo_id
    if algo_id not in QUIZ_DB or ans.question_idx >= len(QUIZ_DB[algo_id]):
        raise HTTPException(status_code=404, detail="Question not found")
    
    q = QUIZ_DB[algo_id][ans.question_idx]
    is_correct = (ans.selected_option == q["correct"])
    return {
        "correct": is_correct,
        "correct_option": q["correct"],
        "explanation": q["explanation"]
    }

@app.post("/api/recommend")
def recommend_algorithm(req: RecommendationRequest):
    # Rule engine matching Module 5 Decision Logic
    rec_id = ""
    reason = []

    if not req.labeled:
        # Unlabeled data -> PCA or K-Means
        if req.size == "large":
            rec_id = "kmeans"
            reason.append("數據無標籤且規模較大，最適合使用 K-Means 進行聚類切分出業務輪廓。")
        else:
            rec_id = "pca"
            reason.append("無標籤數據，適合先使用 PCA 進行降噪與 2D/3D 可視化，再觀察結構特徵。")
    elif req.data_type == "text":
        rec_id = "naive_bayes"
        reason.append("處理文本分類或垃圾郵件過濾任務時，樸素貝氏 (Naive Bayes) 作為第一道快速過濾基線是業界首選。")
    elif req.interpretability and not req.non_linear:
        # Labeled, tabular, needs explain, linear relationship
        rec_id = "linear_regression" if req.size == "small" else "logistic_regression"
        reason.append("數據量極小或需要即時解釋原因，且關係偏線性。優先考慮線性迴歸（預測）或邏輯迴歸（分類）。")
    elif req.non_linear:
        # Labeled, tabular, non-linear
        if req.size == "large":
            if HAS_XGB:
                rec_id = "xgboost"
                reason.append("標籤表格數據且存在非線性關係，規模較大。優先使用 XGBoost，準確度通常封頂。")
            else:
                rec_id = "random_forest"
                reason.append("標籤表格數據且關係非線性。優先使用集成樹模型（隨機森林），具備極佳的泛化效能。")
        else:
            rec_id = "decision_tree"
            reason.append("數據規模中等，關係非線性，且需要白盒模型解釋決策流程。推薦使用決策樹 (Decision Tree)。")
    else:
        # Default fallback
        rec_id = "knn"
        reason.append("特徵關係簡單，無特定強烈假設，推薦使用簡單直觀的 KNN 作為基準線模型。")

    recommended_algo = ALGO_INFO.get(rec_id, ALGO_INFO["linear_regression"])
    return {
        "algo_id": rec_id,
        "name": recommended_algo["name"],
        "reason": " \n".join(reason)
    }

@app.post("/api/train")
def train_model(req: TrainRequest):
    algo = req.algo_id
    params = req.params

    try:
        # 1. Generate appropriate mock data based on algorithm type
        if algo == "linear_regression":
            X, y = generate_linear_data()
            model = LinearRegression()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)

            # Metrics
            train_preds = model.predict(X_train)
            test_preds = model.predict(X_test)
            mse = float(np.mean((y_test - test_preds) ** 2))
            r2 = float(model.score(X_test, y_test))

            # Return raw points
            points = [{"x1": float(X[i][0]), "x2": 0.0, "label": float(y[i])} for i in range(len(X))]
            
            # Regression line
            line_x = np.linspace(float(X.min()), float(X.max()), 100).reshape(-1, 1)
            line_y = model.predict(line_x)
            line_points = [{"x1": float(line_x[i][0]), "y": float(line_y[i])} for i in range(100)]

            return {
                "metrics": {"MSE": mse, "R2 Score": r2},
                "details": f"Weights (Slope): {model.coef_.tolist()}, Intercept: {float(model.intercept_)}",
                "points": points,
                "line": line_points
            }

        elif algo == "kmeans":
            X, y = generate_cluster_data()
            n_clusters = int(params.get("n_clusters", 3))
            init_method = params.get("init", "k-means++")
            
            model = KMeans(n_clusters=n_clusters, init=init_method, random_state=42, n_init=10)
            model.fit(X)
            preds = model.labels_

            points = [{"x1": float(X[i][0]), "x2": float(X[i][1]), "label": int(preds[i])} for i in range(len(X))]
            centroids = [{"x1": float(c[0]), "x2": float(c[1]), "label": idx} for idx, c in enumerate(model.cluster_centers_)]
            
            # Evaluate WCSS (Inertia)
            wcss = float(model.inertia_)

            return {
                "metrics": {"WCSS (Inertia)": wcss},
                "details": f"Centroids initialized using {init_method}.",
                "points": points,
                "centroids": centroids
            }

        elif algo == "pca":
            X = generate_pca_data()
            model = PCA(n_components=2)
            model.fit(X)
            X_trans = model.transform(X)

            points = [{"x1": float(X[i][0]), "x2": float(X[i][1]), "label": 0} for i in range(len(X))]
            
            # Variance Ratio
            var_ratio = model.explained_variance_ratio_.tolist()
            components = model.components_.tolist()
            mean = model.mean_.tolist()

            # Render PCA axes lines starting from mean
            axes = []
            for i, comp in enumerate(components):
                # Scale components by standard deviation (sqrt of eigenvalues/explained variance)
                length = np.sqrt(model.explained_variance_[i]) * 2.0
                end_pt = [mean[0] + comp[0] * length, mean[1] + comp[1] * length]
                axes.append({
                    "component_idx": i,
                    "start": {"x1": mean[0], "x2": mean[1]},
                    "end": {"x1": end_pt[0], "x2": end_pt[1]},
                    "variance_ratio": var_ratio[i]
                })

            return {
                "metrics": {
                    "PC1 Var Ratio": var_ratio[0],
                    "PC2 Var Ratio": var_ratio[1]
                },
                "details": f"Data mean: {mean}, Components: {components}",
                "points": points,
                "axes": axes
            }

        else:
            # Classification algorithms (Logistic Regression, KNN, SVM, Naive Bayes, Decision Tree, RF, XGBoost)
            X, y = generate_classification_data()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if algo == "logistic_regression":
                penalty = params.get("penalty", "l2")
                C = float(params.get("C", 1.0))
                model = LogisticRegression(penalty=penalty, C=C, solver="lbfgs")
            elif algo == "knn":
                n_neighbors = int(params.get("n_neighbors", 5))
                weights = params.get("weights", "uniform")
                model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
            elif algo == "svm":
                C = float(params.get("C", 1.0))
                kernel = params.get("kernel", "rbf")
                model = SVC(C=C, kernel=kernel, probability=True, random_state=42)
            elif algo == "naive_bayes":
                model = GaussianNB()
            elif algo == "decision_tree":
                max_depth = params.get("max_depth", None)
                if max_depth is not None:
                    max_depth = int(max_depth)
                model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
            elif algo == "random_forest":
                n_estimators = int(params.get("n_estimators", 100))
                max_depth = params.get("max_depth", None)
                if max_depth is not None:
                    max_depth = int(max_depth)
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            elif algo == "xgboost":
                if HAS_XGB:
                    n_estimators = int(params.get("n_estimators", 50))
                    max_depth = int(params.get("max_depth", 4))
                    learning_rate = float(params.get("learning_rate", 0.1))
                    model = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, eval_metric="logloss", random_state=42)
                else:
                    # Fallback to Gradient Boosting Classifier if XGBoost isn't loaded
                    from sklearn.ensemble import GradientBoostingClassifier
                    n_estimators = int(params.get("n_estimators", 50))
                    max_depth = int(params.get("max_depth", 4))
                    learning_rate = float(params.get("learning_rate", 0.1))
                    model = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)
            else:
                raise HTTPException(status_code=400, detail="Invalid algorithm ID")

            model.fit(X_train, y_train)
            acc = float(model.score(X_test, y_test))

            # Generate grid mesh for boundary mapping
            # Get boundaries
            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
            
            # Create a 40x40 meshgrid
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 40), np.linspace(y_min, y_max, 40))
            grid_points = np.c_[xx.ravel(), yy.ravel()]
            
            # Predict labels or probability on meshgrid
            if hasattr(model, "predict_proba"):
                grid_preds = model.predict_proba(grid_points)[:, 1] # Probability of class 1
            else:
                grid_preds = model.predict(grid_points).astype(float)
                
            mesh_points = []
            for i in range(len(grid_points)):
                mesh_points.append({
                    "x1": float(grid_points[i][0]),
                    "x2": float(grid_points[i][1]),
                    "val": float(grid_preds[i]) # represents probability or class boundary
                })

            points = [{"x1": float(X[i][0]), "x2": float(X[i][1]), "label": int(y[i])} for i in range(len(X))]

            # Text details
            details_str = ""
            if algo == "decision_tree":
                details_str = "樹的文字規則前 5 層:\n" + export_text(model, max_depth=3)
            elif algo == "logistic_regression":
                details_str = f"Coefficients (Weights): {model.coef_.tolist()}, Intercept: {model.intercept_.tolist()}"
            elif algo == "random_forest" or algo == "xgboost":
                # feature importances
                details_str = f"Feature Importances (Feature 1, Feature 2): {model.feature_importances_.tolist()}"
            else:
                details_str = f"Model parameters: {model.get_params()}"

            return {
                "metrics": {"Accuracy": acc},
                "details": details_str,
                "points": points,
                "mesh": mesh_points
            }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Model execution error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
