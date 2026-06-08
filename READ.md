# 十大核心機器學習演算法動態學習平台 (Top 10 ML Algorithms Learning Hub)

本專案是一個基於 **Next.js (前端)** 與 **FastAPI (後端)** 打造的互動式機器學習演算法學習平台。平台旨在將機器學習教科書中的公式、程式碼與理論，轉化為直觀的「**動態參數沙盒**」與「**決策邊界繪製**」工具，提供資料科學家、AI 工程師及學者沉浸式的學習體驗。

---

## 🛠️ 技術棧 (Technology Stack)

- **前端 (Frontend)**: 
  - **Next.js 14** (Pages Router, Plain JavaScript)
  - **Vanilla CSS Modules** (深色磨砂玻璃 UI / Glassmorphism 設計系統)
  - **HTML5 Canvas API** (高抗鋸齒、無痛渲染二維數據點與決策邊界)
  - **KaTeX** (高效解析與排版複雜 LaTeX 數學公式)
- **後端 (Backend)**:
  - **FastAPI** + **Uvicorn** (高性能 Python API 服務)
  - **Scikit-Learn** + **NumPy** + **Pandas** (核心機器學習演算法、資料清洗與指標評估)
  - **XGBoost** (集成學習演算法支持)

---

## 📁 系統目錄結構 (Directory Structure)

```
.
├── .agent/skills/ui-ux-pro-max/      # UI/UX 設計智能技能包
├── backend/
│   ├── .venv/                         # Python 虛擬環境
│   ├── main.py                        # FastAPI 核心邏輯與模型訓練 API
│   └── requirements.txt               # 後端依賴套件清單
├── frontend/
│   ├── package.json                   # 前端相依套件清單
│   ├── src/
│   │   ├── components/
│   │   │   ├── Sidebar.js             # 導覽與學習進度追蹤模組
│   │   │   └── InteractiveChart.js    # Canvas 圖表繪製元件 (點、線、決策邊界)
│   │   └── pages/
│   │       ├── _app.js, _document.js  # Next.js 頁面模板設定
│   │       └── index.js               # 主工作區整合頁面
│   └── src/styles/globals.css         # 磨砂玻璃視覺風格樣式檔
├── design-system/                     # UI/UX Pro Max 產出的設計規範源
│   └── ml-top-10-learning-platform/
│       ├── MASTER.md                  # 全域設計規範
│       └── pages/index.md             # 專屬首頁字體字元覆蓋 (Outfit/Inter/Fira Code)
├── READ.md                            # 專案中文總結
└── README.md                          # Github 專案總結
```

---

## 🌟 核心功能特點 (Key Features)

### 1. 完整 Module 1-5 結構教學
- **Module 1: 基礎理論** — 監督與非監督分類定義、經驗風險最小化公式、偏誤與變異數權衡 (Bias-Variance Tradeoff)。
- **Module 2: 預測與幾何分類** — 線性迴歸 (01)、邏輯迴歸 (02)、KNN (03)、SVM (04)、樸素貝氏 (05)。
- **Module 3: 樹狀與集成架構** — 決策樹 (06)、隨機森林 (07)、XGBoost (08)。
- **Module 4: 非監督探索與降維** — K-Means (09)、PCA (10)。
- **Module 5: 工程實踐與選型** — 表格、文本、線性、非線性及資料規模等多重決策條件的模型選型引導。

### 2. 動態互動沙盒 (Interactive Playground)
- **即時超參數控制**：提供 Slider 與數值輸入，可直接調整 KNN 的近鄰數 K、SVM 的錯誤懲罰係數 C 與核函數、決策樹深度、XGBoost 學習率等。
- **決策邊界 (Decision Boundary) 繪製**：點擊「擬合與更新模型」後，後端會對 40x40 網格點預測類別概率，前端 Canvas 將其轉換為柔和、漸變透明的二維分類密度熱圖。
- **特徵/聚類可視化**：
  - K-Means 可視化質心位置 (`C` 標記) 以及各點的分群顏色。
  - PCA 可視化主要特徵向量 (PC1 與 PC2) 的投影方向與解釋變異量百分比。
  - 線性迴歸可視化擬合的漸變色迴歸線。
- **終端執行輸出 (Console Output)**：控制台會同步顯示類似 Python 控制台輸出，包含模型權重 (Weights/Coefs), 截距 (Intercept)、決策樹文字規則前五層，以及模型 Accuracy / MSE 指標。

### 3. 精選程式實作與評估
- 提供講義 1:1 的 Scikit-Learn 與 XGBoost 範例程式碼，支援一鍵複製。
- 詳細整理每一種演算法的企業場景應用、優點與限制表格。

### 4. 核心概念小測驗 (Interactive Quizzes)
- 內建 10 大演算法對應的觀念檢測題，點擊選項後由 FastAPI 後端判定對錯，並提供 KaTeX 渲染的數學原理解析。

### 5. 模型選型精靈 (Selection Wizard)
- 回答 5 個關於資料類型、非線性、可解釋性與規模的問答題，系統會根據 Module 5 心法自動輸出最佳推薦模型與分析報告。

---

## 🎨 UI/UX 設計系統 (Design System)

專案整合了 **UI/UX Pro Max** 技能：
- **調色盤 (Color Palette)**:
  - 背景: 深黑藍色 `#020617` (Slate 950)
  - 邊框: 磨砂玻璃半透明 `rgba(255, 255, 255, 0.08)`
  - 指標/焦點色:Forest Green `#22c55e` 與 Sky Blue `#38bdf8`
- **字體搭配 (Typography)**:
  - 標題 Heading: `Outfit`
  - 內文 Body: `Inter`
  - 程式碼 Monospace: `Fira Code`
- **動畫與滑鼠反饋**: 所有可點擊項目均使用 `cursor-pointer`，懸停時具備柔和陰影與 `200ms` 上移過渡動畫。

---

## 🚀 快速啟動指南 (Installation & Run)

### 1. 啟動後端 FastAPI

進入 `backend` 資料夾：
```bash
cd backend
```

建立並啟動虛擬環境 (Windows PowerShell)：
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

安裝依賴套件：
```bash
pip install -r requirements.txt
```

執行後端伺服器 (將監聽在 http://127.0.0.1:8000)：
```bash
python main.py
```

### 2. 啟動前端 Next.js

開啟另一個終端機視窗，進入 `frontend` 資料夾：
```bash
cd frontend
```

安裝依賴套件：
```bash
npm install
```

啟動開發伺服器 (將監聽在 http://localhost:3000)：
```bash
npm run dev
```

啟動後，在瀏覽器中訪問 **[http://localhost:3000](http://localhost:3000)** 即可開始使用！
