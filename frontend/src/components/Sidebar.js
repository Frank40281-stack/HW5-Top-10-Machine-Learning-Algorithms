import React from 'react';

export default function Sidebar({ currentSection, onSelectSection, completedAlgos }) {
  const modules = [
    {
      id: 'module1',
      title: 'Module 1: 核心理論基礎',
      items: [
        { id: 'theory_basics', name: '機器學習範式與基礎' }
      ]
    },
    {
      id: 'module2',
      title: 'Module 2: 預測與幾何分類',
      items: [
        { id: 'linear_regression', name: '01 | 線性迴歸' },
        { id: 'logistic_regression', name: '02 | 邏輯迴歸' },
        { id: 'knn', name: '03 | K-近鄰演算法 (KNN)' },
        { id: 'svm', name: '04 | 支援向量機 (SVM)' },
        { id: 'naive_bayes', name: '05 | 樸素貝氏' }
      ]
    },
    {
      id: 'module3',
      title: 'Module 3: 樹狀與集成架構',
      items: [
        { id: 'decision_tree', name: '06 | 決策樹' },
        { id: 'random_forest', name: '07 | 隨機森林' },
        { id: 'xgboost', name: '08 | 梯度提升與 XGBoost' }
      ]
    },
    {
      id: 'module4',
      title: 'Module 4: 非監督探索與降維',
      items: [
        { id: 'kmeans', name: '09 | K-Means 分群' },
        { id: 'pca', name: '10 | 主成分分析 (PCA)' }
      ]
    },
    {
      id: 'module5',
      title: 'Module 5: 工程實踐與選型',
      items: [
        { id: 'model_selection_wizard', name: '黃金選型決策精研' }
      ]
    }
  ];

  return (
    <aside className="sidebar">
      <div className="logo-section">
        <div className="logo-icon">ML</div>
        <div className="logo-title">
          <div>十大核心演算法</div>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>終極精研教材</div>
        </div>
      </div>
      <nav className="module-list">
        {modules.map((mod) => (
          <div key={mod.id} className="module-group">
            <div className="module-header">{mod.title}</div>
            {mod.items.map((item) => {
              const isActive = currentSection === item.id;
              const isCompleted = completedAlgos[item.id];
              return (
                <div
                  key={item.id}
                  className={`nav-item ${isActive ? 'active' : ''}`}
                  onClick={() => onSelectSection(item.id)}
                >
                  <span style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {item.name}
                  </span>
                  {isCompleted && <span className="check-badge">✓</span>}
                </div>
              );
            })}
          </div>
        ))}
      </nav>
    </aside>
  );
}
