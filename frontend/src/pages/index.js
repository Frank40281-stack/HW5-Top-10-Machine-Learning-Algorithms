import React, { useState, useEffect } from 'react';
import Head from 'next/head';
import Sidebar from '@/components/Sidebar';
import InteractiveChart from '@/components/InteractiveChart';
import katex from 'katex';

// Math Formula component using standard KaTeX rendering
function MathFormula({ formula, inline = false }) {
  const [html, setHtml] = useState('');

  useEffect(() => {
    try {
      const rendered = katex.renderToString(formula, {
        displayMode: !inline,
        throwOnError: false,
      });
      setHtml(rendered);
    } catch (e) {
      setHtml(formula);
    }
  }, [formula, inline]);

  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}

export default function Home() {
  const [currentSection, setCurrentSection] = useState('theory_basics');
  const [completedAlgos, setCompletedAlgos] = useState({});
  const [algos, setAlgos] = useState([]);
  
  // Model training state
  const [params, setParams] = useState({});
  const [trainData, setTrainData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [consoleLogs, setConsoleLogs] = useState('');

  // Quiz state
  const [quizQuestions, setQuizQuestions] = useState([]);
  const [currentQuestionIdx, setCurrentQuestionIdx] = useState(0);
  const [selectedOption, setSelectedOption] = useState(null);
  const [quizResult, setQuizResult] = useState(null);
  const [answeredCount, setAnsweredCount] = useState(0);
  const [correctCount, setCorrectCount] = useState(0);

  // Model Selection Wizard state
  const [wizardAnswers, setWizardAnswers] = useState({
    labeled: true,
    data_type: 'tabular',
    size: 'small',
    interpretability: true,
    non_linear: false
  });
  const [recommendationResult, setRecommendationResult] = useState(null);

  // Load algorithms list on mount
  useEffect(() => {
    fetch('http://127.0.0.1:8000/api/algorithms')
      .then((res) => res.json())
      .then((data) => {
        setAlgos(data);
      })
      .catch((err) => {
        console.error('Error fetching algorithms:', err);
      });

    // Load completed list from localStorage
    const saved = localStorage.getItem('completed_algos');
    if (saved) {
      try {
        setCompletedAlgos(JSON.parse(saved));
      } catch (e) {}
    }
  }, []);

  // Update default params when changing sections
  useEffect(() => {
    resetSectionData();
  }, [currentSection]);

  const resetSectionData = () => {
    setTrainData(null);
    setConsoleLogs('');
    setQuizQuestions([]);
    setCurrentQuestionIdx(0);
    setSelectedOption(null);
    setQuizResult(null);
    setAnsweredCount(0);
    setCorrectCount(0);
    setRecommendationResult(null);

    // Default params
    let defaultParams = {};
    if (currentSection === 'logistic_regression') {
      defaultParams = { C: 1.0, penalty: 'l2' };
    } else if (currentSection === 'knn') {
      defaultParams = { n_neighbors: 5, weights: 'uniform' };
    } else if (currentSection === 'svm') {
      defaultParams = { C: 1.0, kernel: 'rbf' };
    } else if (currentSection === 'decision_tree') {
      defaultParams = { max_depth: 4 };
    } else if (currentSection === 'random_forest') {
      defaultParams = { n_estimators: 100, max_depth: 4 };
    } else if (currentSection === 'xgboost') {
      defaultParams = { n_estimators: 50, max_depth: 4, learning_rate: 0.1 };
    } else if (currentSection === 'kmeans') {
      defaultParams = { n_clusters: 3, init: 'k-means++' };
    }
    setParams(defaultParams);

    // Fetch quiz for algorithms
    if (isAlgoSection(currentSection)) {
      fetch(`http://127.0.0.1:8000/api/quiz/${currentSection}`)
        .then((res) => res.json())
        .then((data) => {
          setQuizQuestions(data);
        })
        .catch((err) => console.error(err));
    }
  };

  const isAlgoSection = (sectionId) => {
    return ![ 'theory_basics', 'model_selection_wizard' ].includes(sectionId);
  };

  const toggleCompletion = (sectionId) => {
    const updated = { ...completedAlgos, [sectionId]: !completedAlgos[sectionId] };
    setCompletedAlgos(updated);
    localStorage.setItem('completed_algos', JSON.stringify(updated));
  };

  const handleParamChange = (name, value) => {
    setParams({ ...params, [name]: value });
  };

  // Run training API call
  const triggerTrainModel = () => {
    setLoading(true);
    setConsoleLogs('>> 初始化模擬數據中...\n>> 載入 Scikit-Learn 模型與核心參數...\n');
    
    fetch('http://127.0.0.1:8000/api/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        algo_id: currentSection,
        params: params
      })
    })
      .then((res) => {
        if (!res.ok) throw new Error('Training request failed');
        return res.json();
      })
      .then((data) => {
        setTrainData(data);
        setConsoleLogs((prev) => 
          prev + `>> 模型擬合成功！訓練指標：\n${JSON.stringify(data.metrics, null, 2)}\n\n>> 模型內部結構參數：\n${data.details}\n\n>> 成功繪製數據分佈與決策範圍。`
        );
      })
      .catch((err) => {
        setConsoleLogs((prev) => prev + `>> [錯誤]：${err.message}`);
      })
      .finally(() => {
        setLoading(false);
      });
  };

  // Submit quiz answer
  const submitQuiz = () => {
    if (selectedOption === null) return;
    
    fetch('http://127.0.0.1:8000/api/quiz/submit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        algo_id: currentSection,
        question_idx: currentQuestionIdx,
        selected_option: selectedOption
      })
    })
      .then((res) => res.json())
      .then((data) => {
        setQuizResult(data);
        setAnsweredCount(answeredCount + 1);
        if (data.correct) {
          setCorrectCount(correctCount + 1);
        }
      })
      .catch((err) => console.error(err));
  };

  const nextQuizQuestion = () => {
    setSelectedOption(null);
    setQuizResult(null);
    setCurrentQuestionIdx(currentQuestionIdx + 1);
  };

  // Recommendation wizard triggers
  const runSelectionWizard = () => {
    setLoading(true);
    fetch('http://127.0.0.1:8000/api/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(wizardAnswers)
    })
      .then((res) => res.json())
      .then((data) => {
        setRecommendationResult(data);
      })
      .catch((err) => console.error(err))
      .finally(() => setLoading(false));
  };

  const activeAlgo = algos.find((a) => a.id === currentSection);

  return (
    <>
      <Head>
        <title>機器學習十大核心演算法終極精研教材</title>
        <meta name="description" content="包含核心理論、數學架構、Python實作與企業應用場景的動態機器學習學習平台" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className="app-container">
        <Sidebar 
          currentSection={currentSection} 
          onSelectSection={setCurrentSection} 
          completedAlgos={completedAlgos}
        />

        <main className="workspace">
          {/* Section 1: Theory Basics */}
          {currentSection === 'theory_basics' && (
            <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
              <div className="page-header">
                <h1 className="page-title">Module 1: 機器學習範式與核心理論基礎</h1>
                <p className="page-subtitle">在深入探討十大演算法之前，必須先建構嚴謹的機器學習（Machine Learning, ML）範式認知。</p>
              </div>

              <div className="intro-grid">
                <div className="card">
                  <div className="card-title">
                    <span className="badge-label badge-blue">范式 1.1</span>
                    監督式學習與非監督式學習
                  </div>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem', lineHeight: '1.6' }}>
                    <strong>監督式學習 (Supervised Learning)</strong>：訓練數據集包含輸入特徵 <MathFormula formula="X" inline /> 與對應標籤 <MathFormula formula="Y" inline />。學習目標是尋找映射函數 <MathFormula formula="f: X \rightarrow Y" inline />。包括迴歸與分類任務。<br /><br />
                    <strong>非監督式學習 (Unsupervised Learning)</strong>：訓練數據僅包含輸入特徵 <MathFormula formula="X" inline />。其核心任務在於探索數據內部的統計結構與分佈模式，常見任務包括分群與降維。
                  </p>
                </div>

                <div className="card">
                  <div className="card-title">
                    <span className="badge-label badge-purple">概念 1.2</span>
                    經驗風險最小化與泛化誤差
                  </div>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem', lineHeight: '1.6' }}>
                    機器學習模型的訓練本質上是最小化損失函數。我們定義在訓練集上的「經驗風險（Empirical Risk）」為：
                  </p>
                  <div className="formula-display" style={{ margin: '1rem 0' }}>
                    <MathFormula formula="R_{emp}(f) = \frac{1}{n} \sum_{i=1}^{n} L(y_i, f(x_i))" />
                  </div>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem', lineHeight: '1.6' }}>
                    過度追求經驗風險最小化會導致模型強行記憶訓練集中的雜訊，進而引發過擬合（Overfitting）。現代演算法皆引入結構風險最小化（SRM），透過加入正規化項來平衡模型複雜度與泛化能力。
                  </p>
                </div>

                <div className="card" style={{ gridColumn: '1 / -1' }}>
                  <div className="card-title">
                    <span className="badge-label badge-green">權衡 1.3</span>
                    偏誤-變異數權衡 (Bias-Variance Tradeoff)
                  </div>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem', lineHeight: '1.6', marginBottom: '1rem' }}>
                    泛化誤差可以完美拆解為三部分：偏誤的平方（Bias²）、變異數（Variance）以及不可消除的隨機雜訊（Irreducible Error）。
                  </p>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
                    <div>
                      <h4 style={{ color: 'var(--accent-red)', marginBottom: '0.25rem' }}>偏誤 (Bias)</h4>
                      <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>描述模型預測值的期望與真實值之間的差距，反映了模型的擬合能力。高偏誤代表模型太簡單，造成欠擬合（Underfitting）。</p>
                    </div>
                    <div>
                      <h4 style={{ color: 'var(--accent-yellow)', marginBottom: '0.25rem' }}>變異數 (Variance)</h4>
                      <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>描述模型在不同訓練集上擬合結果的波動程度，反映了模型的穩定性。高變異數代表模型過於複雜，造成過擬合（Overfitting）。</p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="card" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <h3 style={{ fontSize: '1.1rem', fontWeight: 600 }}>準備好進入十大演算法的世界了嗎？</h3>
                  <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>點擊下方按鈕或使用側邊欄開始學習線性迴歸。</p>
                </div>
                <button className="btn" onClick={() => setCurrentSection('linear_regression')}>開始學習 01 | 線性迴歸</button>
              </div>
            </div>
          )}

          {/* Section 2: Machine Learning Algorithms (01 - 10) */}
          {isAlgoSection(currentSection) && activeAlgo && (
            <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
              <div className="page-header">
                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '0.5rem' }}>
                  <span className="badge-label badge-purple">{activeAlgo.module}</span>
                  <h1 className="page-title" style={{ margin: 0 }}>{activeAlgo.name}</h1>
                </div>
                <p className="page-subtitle">{activeAlgo.explanation}</p>
              </div>

              {/* Math Foundations Card */}
              <div className="card formula-container">
                <h3 className="card-title">
                  <span className="badge-label badge-blue">數學基石</span>
                  公式與核心架構
                </h3>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
                  <div className="form-group">
                    <span className="form-label">核心數學表達式</span>
                    <div className="formula-display">
                      <MathFormula formula={activeAlgo.formula} />
                    </div>
                  </div>
                  <div className="form-group">
                    <span className="form-label">優化目標與損失函數</span>
                    <div className="formula-display">
                      <MathFormula formula={activeAlgo.optimization} />
                    </div>
                  </div>
                </div>
              </div>

              {/* Playground & Live Training Card */}
              <div className="card">
                <h3 className="card-title">
                  <span className="badge-label badge-green">動態沙盒</span>
                  互動式參數模擬與決策視覺化
                </h3>
                
                <div className="playground-grid">
                  <div className="chart-panel">
                    <InteractiveChart 
                      algoId={currentSection} 
                      points={trainData?.points} 
                      mesh={trainData?.mesh}
                      line={trainData?.line}
                      centroids={trainData?.centroids}
                      axes={trainData?.axes}
                    />
                  </div>

                  <div className="params-panel">
                    <h4 style={{ fontSize: '1rem', fontWeight: 600 }}>超參數設定 (Hyperparameters)</h4>
                    
                    {/* Hyperparameter Inputs depending on algorithm */}
                    {currentSection === 'logistic_regression' && (
                      <>
                        <div className="form-group">
                          <label className="form-label">正則化強度 C (L2 Regularization)</label>
                          <input 
                            type="range" min="0.01" max="10" step="0.1" 
                            value={params.C || 1.0} 
                            onChange={(e) => handleParamChange('C', parseFloat(e.target.value))}
                            className="form-control"
                          />
                          <span style={{ fontSize: '0.8rem', color: 'var(--accent-blue)' }}>目前數值: {params.C || 1.0} (數值愈小正則化愈強)</span>
                        </div>
                        <div className="form-group">
                          <label className="form-label">正則化類別 (Penalty)</label>
                          <select 
                            value={params.penalty || 'l2'} 
                            onChange={(e) => handleParamChange('penalty', e.target.value)}
                            className="form-control"
                          >
                            <option value="l2">L2 正則化 (Ridge)</option>
                          </select>
                        </div>
                      </>
                    )}

                    {currentSection === 'knn' && (
                      <>
                        <div className="form-group">
                          <label className="form-label">近鄰數 K (n_neighbors)</label>
                          <input 
                            type="number" min="1" max="25" 
                            value={params.n_neighbors || 5} 
                            onChange={(e) => handleParamChange('n_neighbors', parseInt(e.target.value))}
                            className="form-control"
                          />
                        </div>
                        <div className="form-group">
                          <label className="form-label">權重計算方式 (Weights)</label>
                          <select 
                            value={params.weights || 'uniform'} 
                            onChange={(e) => handleParamChange('weights', e.target.value)}
                            className="form-control"
                          >
                            <option value="uniform">Uniform (均等權重)</option>
                            <option value="distance">Distance (距離反比權重)</option>
                          </select>
                        </div>
                      </>
                    )}

                    {currentSection === 'svm' && (
                      <>
                        <div className="form-group">
                          <label className="form-label">錯誤懲罰係數 C</label>
                          <input 
                            type="range" min="0.1" max="20" step="0.5" 
                            value={params.C || 1.0} 
                            onChange={(e) => handleParamChange('C', parseFloat(e.target.value))}
                            className="form-control"
                          />
                          <span style={{ fontSize: '0.8rem', color: 'var(--accent-blue)' }}>目前數值: {params.C || 1.0}</span>
                        </div>
                        <div className="form-group">
                          <label className="form-label">核函數 (Kernel Trick)</label>
                          <select 
                            value={params.kernel || 'rbf'} 
                            onChange={(e) => handleParamChange('kernel', e.target.value)}
                            className="form-control"
                          >
                            <option value="rbf">RBF (徑向基底高斯核)</option>
                            <option value="linear">Linear (線性核)</option>
                            <option value="poly">Poly (多項式核)</option>
                          </select>
                        </div>
                      </>
                    )}

                    {currentSection === 'decision_tree' && (
                      <div className="form-group">
                        <label className="form-label">樹的最大深度 (max_depth)</label>
                        <input 
                          type="number" min="1" max="10" 
                          value={params.max_depth || 4} 
                          onChange={(e) => handleParamChange('max_depth', parseInt(e.target.value))}
                          className="form-control"
                        />
                      </div>
                    )}

                    {currentSection === 'random_forest' && (
                      <>
                        <div className="form-group">
                          <label className="form-label">子樹數量 (n_estimators)</label>
                          <input 
                            type="number" min="10" max="200" step="10" 
                            value={params.n_estimators || 100} 
                            onChange={(e) => handleParamChange('n_estimators', parseInt(e.target.value))}
                            className="form-control"
                          />
                        </div>
                        <div className="form-group">
                          <label className="form-label">樹最大深度 (max_depth)</label>
                          <input 
                            type="number" min="1" max="8" 
                            value={params.max_depth || 4} 
                            onChange={(e) => handleParamChange('max_depth', parseInt(e.target.value))}
                            className="form-control"
                          />
                        </div>
                      </>
                    )}

                    {currentSection === 'xgboost' && (
                      <>
                        <div className="form-group">
                          <label className="form-label">樹的數量 (n_estimators)</label>
                          <input 
                            type="number" min="10" max="150" step="10" 
                            value={params.n_estimators || 50} 
                            onChange={(e) => handleParamChange('n_estimators', parseInt(e.target.value))}
                            className="form-control"
                          />
                        </div>
                        <div className="form-group">
                          <label className="form-label">學習率 (learning_rate / eta)</label>
                          <input 
                            type="range" min="0.01" max="0.5" step="0.01" 
                            value={params.learning_rate || 0.1} 
                            onChange={(e) => handleParamChange('learning_rate', parseFloat(e.target.value))}
                            className="form-control"
                          />
                          <span style={{ fontSize: '0.8rem', color: 'var(--accent-blue)' }}>目前數值: {params.learning_rate || 0.1}</span>
                        </div>
                        <div className="form-group">
                          <label className="form-label">單棵樹最大深度 (max_depth)</label>
                          <input 
                            type="number" min="1" max="6" 
                            value={params.max_depth || 4} 
                            onChange={(e) => handleParamChange('max_depth', parseInt(e.target.value))}
                            className="form-control"
                          />
                        </div>
                      </>
                    )}

                    {currentSection === 'kmeans' && (
                      <>
                        <div className="form-group">
                          <label className="form-label">分群數 K (n_clusters)</label>
                          <input 
                            type="number" min="2" max="6" 
                            value={params.n_clusters || 3} 
                            onChange={(e) => handleParamChange('n_clusters', parseInt(e.target.value))}
                            className="form-control"
                          />
                        </div>
                        <div className="form-group">
                          <label className="form-label">初始化演算法 (Init)</label>
                          <select 
                            value={params.init || 'k-means++'} 
                            onChange={(e) => handleParamChange('init', e.target.value)}
                            className="form-control"
                          >
                            <option value="k-means++">k-means++ (推薦優化點位)</option>
                            <option value="random">random (隨機點位)</option>
                          </select>
                        </div>
                      </>
                    )}

                    {/* Fallback description for non-param algos */}
                    {['linear_regression', 'naive_bayes', 'pca'].includes(currentSection) && (
                      <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>此演算法採用閉式解或無超參數調節。請直接點擊擬合按鈕查看動態效果。</p>
                    )}

                    <button className="btn" style={{ marginTop: '1rem' }} onClick={triggerTrainModel} disabled={loading}>
                      {loading ? '擬合中...' : '擬合與更新模型'}
                    </button>
                  </div>
                </div>

                {/* Simulated runtime terminal console */}
                {consoleLogs && (
                  <div style={{ marginTop: '1.5rem' }}>
                    <div className="code-title-bar">
                      <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)' }}>Python 執行終端輸出</span>
                    </div>
                    <div className="console-window">{consoleLogs}</div>
                  </div>
                )}
              </div>

              {/* Code Snippet Card */}
              <div className="card">
                <div className="code-title-bar">
                  <h3 className="card-title" style={{ margin: 0 }}>
                    <span className="badge-label badge-purple">程式實作</span>
                    Python Code (Scikit-Learn)
                  </h3>
                </div>
                <pre className="code-display">
                  <code>{activeAlgo.python_code}</code>
                </pre>
              </div>

              {/* Enterprise Scenarios Pros/Cons */}
              <div className="card">
                <h3 className="card-title">
                  <span className="badge-label badge-blue">實踐應用</span>
                  企業場景與優缺點評估
                </h3>
                
                <table className="feature-table">
                  <thead>
                    <tr>
                      <th style={{ width: '30%' }}>企業場景</th>
                      <th style={{ width: '35%' }}>優點 (Advantages)</th>
                      <th style={{ width: '35%' }}>缺點 (Limitations)</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>{activeAlgo.use_cases}</td>
                      <td>
                        <ul>
                          {activeAlgo.pros.map((p, i) => <li key={i}>{p}</li>)}
                        </ul>
                      </td>
                      <td>
                        <ul>
                          {activeAlgo.cons.map((c, i) => <li key={i}>{c}</li>)}
                        </ul>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>

              {/* Quiz Card */}
              {quizQuestions.length > 0 && currentQuestionIdx < quizQuestions.length && (
                <div className="card">
                  <h3 className="card-title">
                    <span className="badge-label badge-green">知識自測</span>
                    核心概念小測驗
                  </h3>

                  <div className="quiz-question">
                    <strong>問題：</strong> {quizQuestions[currentQuestionIdx].question}
                  </div>

                  <div className="options-list">
                    {quizQuestions[currentQuestionIdx].options.map((opt, idx) => {
                      let btnClass = '';
                      if (selectedOption === idx) {
                        btnClass = 'selected';
                      }
                      if (quizResult !== null) {
                        if (idx === quizResult.correct_option) {
                          btnClass = 'correct';
                        } else if (selectedOption === idx) {
                          btnClass = 'incorrect';
                        }
                      }
                      return (
                        <button 
                          key={idx} 
                          className={`option-btn ${btnClass}`} 
                          onClick={() => quizResult === null && setSelectedOption(idx)}
                          disabled={quizResult !== null}
                        >
                          {opt}
                        </button>
                      );
                    })}
                  </div>

                  {quizResult === null ? (
                    <button className="btn" onClick={submitQuiz} disabled={selectedOption === null}>送出答案</button>
                  ) : (
                    <div>
                      <div className={`quiz-feedback ${quizResult.correct ? 'correct' : 'incorrect'}`}>
                        <strong>{quizResult.correct ? '✓ 答對了！' : '✗ 答錯了！'}</strong>
                        <p style={{ marginTop: '0.5rem', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                          <MathFormula formula={quizResult.explanation} />
                        </p>
                      </div>
                      
                      {currentQuestionIdx < quizQuestions.length - 1 ? (
                        <button className="btn btn-secondary" style={{ marginTop: '1rem' }} onClick={nextQuizQuestion}>下一題</button>
                      ) : (
                        <p style={{ marginTop: '1rem', color: 'var(--accent-green)', fontWeight: 600 }}>
                          測驗完成！您答對了 {correctCount} / {answeredCount} 題。
                        </p>
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* Completion Action */}
              <div className="card" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderColor: completedAlgos[currentSection] ? 'var(--accent-green)' : 'var(--border-light)' }}>
                <div>
                  <h3 style={{ fontSize: '1.05rem', fontWeight: 600 }}>完成了本演算法的學習嗎？</h3>
                  <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>標記完成可以追蹤您的十大演算法學習進度。</p>
                </div>
                <button 
                  className={`btn ${completedAlgos[currentSection] ? 'btn-secondary' : ''}`}
                  onClick={() => toggleCompletion(currentSection)}
                >
                  {completedAlgos[currentSection] ? '已標記完成 (點擊取消)' : '✓ 標記本單元為完成'}
                </button>
              </div>
            </div>
          )}

          {/* Section 3: Model Selection Wizard (Module 5) */}
          {currentSection === 'model_selection_wizard' && (
            <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
              <div className="page-header">
                <h1 className="page-title">Module 5: 機器學習工程實踐與模型選型心法</h1>
                <p className="page-subtitle">在真實企業級 AI 項目中，沒有最好的演算法，只有最適合業務場景的演算法。以下是資深資料科學家總結的「選型 decision logic」互動版。</p>
              </div>

              {/* Decision logic cheat sheet */}
              <div className="card">
                <h3 className="card-title">
                  <span className="badge-label badge-purple">黃金心法</span>
                  業界選型決策心法 (Cheat Sheet)
                </h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1.25rem' }}>
                  <div style={{ padding: '1rem', background: 'rgba(255,255,255,0.02)', borderRadius: '0.5rem', borderLeft: '3px solid var(--accent-blue)' }}>
                    <h4 style={{ color: 'var(--text-primary)', marginBottom: '0.5rem' }}>1. 表格數據（有標籤）</h4>
                    <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>優先使用 <strong>XGBoost / 隨機森林</strong>，準確度通常封頂。</p>
                  </div>
                  <div style={{ padding: '1rem', background: 'rgba(255,255,255,0.02)', borderRadius: '0.5rem', borderLeft: '3px solid var(--accent-purple)' }}>
                    <h4 style={{ color: 'var(--text-primary)', marginBottom: '0.5rem' }}>2. 數據量極小或需解釋</h4>
                    <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>優先考慮 <strong>線性迴歸 / 邏輯迴歸</strong>，直接分析權重影響力。</p>
                  </div>
                  <div style={{ padding: '1rem', background: 'rgba(255,255,255,0.02)', borderRadius: '0.5rem', borderLeft: '3px solid var(--accent-green)' }}>
                    <h4 style={{ color: 'var(--text-primary)', marginBottom: '0.5rem' }}>3. 文本分類 / 垃圾郵件</h4>
                    <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>首選 <strong>樸素貝氏</strong> 作為快速建立的基準線模型。</p>
                  </div>
                  <div style={{ padding: '1rem', background: 'rgba(255,255,255,0.02)', borderRadius: '0.5rem', borderLeft: '3px solid var(--accent-yellow)' }}>
                    <h4 style={{ color: 'var(--text-primary)', marginBottom: '0.5rem' }}>4. 完全無標籤數據</h4>
                    <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>先用 <strong>PCA</strong> 降噪與視覺化，再用 <strong>K-Means</strong> 切分業務輪廓。</p>
                  </div>
                </div>
              </div>

              {/* Wizard questionnaire */}
              <div className="card">
                <h3 className="card-title">
                  <span className="badge-label badge-green">決策精靈</span>
                  互動式模型選型精靈 (Model Selection Wizard)
                </h3>

                <div className="wizard-form">
                  <div className="form-group">
                    <label className="form-label">1. 數據是否包含標籤 (Is data labeled)?</label>
                    <div className="wizard-options">
                      <div 
                        className={`wizard-card ${wizardAnswers.labeled ? 'active' : ''}`}
                        onClick={() => setWizardAnswers({ ...wizardAnswers, labeled: true })}
                      >
                        <h4>有標籤 (Labeled)</h4>
                        <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>用於預測類別或數值 (分類/迴歸)</p>
                      </div>
                      <div 
                        className={`wizard-card ${!wizardAnswers.labeled ? 'active' : ''}`}
                        onClick={() => setWizardAnswers({ ...wizardAnswers, labeled: false })}
                      >
                        <h4>無標籤 (Unlabeled)</h4>
                        <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>用於尋找內部分群或維度壓縮 (非監督)</p>
                      </div>
                    </div>
                  </div>

                  {wizardAnswers.labeled && (
                    <>
                      <div className="form-group">
                        <label className="form-label">2. 數據主要類型為何 (Data type)?</label>
                        <div className="wizard-options">
                          <div 
                            className={`wizard-card ${wizardAnswers.data_type === 'tabular' ? 'active' : ''}`}
                            onClick={() => setWizardAnswers({ ...wizardAnswers, data_type: 'tabular' })}
                          >
                            <h4>表格數據 (Tabular)</h4>
                            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>二維數值或類別欄位數據</p>
                          </div>
                          <div 
                            className={`wizard-card ${wizardAnswers.data_type === 'text' ? 'active' : ''}`}
                            onClick={() => setWizardAnswers({ ...wizardAnswers, data_type: 'text' })}
                          >
                            <h4>文本數據 (Text / NLP)</h4>
                            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>垃圾郵件過濾、輿情文字分析</p>
                          </div>
                        </div>
                      </div>

                      <div className="form-group">
                        <label className="form-label">3. 數據間關係是否高度非線性 (Non-linear relationship)?</label>
                        <div className="wizard-options">
                          <div 
                            className={`wizard-card ${!wizardAnswers.non_linear ? 'active' : ''}`}
                            onClick={() => setWizardAnswers({ ...wizardAnswers, non_linear: false })}
                          >
                            <h4>線性 / 偏簡單關係</h4>
                            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>變數間接近正反比或單純線性相加</p>
                          </div>
                          <div 
                            className={`wizard-card ${wizardAnswers.non_linear ? 'active' : ''}`}
                            onClick={() => setWizardAnswers({ ...wizardAnswers, non_linear: true })}
                          >
                            <h4>高度非線性 / 複雜邊界</h4>
                            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>具有多層複雜交互作用</p>
                          </div>
                        </div>
                      </div>

                      <div className="form-group">
                        <label className="form-label">4. 是否需要極佳的可解釋性 (Need explainability)?</label>
                        <div className="wizard-options">
                          <div 
                            className={`wizard-card ${wizardAnswers.interpretability ? 'active' : ''}`}
                            onClick={() => setWizardAnswers({ ...wizardAnswers, interpretability: true })}
                          >
                            <h4>需要白盒解釋</h4>
                            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>必須明確印出決策樹規則或權重影響力</p>
                          </div>
                          <div 
                            className={`wizard-card ${!wizardAnswers.interpretability ? 'active' : ''}`}
                            onClick={() => setWizardAnswers({ ...wizardAnswers, interpretability: false })}
                          >
                            <h4>準確度優先</h4>
                            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>接受黑盒模型，只要模型預測最準即可</p>
                          </div>
                        </div>
                      </div>
                    </>
                  )}

                  <div className="form-group">
                    <label className="form-label">{wizardAnswers.labeled ? '5. 數據規模大小 (Data size)' : '2. 數據規模大小 (Data size)'}?</label>
                    <div className="wizard-options">
                      <div 
                        className={`wizard-card ${wizardAnswers.size === 'small' ? 'active' : ''}`}
                        onClick={() => setWizardAnswers({ ...wizardAnswers, size: 'small' })}
                      >
                        <h4>小規模數據</h4>
                        <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>數千筆以內，注重運算資源與防過擬合</p>
                      </div>
                      <div 
                        className={`wizard-card ${wizardAnswers.size === 'large' ? 'active' : ''}`}
                        onClick={() => setWizardAnswers({ ...wizardAnswers, size: 'large' })}
                      >
                        <h4>大規模數據</h4>
                        <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>數萬到百萬筆，注重平行算力與泛化</p>
                      </div>
                    </div>
                  </div>

                  <button className="btn" style={{ marginTop: '1rem', width: '200px' }} onClick={runSelectionWizard} disabled={loading}>
                    {loading ? '分析中...' : '進行智慧模型推薦'}
                  </button>
                </div>

                {recommendationResult && (
                  <div style={{ marginTop: '2rem', padding: '1.5rem', background: 'rgba(168, 85, 247, 0.08)', border: '1px solid rgba(168, 85, 247, 0.25)', borderRadius: '0.75rem' }}>
                    <h4 style={{ color: 'var(--accent-purple)', fontSize: '1.1rem', marginBottom: '0.5rem' }}>智能推薦模型結果</h4>
                    <p style={{ fontSize: '1.2rem', fontWeight: 700, color: '#ffffff', marginBottom: '0.75rem' }}>
                      推薦演算法：{recommendationResult.name}
                    </p>
                    <p style={{ fontSize: '0.95rem', color: 'var(--text-secondary)', lineHeight: '1.6', whiteSpace: 'pre-line' }}>
                      <strong>決策依據：</strong><br />
                      {recommendationResult.reason}
                    </p>
                    <button 
                      className="btn" 
                      style={{ marginTop: '1.25rem' }} 
                      onClick={() => setCurrentSection(recommendationResult.algo_id)}
                    >
                      進入 {recommendationResult.name} 的沙盒學習
                    </button>
                  </div>
                )}
              </div>
            </div>
          )}
        </main>
      </div>
    </>
  );
}
