import React, { useState, useRef, type FormEvent } from 'react';
import { Upload, Activity, ShieldAlert, FileText, CheckCircle2, ChevronRight } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [symptoms, setSymptoms] = useState('');
  const [medications, setMedications] = useState('');
  
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState({ step: 0, total: 7, message: '' });
  const [reportData, setReportData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selected = e.target.files[0];
      setFile(selected);
      setPreview(URL.createObjectURL(selected));
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const selected = e.dataTransfer.files[0];
      setFile(selected);
      setPreview(URL.createObjectURL(selected));
    }
  };

  const loadSample = async () => {
    const res = await fetch('/samples/sample_cxr.png');
    const blob = await res.blob();
    const sample = new File([blob], 'sample_cxr.png', { type: 'image/png' });
    setFile(sample);
    setPreview(URL.createObjectURL(sample));
    setSymptoms('65 year old male with chronic cough and shortness of breath for two weeks, mild fever');
    setMedications('Aspirin, Warfarin');
    setError(null);
  };

  const startAnalysis = async (e: FormEvent) => {
    e.preventDefault();
    if (!file || !symptoms) {
      setError("Please provide both an X-ray image and symptoms.");
      return;
    }

    setIsAnalyzing(true);
    setError(null);
    setReportData(null);
    setProgress({ step: 1, total: 7, message: 'Connecting to AI Copilot...' });

    const formData = new FormData();
    formData.append('image', file);
    formData.append('symptoms', symptoms);
    formData.append('current_medications', medications);

    try {
      const apiBase = import.meta.env.VITE_API_URL || '';
      const response = await fetch(`${apiBase}/api/analyze`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok || !response.body) {
        throw new Error(`Failed to start analysis. Status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        
        // Keep the last partial line in the buffer
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          
          const dataStr = line.replace('data: ', '').trim();
          if (!dataStr) continue;

          try {
            const json = JSON.parse(dataStr);

            if (json.type === 'status') {
              setProgress({
                step: json.step || progress.step,
                total: json.total_steps || 7,
                message: json.message || ''
              });
            } else if (json.type === 'result') {
              setReportData(json.payload);
            } else if (json.type === 'error') {
              setError(json.message);
            } else if (json.type === 'done') {
              setIsAnalyzing(false);
            }
          } catch (err) {
            console.error("Failed to parse SSE chunk:", dataStr);
          }
        }
      }
    } catch (err: any) {
      setError(err.message || 'An error occurred during analysis.');
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="container fade-in">
      {/* Navbar */}
      <header className="glass-panel" style={{ padding: '16px 24px', marginBottom: '32px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <Activity size={28} color="var(--primary)" className="animate-pulse-glow" style={{ borderRadius: '50%' }} />
          <h1 style={{ fontSize: '1.5rem', margin: 0 }}>Clinical AI Copilot</h1>
        </div>
        <div className="badge warning" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <ShieldAlert size={14} />
          Research & Demonstration Only
        </div>
      </header>

      <div className="grid-layout">
        {/* LEFT PANEL */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          <form onSubmit={startAnalysis} style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
            
            <div className="glass-panel" style={{ padding: '24px' }}>
              <h3 style={{ marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Upload size={20} color="var(--primary)" /> 
                1. Upload X-Ray
              </h3>
              
              <button
                type="button"
                id="load-sample-btn"
                onClick={loadSample}
                style={{ marginBottom: '12px', background: 'transparent', color: 'var(--primary)', border: '1px dashed var(--primary)', borderRadius: '8px', padding: '8px 14px', cursor: 'pointer', width: '100%' }}
              >
                Load sample case (X-ray + symptoms)
              </button>
              
              <div 
                className={`drop-zone ${preview ? 'active' : ''}`}
                onDragOver={(e) => e.preventDefault()}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <input 
                  type="file" 
                  accept="image/png, image/jpeg, image/dicom" 
                  ref={fileInputRef} 
                  style={{ display: 'none' }} 
                  onChange={handleFileChange}
                />
                
                {preview ? (
                  <img src={preview} alt="Preview" style={{ maxHeight: '200px', borderRadius: '8px', objectFit: 'contain' }} />
                ) : (
                  <>
                    <Upload />
                    <p style={{ color: 'var(--text-secondary)' }}>Drag & drop X-ray image here</p>
                    <p style={{ fontSize: '0.8rem', opacity: 0.5 }}>Supports PNG, JPG, DICOM</p>
                  </>
                )}
              </div>
            </div>

            <div className="glass-panel" style={{ padding: '24px' }}>
              <h3 style={{ marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <FileText size={20} color="var(--primary)" /> 
                2. Patient Context
              </h3>
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                <div>
                  <label style={{ display: 'block', marginBottom: '8px', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                    Clinical Symptoms *
                  </label>
                  <textarea 
                    className="input-field" 
                    rows={4} 
                    placeholder="e.g. 65yo male presenting with acute chest pain, shortness of breath..."
                    value={symptoms}
                    onChange={(e) => setSymptoms(e.target.value)}
                    required
                  />
                </div>
                <div>
                  <label style={{ display: 'block', marginBottom: '8px', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                    Current Medications (comma separated)
                  </label>
                  <input 
                    type="text" 
                    className="input-field" 
                    placeholder="e.g. Aspirin, Lisinopril"
                    value={medications}
                    onChange={(e) => setMedications(e.target.value)}
                  />
                </div>
              </div>
            </div>

            {error && (
              <div className="glass-panel" style={{ padding: '16px', borderLeft: '4px solid var(--danger)', backgroundColor: 'rgba(255,59,92,0.05)' }}>
                <p style={{ color: 'var(--danger)', fontSize: '0.9rem' }}>{error}</p>
              </div>
            )}

            <button 
              type="submit" 
              className="btn-primary" 
              disabled={isAnalyzing || !file || !symptoms}
              style={{ width: '100%', fontSize: '1.1rem' }}
            >
              {isAnalyzing ? (
                <>Analyzing... <Activity size={20} className="animate-pulse-glow" /></>
              ) : (
                <>Run AI Analysis <ChevronRight size={20} /></>
              )}
            </button>
          </form>
        </div>

        {/* RIGHT PANEL */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          
          {/* Analysis Report */}
          <div className="glass-panel" style={{ padding: '32px', flex: 1, minHeight: '600px', display: 'flex', flexDirection: 'column' }}>
            <h2 style={{ marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              Analysis Report
              {reportData && <span className="badge success"><CheckCircle2 size={14} style={{marginRight: '4px'}}/> Complete</span>}
            </h2>

            {isAnalyzing && !reportData && (
              <div className="fade-in" style={{ marginTop: 'auto', marginBottom: 'auto', maxWidth: '400px', alignSelf: 'center', width: '100%' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <span style={{ fontSize: '0.9rem', color: 'var(--primary)' }}>{progress.message}</span>
                  <span style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
                    Step {progress.step}/{progress.total}
                  </span>
                </div>
                <div className="progress-container">
                  <div 
                    className="progress-bar" 
                    style={{ width: `${(progress.step / progress.total) * 100}%` }}
                  ></div>
                </div>
              </div>
            )}

            {!isAnalyzing && !reportData && (
              <div style={{ marginTop: 'auto', marginBottom: 'auto', textAlign: 'center', opacity: 0.5 }}>
                <FileText size={48} style={{ marginBottom: '16px' }} />
                <p>Awaiting patient inputs...</p>
              </div>
            )}

            {reportData && (
              <div className="fade-in markdown-content" style={{ flex: 1, overflowY: 'auto' }}>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', alignItems: 'center', marginBottom: '20px' }}>
                  <span className="badge info">Confidence: {reportData.confidence_level}</span>
                  {(reportData.icd10_codes || []).map((c: string, i: number) => (
                    <span key={i} className="badge success" title={c}>ICD-10: {String(c).slice(0, 60)}</span>
                  ))}
                  <a
                    href={`${import.meta.env.VITE_API_URL || ''}${reportData.pdf_download_url}`}
                    className="btn-primary"
                    style={{ marginLeft: 'auto', padding: '8px 16px', fontSize: '0.9rem', textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: '6px' }}
                    download
                  >
                    <FileText size={16} /> Download PDF
                  </a>
                </div>
                {reportData.imaging_findings && reportData.imaging_findings !== 'No findings.' && (
                  <details style={{ marginBottom: '20px', background: 'rgba(0,0,0,0.25)', padding: '12px 16px', borderRadius: '8px', border: '1px solid var(--glass-border)' }}>
                    <summary style={{ cursor: 'pointer', color: 'var(--primary)' }}>Vision Agent – Raw Imaging Findings</summary>
                    <div style={{ marginTop: '12px', fontSize: '0.9rem' }}>
                      <ReactMarkdown>{String(reportData.imaging_findings)}</ReactMarkdown>
                    </div>
                  </details>
                )}
                <ReactMarkdown>{reportData.full_markdown_report || "No markdown report generated."}</ReactMarkdown>
                
                <hr style={{ border: 'none', borderTop: '1px solid var(--glass-border)', margin: '32px 0' }} />
                <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', fontStyle: 'italic' }}>
                  {reportData.disclaimer}
                </div>
              </div>
            )}
          </div>

          {/* Similar Cases Panel */}
          {reportData && reportData.similar_cases && reportData.similar_cases.length > 0 && (
            <div className="glass-panel fade-in" style={{ padding: '24px' }}>
              <h3 style={{ marginBottom: '16px', fontSize: '1.1rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Activity size={18} color="var(--primary)" /> Historical Similar Cases (Open-I / NIH RAG)
              </h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '16px' }}>
                {reportData.similar_cases.map((sc: any, idx: number) => (
                  <div key={idx} style={{ background: 'rgba(0,0,0,0.35)', padding: '16px', borderRadius: '8px', border: '1px solid var(--glass-border)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                      <span className="badge info">{sc.case_id || `Case ${idx + 1}`}</span>
                      <span style={{ fontSize: '0.8rem', color: 'var(--primary)', fontFamily: 'var(--font-mono)' }}>
                        {Math.round((sc.similarity_score || 0.85) * 100)}% Match
                      </span>
                    </div>
                    <p style={{ fontWeight: 600, fontSize: '0.88rem', marginBottom: '6px' }}>
                      {sc.diagnosis || 'Clinical Case'}
                    </p>
                    <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                      {(sc.key_findings || sc.description || '').slice(0, 120)}...
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

        </div>
      </div>
    </div>
  );
}
