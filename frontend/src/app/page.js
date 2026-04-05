"use client";

import { useState } from "react";

export default function Home() {
  const [query, setQuery] = useState("");
  const [submittedQuery, setSubmittedQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [resultHTML, setResultHTML] = useState("");
  const [hasSearched, setHasSearched] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setHasSearched(true);
    setResultHTML("");
    setSubmittedQuery(query);

    try {
      // Connect to the FastAPI backend
      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL;
      const res = await fetch(`${backendUrl}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: query }),
      });

      if (!res.ok) {
        throw new Error("Failed to fetch response from server");
      }

      const data = await res.json();
      setResultHTML(data.response || "<p>No response generated.</p>");
    } catch (err) {
      console.error(err);
      setResultHTML("<p style='color: #f85149;'>Error: Could not connect to the Mutual Fund AI Analyzer backend. Is the server running?</p>");
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadPDF = async () => {
    try {
      const element = document.getElementById("pdf-content");
      
      // Keep PDF background cohesive with dark mode aesthetics
      const originalBackground = element.style.background;
      element.style.background = "#0d1117"; 
      element.style.padding = "20px";
      
      // Dynamically import html2pdf
      const html2pdf = (await import("html2pdf.js")).default;
      
      const opt = {
        margin: 0.5,
        filename: 'Mutual_Fund_Analysis.pdf',
        image: { type: 'jpeg', quality: 0.98 },
        html2canvas: { scale: 2, useCORS: true, logging: false },
        jsPDF: { unit: 'in', format: 'letter', orientation: 'portrait' }
      };

      await html2pdf().set(opt).from(element).save();
      
      // Revert styles
      element.style.background = originalBackground;
      element.style.padding = "10px";
    } catch (err) {
      console.error("Failed to generate PDF:", err);
    }
  };

  return (
    <main>
      <div className="header">
        <h1>Mutual Fund Intelligence</h1>
        <p>Your premium AI advisor for navigating the Indian market.</p>
      </div>

      <div className="search-container">
        <form className="search-form" onSubmit={handleSubmit}>
          <input
            type="text"
            className="search-input"
            placeholder="Ask about funds, SIP plans, market sentiment, or share your portfolio..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={loading}
          />
          <button type="submit" className="search-button" disabled={loading || !query.trim()}>
            {loading ? <div className="spinner"></div> : "Analyze"}
          </button>
        </form>
      </div>

      {hasSearched ? (
        <div className="result-card">
          <div className="result-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ display: "flex", alignItems: "center", gap: "0.8rem" }}>
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
                <polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline>
                <line x1="12" y1="22.08" x2="12" y2="12"></line>
              </svg>
              Analysis Result
            </span>
            {!loading && (
              <button className="download-btn" onClick={handleDownloadPDF} title="Download Report as PDF">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                  <polyline points="7 10 12 15 17 10"></polyline>
                  <line x1="12" y1="15" x2="12" y2="3"></line>
                </svg>
                Download PDF
              </button>
            )}
          </div>
          {loading ? (
            <div style={{ padding: "2rem 0", color: "var(--text-secondary)", fontStyle: "italic" }}>
              Synthesizing market data and generating insights...
            </div>
          ) : (
             <div id="pdf-content" style={{ padding: "10px", borderRadius: "12px" }}>
               {/* Display the query for reference inside the PDF */}
               <div style={{ marginBottom: "2rem", padding: "1.2rem", background: "rgba(47, 129, 247, 0.08)", borderRadius: "12px", borderLeft: "4px solid var(--accent-blue)" }}>
                 <div style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginBottom: "0.4rem", textTransform: "uppercase", letterSpacing: "0.5px", fontWeight: "600" }}>Your Query</div>
                 <div style={{ fontSize: "1.1rem", color: "var(--text-primary)", fontWeight: "500" }}>{submittedQuery}</div>
               </div>
               
               {/* Display the AI response */}
               <div 
                 className="result-content" 
                 dangerouslySetInnerHTML={{ __html: resultHTML }} 
               />
             </div>
          )}
        </div>
      ) : (
        <div className="empty-state">
          <div className="empty-state-icon">❖</div>
          <h3>Ready to discover insights?</h3>
          <p style={{ marginTop: "0.5rem" }}>
            Try asking to compare funds, check current market sentiment, or review a SIP plan.
          </p>
        </div>
      )}
    </main>
  );
}
