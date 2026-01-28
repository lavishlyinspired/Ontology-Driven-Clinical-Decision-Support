"use client";

import { useState } from "react";

const mcpToolCategories = [
  {
    name: "Patient Management",
    icon: "P",
    tools: [
      { name: "create_patient", description: "Create a new patient in LUCADA ontology" },
      { name: "get_patient", description: "Retrieve patient data by ID" },
      { name: "update_patient", description: "Update patient clinical data" },
      { name: "delete_patient", description: "Soft delete with audit trail" },
      { name: "validate_patient_schema", description: "Validate against LUCADA schema" },
      { name: "find_similar_patients", description: "Vector similarity search" },
      { name: "get_patient_history", description: "Complete inference history" },
      { name: "get_cohort_stats", description: "Cohort statistics with filters" }
    ]
  },
  {
    name: "11-Agent Workflow",
    icon: "A",
    tools: [
      { name: "run_6agent_workflow", description: "Complete 11-agent integrated workflow" },
      { name: "get_workflow_info", description: "Workflow architecture information" },
      { name: "run_ingestion_agent", description: "Validate and normalize patient data" },
      { name: "run_semantic_mapping_agent", description: "Map to SNOMED-CT codes" },
      { name: "run_classification_agent", description: "Apply LUCADA and NICE guidelines" },
      { name: "run_conflict_resolution_agent", description: "Resolve conflicting recommendations" },
      { name: "run_explanation_agent", description: "Generate MDT summary" },
      { name: "generate_mdt_summary", description: "Complete MDT summary" }
    ]
  },
  {
    name: "Specialized Agents",
    icon: "S",
    tools: [
      { name: "run_nsclc_agent", description: "NSCLC-specific treatment recommendations" },
      { name: "run_sclc_agent", description: "SCLC-specific treatment recommendations" },
      { name: "run_biomarker_agent", description: "Precision medicine based on molecular profiling" },
      { name: "run_comorbidity_agent", description: "Treatment safety assessment" },
      { name: "recommend_biomarker_testing", description: "Recommend biomarker tests" },
      { name: "run_integrated_workflow", description: "Complete 2024-2026 enhanced workflow" }
    ]
  },
  {
    name: "Adaptive Workflow 2026",
    icon: "W",
    tools: [
      { name: "assess_case_complexity", description: "Complexity routing (simple/moderate/complex/critical)" },
      { name: "run_adaptive_workflow", description: "Dynamic multi-agent with self-correction" },
      { name: "query_context_graph", description: "Query reasoning chains and relationships" },
      { name: "execute_parallel_agents", description: "Parallel agent execution" },
      { name: "analyze_with_self_correction", description: "Self-corrective analysis loops" },
      { name: "get_workflow_metrics", description: "Performance metrics" }
    ]
  },
  {
    name: "Analytics Suite",
    icon: "Y",
    tools: [
      { name: "survival_kaplan_meier", description: "Kaplan-Meier survival analysis" },
      { name: "survival_cox_regression", description: "Cox proportional hazards" },
      { name: "survival_compare_treatments", description: "Compare treatment outcomes" },
      { name: "quantify_uncertainty", description: "Bayesian confidence estimation" },
      { name: "match_clinical_trials", description: "ClinicalTrials.gov matching" },
      { name: "analyze_counterfactuals", description: "What-if scenario analysis" },
      { name: "stratify_risk", description: "Risk stratification" }
    ]
  },
  {
    name: "Knowledge Base",
    icon: "K",
    tools: [
      { name: "get_nice_guidelines", description: "NICE CG121 lung cancer guidelines" },
      { name: "search_guidelines_semantic", description: "Semantic guideline search" },
      { name: "get_snomed_mapping", description: "SNOMED-CT code lookup" },
      { name: "get_loinc_code", description: "LOINC laboratory code lookup" },
      { name: "get_rxnorm_drug", description: "RxNorm medication lookup" },
      { name: "query_ontology", description: "LUCADA ontology queries" }
    ]
  },
  {
    name: "Digital Twin",
    icon: "T",
    tools: [
      { name: "create_digital_twin", description: "Initialize patient digital twin" },
      { name: "update_twin", description: "Update twin with new data" },
      { name: "get_twin_predictions", description: "Trajectory predictions" },
      { name: "get_twin_alerts", description: "Active alerts" },
      { name: "get_twin_snapshot", description: "Point-in-time snapshot" }
    ]
  },
  {
    name: "Graph & Export",
    icon: "G",
    tools: [
      { name: "query_neo4j", description: "Execute Cypher queries" },
      { name: "find_similar_by_graph", description: "Graph-based similarity" },
      { name: "analyze_temporal_patterns", description: "Temporal progression analysis" },
      { name: "export_fhir", description: "Export to FHIR format" },
      { name: "export_pdf_report", description: "Generate PDF report" },
      { name: "get_provenance", description: "W3C PROV-DM audit trail" }
    ]
  }
];

export default function MCPToolsPage() {
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState("");

  const filteredCategories = mcpToolCategories.map(cat => ({
    ...cat,
    tools: cat.tools.filter(tool =>
      tool.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      tool.description.toLowerCase().includes(searchTerm.toLowerCase())
    )
  })).filter(cat => cat.tools.length > 0);

  const totalTools = mcpToolCategories.reduce((acc, cat) => acc + cat.tools.length, 0);

  return (
    <div className="page-container">
      <header className="page-header">
        <h1>MCP Tools Reference</h1>
        <p className="muted">60+ tools exposed via Model Context Protocol for Claude AI integration</p>
      </header>

      <section className="section">
        <div className="card stats-card">
          <div className="stats-row">
            <div className="stat-item">
              <span className="stat-value">{totalTools}</span>
              <span className="stat-label">Total Tools</span>
            </div>
            <div className="stat-item">
              <span className="stat-value">{mcpToolCategories.length}</span>
              <span className="stat-label">Categories</span>
            </div>
            <div className="stat-item">
              <span className="stat-value">3000</span>
              <span className="stat-label">Default Port</span>
            </div>
          </div>
        </div>
      </section>

      <section className="section">
        <div className="search-container">
          <input
            type="text"
            placeholder="Search tools..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="input search-input"
          />
        </div>
      </section>

      <section className="section">
        <h2>Tool Categories</h2>
        <div className="categories-grid">
          {filteredCategories.map((category) => (
            <div
              key={category.name}
              className={`card category-card ${selectedCategory === category.name ? "selected" : ""}`}
              onClick={() => setSelectedCategory(selectedCategory === category.name ? null : category.name)}
            >
              <div className="category-header">
                <span className="category-icon">{category.icon}</span>
                <h3>{category.name}</h3>
                <span className="tool-count">{category.tools.length} tools</span>
              </div>
              {selectedCategory === category.name && (
                <div className="tools-list">
                  {category.tools.map((tool) => (
                    <div key={tool.name} className="tool-item">
                      <code className="tool-name">{tool.name}</code>
                      <p className="tool-description">{tool.description}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </section>

      <section className="section">
        <h2>Configuration</h2>
        <div className="card">
          <h3>Claude Desktop Config</h3>
          <p className="muted">Add to your claude_desktop_config.json:</p>
          <pre className="code-block">
{`{
  "mcpServers": {
    "lung-cancer-assistant": {
      "command": "python",
      "args": ["backend/src/mcp_server/lca_mcp_server.py"],
      "cwd": "H:\\\\akash\\\\git\\\\CoherencePLM\\\\Version22",
      "env": {
        "PYTHONPATH": "H:\\\\akash\\\\git\\\\CoherencePLM\\\\Version22\\\\backend",
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "123456789"
      }
    }
  }
}`}
          </pre>
        </div>
      </section>

      <section className="section">
        <h2>Start MCP Server</h2>
        <div className="card">
          <pre className="code-block">
{`# Start MCP Server
python start_mcp_server.py

# Or run directly
python backend/src/mcp_server/lca_mcp_server.py`}
          </pre>
        </div>
      </section>
    </div>
  );
}
