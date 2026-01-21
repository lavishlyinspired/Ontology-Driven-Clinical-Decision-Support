"""
Lung Cancer Assistant (LCA) - Command Line Interface
Complete CLI for running the LCA system end-to-end.

Available Commands:
- setup: Initialize project (install deps, create dirs, check services)
- check: Verify all prerequisites and services
- start-api: Launch FastAPI REST server
- start-mcp: Launch MCP server for Claude Desktop integration
- run-workflow: Execute 11-agent integrated clinical decision support workflow
- generate-samples: Create sample patient data files
- test: Run comprehensive system tests
"""
import json
import sys
import os
import subprocess
import uvicorn
from pathlib import Path
from typing import Optional
import time

try:
    import typer
    from rich.console import Console
    from rich.table import Table
    from rich import print as rprint
except ImportError:
    print("ERROR: Required packages not installed.")
    print("Run: pip install typer rich")
    sys.exit(1)

# Fix Windows console encoding for Unicode characters
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parent

# Prioritize backend and workspace packages for in-project imports
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT))

console = Console()
cli = typer.Typer(help="Lung Cancer Assistant - Clinical Decision Support System")


# ============================================================================
# SETUP & CHECK COMMANDS
# ============================================================================

@cli.command()
def setup(
    skip_install: bool = typer.Option(False, help="Skip pip install step"),
):
    """Complete project setup: dependencies, directories, environment."""
    console.print("\n[bold cyan]═══ LUNG CANCER ASSISTANT - SETUP ═══[/bold cyan]\n")
    
    # Check Python version
    console.print("[yellow]→[/yellow] Checking Python version...")
    if sys.version_info < (3, 9):
        console.print("[red]✗ Python 3.9+ required[/red]")
        raise typer.Exit(1)
    console.print(f"[green]✓[/green] Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Install dependencies
    if not skip_install:
        console.print("\n[yellow]→[/yellow] Installing dependencies...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            cwd=PROJECT_ROOT,
            capture_output=True
        )
        if result.returncode == 0:
            console.print("[green]✓[/green] Dependencies installed")
        else:
            console.print("[red]✗[/red] Installation failed")
            console.print(result.stderr.decode())
    
    # Create .env file
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        console.print("\n[yellow]→[/yellow] Creating .env file...")
        env_example = PROJECT_ROOT / ".env.example"
        if env_example.exists():
            env_file.write_text(env_example.read_text())
            console.print("[green]✓[/green] .env created from template")
        else:
            console.print("[red]✗[/red] .env.example not found")
    else:
        console.print("[green]✓[/green] .env file exists")
    
    # Create directories
    console.print("\n[yellow]→[/yellow] Creating directories...")
    for dir_name in ["logs", "output", "data"]:
        (PROJECT_ROOT / dir_name).mkdir(exist_ok=True)
        console.print(f"[green]✓[/green] {dir_name}/")
    
    console.print("\n[bold green]Setup complete![/bold green]")
    console.print("\n[yellow]Next steps:[/yellow]")
    console.print("  1. Run: [cyan]python cli.py check[/cyan] to verify services")
    console.print("  2. Start Ollama: [cyan]ollama serve[/cyan]")
    console.print("  3. Pull model: [cyan]ollama pull deepseek-v3.1:671b-cloud[/cyan]")
    console.print("  4. Start Neo4j: [cyan]docker-compose up -d neo4j[/cyan]")
    console.print("  5. Run workflow: [cyan]python cli.py run-workflow[/cyan]\n")


@cli.command()
def check():
    """Verify all prerequisites and service availability."""
    console.print("\n[bold cyan]═══ SYSTEM CHECK ═══[/bold cyan]\n")
    
    table = Table(title="Service Status")
    table.add_column("Service", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="dim")
    
    # Check Ollama
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, timeout=5)
        if result.returncode == 0:
            models = result.stdout.decode().strip().split('\n')[1:]  # Skip header
            table.add_row("Ollama", "✓ Running", f"{len(models)} models")
        else:
            table.add_row("Ollama", "✗ Not running", "Start with: ollama serve")
    except Exception as e:
        table.add_row("Ollama", "✗ Not found", "Install from https://ollama.ai")
    
    # Check Neo4j
    try:
        from neo4j import GraphDatabase
        from dotenv import load_dotenv
        load_dotenv()
        
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        driver.close()
        table.add_row("Neo4j", "✓ Connected", uri)
    except Exception as e:
        table.add_row("Neo4j", "✗ Not available", "Start with: docker-compose up -d neo4j")
    
    # Check Python packages
    packages = {
        "langchain": "LangChain",
        "langgraph": "LangGraph",
        "owlready2": "OWL Ontology",
        "sentence_transformers": "Embeddings",
        "fastapi": "FastAPI",
        "mcp": "MCP Server"
    }
    
    for pkg, name in packages.items():
        try:
            __import__(pkg)
            table.add_row(name, "✓ Installed", "")
        except ImportError:
            table.add_row(name, "✗ Missing", f"pip install {pkg}")
    
    console.print(table)
    console.print()


# ============================================================================
# SERVICE COMMANDS
# ============================================================================

# ============================================================================
# SERVICE COMMANDS
# ============================================================================

@cli.command()
def start_api(
    host: str = typer.Option("0.0.0.0", help="API host"),
    port: int = typer.Option(8000, help="API port"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
):
    """Start the FastAPI REST server."""
    console.print(f"\n[bold cyan]Starting LCA API Server on {host}:{port}[/bold cyan]\n")
    console.print("  Endpoint: http://localhost:8000")
    console.print("  Docs: http://localhost:8000/docs")
    console.print("  Stop with: Ctrl+C\n")
    
    uvicorn.run(
        "backend.src.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


@cli.command()
def start_mcp(
    transport: str = typer.Option("stdio", help="Transport: stdio or sse"),
):
    """Start the MCP server for Claude Desktop integration."""
    console.print("\n[bold cyan]Starting LCA MCP Server[/bold cyan]\n")
    console.print("  Transport: " + transport)
    console.print("  Configure Claude Desktop with:")
    
    mcp_server_path = PROJECT_ROOT / "backend" / "src" / "mcp_server" / "lca_mcp_server.py"
    config_example = '{{"command": "python", "args": ["{}"]}}'.format(str(mcp_server_path).replace('\\', '\\\\'))
    console.print(f"    {config_example}\n")
    
    # Run MCP server
    if not mcp_server_path.exists():
        console.print(f"[red]✗ MCP server not found at {mcp_server_path}[/red]")
        raise typer.Exit(1)
    
    subprocess.run([sys.executable, str(mcp_server_path)])


# ============================================================================
# WORKFLOW COMMANDS
# ============================================================================

@cli.command()
def run_workflow(
    patient_file: Optional[Path] = typer.Option(
        None, exists=True, help="JSON file with patient data"
    ),
    patient_id: Optional[str] = typer.Option(None, help="Use built-in patient by ID"),
    persist: bool = typer.Option(False, help="Save results to Neo4j"),
    verbose: bool = typer.Option(False, help="Show detailed output"),
):
    """Run the 11-agent integrated clinical decision support workflow."""
    console.print("\n[bold cyan]═══ LCA 11-AGENT INTEGRATED WORKFLOW ═══[/bold cyan]\n")
    
    # Load patient data
    if patient_file:
        console.print(f"[yellow]→[/yellow] Loading patient from: {patient_file}")
        with open(patient_file) as f:
            patient_data = json.load(f)
    elif patient_id:
        from data.sample_patients import get_patient_by_id
        patient_data = get_patient_by_id(patient_id)
        if not patient_data:
            console.print(f"[red]✗ Patient {patient_id} not found[/red]")
            raise typer.Exit(1)
    else:
        from data.sample_patients import get_jenny_sesen
        patient_data = get_jenny_sesen()
        console.print("[yellow]→[/yellow] Using default patient: Jenny Sesen")
    
    console.print(f"\n[cyan]Patient:[/cyan] {patient_data.get('patient_id')}")
    console.print(f"  Name: {patient_data.get('name')}")
    console.print(f"  Stage: {patient_data.get('tnm_stage')}")
    console.print(f"  Histology: {patient_data.get('histology_type')}\n")
    
    # Run workflow
    try:
        from src.agents.lca_workflow import LCAWorkflow
        
        console.print("[yellow]→[/yellow] Initializing workflow...")
        workflow = LCAWorkflow(persist_results=persist)
        
        console.print("[yellow]→[/yellow] Running 11-agent integrated pipeline...\n")
        start_time = time.time()
        
        result = workflow.run(patient_data)
        
        elapsed = time.time() - start_time
        
        # Display results
        console.print(f"\n[bold green]✓ Workflow Complete[/bold green] ({elapsed:.2f}s)\n")
        
        table = Table(title="Classification Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Status", result.workflow_status)
        table.add_row("Scenario", result.scenario)
        table.add_row("Confidence", f"{result.scenario_confidence:.1%}")
        table.add_row("Agents", str(len(result.agent_chain)))
        
        console.print(table)
        
        # Recommendations
        console.print("\n[bold]Top Recommendations:[/bold]")
        for i, rec in enumerate(result.recommendations[:5], 1):
            console.print(f"  {i}. [cyan]{rec['treatment']}[/cyan] ({rec['evidence_level']})")
            console.print(f"     Source: {rec.get('rule_source', 'N/A')}")
        
        # MDT Summary
        if result.mdt_summary:
            console.print("\n[bold]MDT Summary:[/bold]")
            summary_preview = result.mdt_summary[:500]
            console.print(f"  {summary_preview}..." if len(result.mdt_summary) > 500 else f"  {result.mdt_summary}")
        
        if verbose:
            console.print("\n[bold]Full Output:[/bold]")
            console.print(json.dumps(result.to_dict(), indent=2))
        
        console.print()
        
    except ImportError as e:
        console.print(f"[red]✗ Import error: {e}[/red]")
        console.print("  Run: [cyan]python cli.py setup[/cyan]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]✗ Workflow failed: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)


@cli.command()
def run_advanced_workflow(
    patient_file: Optional[Path] = typer.Option(
        None, exists=True, help="JSON file with patient data"
    ),
    patient_id: Optional[str] = typer.Option(None, help="Use built-in patient by ID"),
    persist: bool = typer.Option(False, help="Save results to Neo4j"),
    verbose: bool = typer.Option(False, help="Show detailed output"),
):
    """Run the ADVANCED integrated workflow with all enhancements."""
    console.print("\n[bold magenta]═══ LCA ADVANCED INTEGRATED WORKFLOW ═══[/bold magenta]\n")
    console.print("[yellow]Features:[/yellow] Complexity routing, specialized agents, analytics, provenance\n")
    
    # Load patient data
    if patient_file:
        console.print(f"[yellow]→[/yellow] Loading patient from: {patient_file}")
        with open(patient_file) as f:
            patient_data = json.load(f)
    elif patient_id:
        from data.sample_patients import get_patient_by_id
        patient_data = get_patient_by_id(patient_id)
        if not patient_data:
            console.print(f"[red]✗ Patient {patient_id} not found[/red]")
            raise typer.Exit(1)
    else:
        from data.sample_patients import get_jenny_sesen
        patient_data = get_jenny_sesen()
        console.print("[yellow]→[/yellow] Using default patient: Jenny Sesen")
    
    console.print(f"\n[cyan]Patient:[/cyan] {patient_data.get('patient_id')}")
    console.print(f"[cyan]Stage:[/cyan] {patient_data.get('tnm_stage')}")
    console.print(f"[cyan]Histology:[/cyan] {patient_data.get('histology_type')}\n")
    
    try:
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "backend"))
        
        from src.services.lca_service import LungCancerAssistantService
        import asyncio
        
        console.print("[yellow]→[/yellow] Initializing service with advanced workflow...")
        service = LungCancerAssistantService(
            use_neo4j=persist,
            use_vector_store=True,
            enable_advanced_workflow=True,
            enable_provenance=True
        )
        
        console.print("[yellow]→[/yellow] Running complexity assessment...")
        
        # Run advanced workflow
        start = time.time()
        
        async def run():
            result = await service.process_patient(
                patient_data=patient_data,
                use_ai_workflow=True,
                force_advanced=True
            )
            return result
        
        result = asyncio.run(run())
        elapsed = time.time() - start
        
        console.print(f"\n[bold green]✓ Advanced Workflow Complete[/bold green] ({elapsed:.2f}s)\n")
        
        # Display results
        table = Table(title="Advanced Workflow Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Patient ID", result.patient_id)
        table.add_row("Workflow Type", result.workflow_type)
        table.add_row("Complexity Level", result.complexity_level or "N/A")
        table.add_row("Execution Time", f"{result.execution_time_ms}ms")
        table.add_row("Recommendations", str(len(result.recommendations)))
        table.add_row("Provenance Record", result.provenance_record_id or "N/A")
        
        console.print(table)
        
        # Display recommendations
        if result.recommendations:
            console.print("\n[bold]Top Recommendations:[/bold]")
            for i, rec in enumerate(result.recommendations[:3], 1):
                console.print(f"\n{i}. [green]{rec.treatment_type}[/green]")
                console.print(f"   Evidence: {rec.evidence_level}")
                console.print(f"   Source: {rec.rule_source}")
                console.print(f"   Confidence: {rec.confidence_score:.2%}")
        
        # Display MDT summary if verbose
        if verbose and result.mdt_summary:
            console.print("\n[bold]MDT Summary:[/bold]")
            console.print(result.mdt_summary)
        
        # Display provenance info
        if result.provenance_record_id:
            console.print(f"\n[dim]Provenance record saved: {result.provenance_record_id}[/dim]")
            console.print(f"[dim]Use 'python cli.py show-provenance {result.provenance_record_id}' to view details[/dim]")
        
        console.print()
        
    except ImportError as e:
        console.print(f"[red]✗ Import error: {e}[/red]")
        console.print("  Run: [cyan]python cli.py setup[/cyan]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]✗ Advanced workflow failed: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)


@cli.command()
def assess_complexity(
    patient_file: Optional[Path] = typer.Option(
        None, exists=True, help="JSON file with patient data"
    ),
    patient_id: Optional[str] = typer.Option(None, help="Use built-in patient by ID"),
):
    """Assess patient complexity and get workflow recommendation."""
    console.print("\n[bold cyan]═══ COMPLEXITY ASSESSMENT ═══[/bold cyan]\n")
    
    # Load patient data
    if patient_file:
        with open(patient_file) as f:
            patient_data = json.load(f)
    elif patient_id:
        from data.sample_patients import get_patient_by_id
        patient_data = get_patient_by_id(patient_id)
        if not patient_data:
            console.print(f"[red]✗ Patient {patient_id} not found[/red]")
            raise typer.Exit(1)
    else:
        from data.sample_patients import get_jenny_sesen
        patient_data = get_jenny_sesen()
    
    try:
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "backend"))
        
        from src.services.lca_service import LungCancerAssistantService
        import asyncio
        
        service = LungCancerAssistantService(
            enable_advanced_workflow=True
        )
        
        async def run():
            return await service.assess_complexity(patient_data)
        
        result = asyncio.run(run())
        
        # Display results
        table = Table(title=f"Complexity Assessment: {patient_data.get('patient_id')}")
        table.add_column("Factor", style="cyan")
        table.add_column("Value", style="yellow")
        
        table.add_row("Complexity Level", f"[bold]{result['complexity']}[/bold]")
        table.add_row("Recommended Workflow", result['recommended_workflow'])
        table.add_row("Reason", result['reason'])
        
        console.print(table)
        
        # Display factors
        console.print("\n[bold]Contributing Factors:[/bold]")
        factors = result.get('factors', {})
        for key, value in factors.items():
            console.print(f"  • {key}: {value}")
        
        console.print()
        
    except Exception as e:
        console.print(f"[red]✗ Assessment failed: {e}[/red]")
        raise typer.Exit(1)


@cli.command()
def show_provenance(
    record_id: str = typer.Argument(..., help="Provenance record ID")
):
    """Display provenance record details for audit/compliance."""
    console.print(f"\n[bold cyan]═══ PROVENANCE RECORD: {record_id} ═══[/bold cyan]\n")
    
    try:
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "backend"))
        
        from src.services.lca_service import LungCancerAssistantService
        
        service = LungCancerAssistantService(enable_provenance=True)
        record = service.get_provenance_record(record_id)
        
        if not record:
            console.print(f"[red]✗ Record not found: {record_id}[/red]")
            raise typer.Exit(1)
        
        # Display record summary
        table = Table(title="Provenance Summary")
        table.add_column("Attribute", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Record ID", record.get('record_id'))
        table.add_row("Patient ID", record.get('patient_id'))
        table.add_row("Workflow Type", record.get('workflow_type'))
        table.add_row("Complexity", record.get('complexity_routing') or "N/A")
        table.add_row("Created At", record.get('created_at'))
        
        console.print(table)
        
        # Display execution chain
        console.print("\n[bold]Agent Execution Chain:[/bold]")
        for i, agent in enumerate(record.get('execution_chain', []), 1):
            console.print(f"  {i}. {agent}")
        
        # Display data sources
        if record.get('data_sources'):
            console.print("\n[bold]Data Sources:[/bold]")
            for source in record.get('data_sources', []):
                console.print(f"  • {source.get('source')} ({source.get('timestamp')})")
        
        # Display ontology versions
        if record.get('ontology_versions'):
            console.print("\n[bold]Ontology Versions:[/bold]")
            for onto, version in record.get('ontology_versions', {}).items():
                console.print(f"  • {onto}: {version}")
        
        console.print()
        
        # Offer to export
        export = typer.confirm("Export full record to JSON?", default=False)
        if export:
            output_file = PROJECT_ROOT / f"provenance_{record_id}.json"
            with open(output_file, 'w') as f:
                json.dump(record, f, indent=2)
            console.print(f"[green]✓ Exported to: {output_file}[/green]")
        
    except Exception as e:
        console.print(f"[red]✗ Failed to retrieve provenance: {e}[/red]")
        raise typer.Exit(1)


@cli.command()
def test(
    component: Optional[str] = typer.Option(None, help="Test specific component: ontology, workflow, api"),
):
    """Run system tests."""
    console.print("\n[bold cyan]═══ RUNNING TESTS ═══[/bold cyan]\n")
    
    if component == "ontology":
        console.print("[yellow]→[/yellow] Testing ontology...")
        subprocess.run([sys.executable, "backend/src/ontology/lucada_ontology.py"], cwd=PROJECT_ROOT)
    elif component == "workflow":
        console.print("[yellow]→[/yellow] Testing workflow...")
        subprocess.run([sys.executable, "test_6agent_workflow.py"], cwd=PROJECT_ROOT)
    elif component == "api":
        console.print("[yellow]→[/yellow] Testing API...")
        subprocess.run([sys.executable, "-m", "pytest", "backend/tests/"], cwd=PROJECT_ROOT)
    else:
        console.print("[yellow]→[/yellow] Running all tests...")
        subprocess.run([sys.executable, "test_installation.py"], cwd=PROJECT_ROOT)


# ============================================================================
# DATA COMMANDS
# ============================================================================

# ============================================================================
# DATA COMMANDS
# ============================================================================

@cli.command()
def generate_samples(
    output: Path = typer.Option("data/sample_patients.json", help="Output JSON file"),
    count: int = typer.Option(10, help="Number of sample patients (includes built-ins)"),
):
    """Generate sample patient data files."""
    console.print("\n[bold cyan]Generating Sample Patients[/bold cyan]\n")
    
    from data.sample_patients import save_sample_patients, generate_random_patient, SAMPLE_PATIENTS
    
    output.parent.mkdir(parents=True, exist_ok=True)
    save_sample_patients(str(output))
    
    console.print(f"[green]✓[/green] {len(SAMPLE_PATIENTS)} sample patients saved to {output}")
    console.print("\n[cyan]Available patients:[/cyan]")
    for p in SAMPLE_PATIENTS[:5]:
        console.print(f"  • {p['patient_id']}: {p['tnm_stage']} {p['histology_type']}")
    console.print()


@cli.command()
def list_patients():
    """List all available sample patients."""
    from data.sample_patients import SAMPLE_PATIENTS
    
    table = Table(title="Sample Patients")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="white")
    table.add_column("Stage", style="yellow")
    table.add_column("Histology", style="green")
    table.add_column("PS", style="magenta")
    
    for p in SAMPLE_PATIENTS:
        table.add_row(
            p['patient_id'],
            p['name'],
            p['tnm_stage'],
            p['histology_type'],
            str(p['performance_status'])
        )
    
    console.print()
    console.print(table)
    console.print()


# ============================================================================
# HELPER & INFO COMMANDS
# ============================================================================

@cli.command()
def info():
    """Display project information and architecture overview."""
    console.print("\n[bold cyan]═══ LUNG CANCER ASSISTANT (LCA) ═══[/bold cyan]\n")
    console.print("[bold]6-Agent Architecture:[/bold]")
    console.print("  1. [cyan]IngestionAgent[/cyan]        → Validates raw patient data")
    console.print("  2. [cyan]SemanticMappingAgent[/cyan]  → Maps to SNOMED-CT codes")
    console.print("  3. [cyan]ClassificationAgent[/cyan]   → Applies LUCADA ontology + NICE guidelines")
    console.print("  4. [cyan]ConflictResolutionAgent[/cyan] → Resolves conflicting recommendations")
    console.print("  5. [cyan]PersistenceAgent[/cyan]      → Saves to Neo4j (ONLY writer)")
    console.print("  6. [cyan]ExplanationAgent[/cyan]      → Generates MDT summaries\n")
    
    console.print("[bold]Technology Stack:[/bold]")
    console.print("  • [yellow]OWL Ontology[/yellow]: LUCADA + SNOMED-CT")
    console.print("  • [yellow]LLM[/yellow]: Ollama (DeepSeek v3.1, Llama, Mistral)")
    console.print("  • [yellow]Orchestration[/yellow]: LangGraph multi-agent workflow")
    console.print("  • [yellow]Vector Store[/yellow]: Neo4j vector indexes")
    console.print("  • [yellow]Graph DB[/yellow]: Neo4j knowledge graph")
    console.print("  • [yellow]MCP[/yellow]: Claude Desktop integration\n")
    
    console.print("[bold]Available Workflows:[/bold]")
    console.print("  1. [green]Standard Workflow[/green]   → Fast, basic cases (20s)")
    console.print("  2. [magenta]Advanced Workflow[/magenta]   → Complex cases with full analytics (45s)\n")
    
    console.print("[bold]Quick Start:[/bold]")
    console.print("  [dim]$[/dim] python cli.py setup                # Install & configure")
    console.print("  [dim]$[/dim] python cli.py check                # Verify services")
    console.print("  [dim]$[/dim] python cli.py assess-complexity    # Check patient complexity")
    console.print("  [dim]$[/dim] python cli.py run-workflow         # Run standard workflow")
    console.print("  [dim]$[/dim] python cli.py run-advanced-workflow # Run advanced workflow")
    console.print("  [dim]$[/dim] python cli.py show-provenance <id>  # View audit trail")
    console.print("  [dim]$[/dim] python cli.py start-api            # Start REST API")
    console.print("  [dim]$[/dim] python cli.py start-mcp            # Start MCP server\n")


if __name__ == "__main__":
    cli()

