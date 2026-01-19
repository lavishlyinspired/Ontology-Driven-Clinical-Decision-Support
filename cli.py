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
    console.print("[bold]11-Agent Integrated Architecture (2025-2026):[/bold]")
    console.print("  [yellow]Core Processing (4):[/yellow]")
    console.print("    • [cyan]IngestionAgent[/cyan] → Validates raw patient data")
    console.print("    • [cyan]SemanticMappingAgent[/cyan] → Maps to SNOMED-CT/LUCADA codes")
    console.print("    • [cyan]ExplanationAgent[/cyan] → Generates MDT summaries")
    console.print("    • [cyan]PersistenceAgent[/cyan] → Saves to Neo4j (ONLY writer)\n")
    console.print("  [yellow]Specialized Clinical (5):[/yellow]")
    console.print("    • [cyan]NSCLCAgent[/cyan] → Non-small cell lung cancer treatment")
    console.print("    • [cyan]SCLCAgent[/cyan] → Small cell lung cancer protocols")
    console.print("    • [cyan]BiomarkerAgent[/cyan] → Precision medicine analysis")
    console.print("    • [cyan]ComorbidityAgent[/cyan] → Comorbidity impact assessment")
    console.print("    • [cyan]NegotiationAgent[/cyan] → Multi-agent consensus\n")
    console.print("  [yellow]Orchestration (2):[/yellow]")
    console.print("    • [cyan]DynamicOrchestrator[/cyan] → Intelligent agent routing")
    console.print("    • [cyan]IntegratedWorkflow[/cyan] → End-to-end coordination\n")
    
    console.print("[bold]Technology Stack:[/bold]")
    console.print("  • [yellow]OWL Ontology[/yellow]: LUCADA + SNOMED-CT")
    console.print("  • [yellow]LLM[/yellow]: Ollama (DeepSeek v3.1, Llama, Mistral)")
    console.print("  • [yellow]Orchestration[/yellow]: LangGraph multi-agent workflow")
    console.print("  • [yellow]Vector Store[/yellow]: Neo4j vector indexes")
    console.print("  • [yellow]Graph DB[/yellow]: Neo4j knowledge graph")
    console.print("  • [yellow]MCP[/yellow]: Claude Desktop integration\n")
    
    console.print("[bold]Quick Start:[/bold]")
    console.print("  [dim]$[/dim] python cli.py setup          # Install & configure")
    console.print("  [dim]$[/dim] python cli.py check          # Verify services")
    console.print("  [dim]$[/dim] python cli.py run-workflow   # Run decision support")
    console.print("  [dim]$[/dim] python cli.py start-api      # Start REST API")
    console.print("  [dim]$[/dim] python cli.py start-mcp      # Start MCP server\n")


if __name__ == "__main__":
    cli()

