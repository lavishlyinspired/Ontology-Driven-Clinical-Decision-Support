# Ontology and Knowledge Base Download Guide

## Complete Setup Guide for LCA Dependencies

This guide provides step-by-step instructions for downloading and configuring all required ontologies, terminologies, and knowledge bases for the Lung Cancer Assistant system.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [SNOMED CT Download](#snomed-ct-download)
3. [LOINC Ontology Download](#loinc-ontology-download)
4. [RxNorm Download](#rxnorm-download)
5. [Neo4j Setup](#neo4j-setup)
6. [Directory Structure](#directory-structure)
7. [Verification](#verification)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites

- **Operating System**: Linux, macOS, or Windows with WSL2
- **Disk Space**: Minimum 10GB free space
- **RAM**: 8GB minimum, 16GB recommended
- **Internet**: Stable connection for large downloads
- **Accounts Required**:
  - UMLS account (free) for SNOMED CT and RxNorm
  - Regenstrief Institute account (free) for LOINC

### Installation Time

- **Total**: ~45-60 minutes
- **SNOMED CT**: 20 minutes (5GB download + extraction)
- **LOINC**: 10 minutes (2GB download)
- **RxNorm**: 10 minutes (1.5GB download)
- **Neo4j**: 5 minutes (setup + configuration)

---

## SNOMED CT Download

### Step 1: Create UMLS Account

1. Visit [https://www.nlm.nih.gov/research/umls/](https://www.nlm.nih.gov/research/umls/)
2. Click "Request a License"
3. Create account (requires email verification)
4. Accept UMLS Metathesaurus License Agreement
5. **Processing time**: Usually same-day approval

### Step 2: Access SNOMED CT

1. Log in to UMLS Terminology Services: [https://uts.nlm.nih.gov/uts/](https://uts.nlm.nih.gov/uts/)
2. Navigate to "Download" → "UMLS Full Release"
3. Select latest version (e.g., 2025AB)
4. Download method: Choose "Metathesaurus"

### Step 3: Download SNOMED CT OWL Distribution

```bash
# Create directory structure
mkdir -p ~/lca_ontologies/snomed_ct
cd ~/lca_ontologies/snomed_ct

# Download SNOMED CT International Edition
# Option 1: Via UMLS (requires license)
# Download from: https://uts.nlm.nih.gov/uts/
# File: SnomedCT_InternationalRF2_PRODUCTION_20250131T120000Z.zip

# Option 2: Via SNOMED International (if licensed)
# Visit: https://www.snomed.org/
# Navigate to: SNOMED CT > Download

# Extract the archive
unzip SnomedCT_InternationalRF2_PRODUCTION_*.zip

# Navigate to OWL distribution
cd Snapshot/Terminology/
ls -lh sct2_*.owl
```

### Step 4: Convert to Owlready2 Format (if needed)

```bash
# Install owlready2 if not already installed
pip install owlready2

# Convert large OWL file (optional optimization)
python3 << 'EOF'
from owlready2 import *

# Load SNOMED CT OWL
onto_path.append("./")
snomed = get_ontology("sct2_Concept_Snapshot_INT_20250131.owl").load()

# Save in optimized format
snomed.save(file="snomed_ct_optimized.owl", format="rdfxml")
print(f"✓ Optimized SNOMED CT saved")
print(f"Classes: {len(list(snomed.classes()))}")
print(f"Properties: {len(list(snomed.properties()))}")
EOF
```

**Expected Output**:
```
✓ Optimized SNOMED CT saved
Classes: 378416
Properties: 126
```

### Step 5: Configure LCA to Use SNOMED CT

```python
# In your config file or environment
export SNOMED_CT_PATH="/home/user/lca_ontologies/snomed_ct/snomed_ct_optimized.owl"
```

---

## LOINC Ontology Download

### Step 1: Create Regenstrief Account

1. Visit [https://loinc.org/](https://loinc.org/)
2. Click "Download LOINC"
3. Create free account
4. Accept LOINC License Agreement

### Step 2: Download LOINC Ontology 2.0

```bash
# Create directory
mkdir -p ~/lca_ontologies/loinc
cd ~/lca_ontologies/loinc

# Download from Regenstrief
# Visit: https://loinc.org/downloads/
# Select: "LOINC Table File - CSV/Text Format"
# Version: 2.78 (October 2025 or latest)

# Expected file: Loinc_2.78_Text.zip (approx 150MB compressed)
unzip Loinc_2.78_Text.zip

# Extract LOINC codes
cd Loinc_2.78

# Key files:
# - LoincTable/Loinc.csv (main table)
# - AnswerList/ (lab test answer lists)
# - AccessoryFiles/ (additional mappings)

# Verify download
wc -l LoincTable/Loinc.csv
# Expected: ~99,000 lines (98,000+ LOINC codes)
```

### Step 3: Load into LCA System

```python
# The LOINC Integrator will automatically load from this path
export LOINC_TABLE_PATH="/home/user/lca_ontologies/loinc/Loinc_2.78/LoincTable/Loinc.csv"

# Test loading
python3 << 'EOF'
from backend.src.ontology.loinc_integrator import LOINCIntegrator

integrator = LOINCIntegrator()
integrator.load_loinc_table("/home/user/lca_ontologies/loinc/Loinc_2.78/LoincTable/Loinc.csv")

print(f"✓ LOINC concepts loaded: {len(integrator.loinc_codes)}")

# Test mapping
result = integrator.interpret_lab_result("Hemoglobin", 11.5, "g/dL", sex="Female")
print(f"✓ Test mapping successful: {result}")
EOF
```

**Expected Output**:
```
✓ LOINC concepts loaded: 98247
✓ Test mapping successful: {'loinc_code': '718-7', 'interpretation': 'LOW', ...}
```

---

## RxNorm Download

### Step 1: Download RxNorm from UMLS

```bash
# Create directory
mkdir -p ~/lca_ontologies/rxnorm
cd ~/lca_ontologies/rxnorm

# Download RxNorm Full Monthly Release
# Via UMLS: https://www.nlm.nih.gov/research/umls/rxnorm/docs/rxnormfiles.html
# Current version: RxNorm Full Release - January 2026

# Download file: RxNorm_full_01032026.zip (approx 800MB)
# Extract
unzip RxNorm_full_01032026.zip

cd rrf/

# Key files:
# - RXNCONSO.RRF (concept names and synonyms)
# - RXNREL.RRF (relationships between concepts)
# - RXNSAT.RRF (attributes)
```

### Step 2: Load RxNorm Data

```python
export RXNORM_PATH="/home/user/lca_ontologies/rxnorm/rrf/"

# Test loading
python3 << 'EOF'
from backend.src.ontology.rxnorm_mapper import RxNormMapper

mapper = RxNormMapper()
mapper.load_rxnorm_data("/home/user/lca_ontologies/rxnorm/rrf/")

print(f"✓ RxNorm concepts loaded: {len(mapper.rxnorm_concepts)}")

# Test medication mapping
result = mapper.map_medication("Osimertinib")
print(f"✓ Osimertinib RxCUI: {result['rxcui']}")

# Test interaction check
interaction = mapper.check_drug_interactions("Osimertinib", "Itraconazole")
print(f"✓ Interaction check: {interaction}")
EOF
```

**Expected Output**:
```
✓ RxNorm concepts loaded: 147523
✓ Osimertinib RxCUI: 1856076
✓ Interaction check: {'has_interaction': True, 'severity': 'major', 'mechanism': 'CYP3A4 inhibition'}
```

---

## Neo4j Setup

### Step 1: Install Neo4j

**Option A: Docker (Recommended)**
```bash
# Pull Neo4j 5.15+ (required for vector search and GDS)
docker pull neo4j:5.15-enterprise

# Run with Graph Data Science plugin
docker run \
    --name lca-neo4j \
    -p7474:7474 -p7687:7687 \
    -e NEO4J_AUTH=neo4j/your_password_here \
    -e NEO4J_PLUGINS='["graph-data-science","apoc"]' \
    -e NEO4J_dbms_security_procedures_unrestricted=gds.*,apoc.* \
    -v $HOME/neo4j/data:/data \
    -v $HOME/neo4j/logs:/logs \
    -d neo4j:5.15-enterprise
```

**Option B: Native Installation (Linux)**
```bash
# Add Neo4j repository
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
echo 'deb https://debian.neo4j.com stable latest' | sudo tee /etc/apt/sources.list.d/neo4j.list

# Install Neo4j
sudo apt-get update
sudo apt-get install neo4j=1:5.15.0

# Start service
sudo systemctl start neo4j
sudo systemctl enable neo4j

# Check status
sudo systemctl status neo4j
```

### Step 2: Install Graph Data Science Plugin

```bash
# Download GDS plugin
cd ~/neo4j/plugins/
wget https://graphdatascience.ninja/neo4j-graph-data-science-2.5.5.jar

# Restart Neo4j
docker restart lca-neo4j
# OR
sudo systemctl restart neo4j

# Verify GDS installation
cypher-shell -u neo4j -p your_password_here << 'EOF'
CALL gds.list();
EOF
```

**Expected**: List of ~50 GDS algorithms

### Step 3: Create Database and Indexes

```bash
# Connect to Neo4j
cypher-shell -u neo4j -p your_password_here

# Create indexes for LCA
CREATE INDEX patient_id_index IF NOT EXISTS FOR (p:Patient) ON (p.patient_id);
CREATE INDEX tnm_stage_index IF NOT EXISTS FOR (p:Patient) ON (p.tnm_stage);
CREATE INDEX histology_index IF NOT EXISTS FOR (p:Patient) ON (p.histology_type);

# Create vector index for patient embeddings
CREATE VECTOR INDEX patient_embeddings IF NOT EXISTS
FOR (p:Patient) ON (p.embedding)
OPTIONS {indexConfig: {
    `vector.dimensions`: 384,
    `vector.similarity_function`: 'cosine'
}};

# Verify indexes
SHOW INDEXES;
```

### Step 4: Configure LCA Connection

```python
# In .env file or environment
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="your_password_here"
export NEO4J_DATABASE="neo4j"
```

---

## Directory Structure

After completing all downloads, your directory structure should look like:

```
~/lca_ontologies/
├── snomed_ct/
│   ├── SnomedCT_InternationalRF2_PRODUCTION_20250131T120000Z/
│   │   ├── Snapshot/
│   │   │   └── Terminology/
│   │   │       └── sct2_Concept_Snapshot_INT_20250131.owl
│   │   └── Full/
│   └── snomed_ct_optimized.owl (generated)
│
├── loinc/
│   └── Loinc_2.78/
│       ├── LoincTable/
│       │   └── Loinc.csv
│       ├── AnswerList/
│       └── AccessoryFiles/
│
├── rxnorm/
│   └── rrf/
│       ├── RXNCONSO.RRF
│       ├── RXNREL.RRF
│       └── RXNSAT.RRF
│
└── lucada/
    └── lucada_ontology.owl (created by LCA)
```

---

## Verification

### Complete Verification Script

```bash
#!/bin/bash

echo "=== LCA Ontology Verification ==="
echo ""

# Check SNOMED CT
if [ -f ~/lca_ontologies/snomed_ct/snomed_ct_optimized.owl ]; then
    echo "✓ SNOMED CT found"
    SIZE=$(du -h ~/lca_ontologies/snomed_ct/snomed_ct_optimized.owl | cut -f1)
    echo "  Size: $SIZE"
else
    echo "✗ SNOMED CT missing"
fi

# Check LOINC
if [ -f ~/lca_ontologies/loinc/Loinc_2.78/LoincTable/Loinc.csv ]; then
    echo "✓ LOINC found"
    LINES=$(wc -l < ~/lca_ontologies/loinc/Loinc_2.78/LoincTable/Loinc.csv)
    echo "  LOINC codes: $((LINES-1))"
else
    echo "✗ LOINC missing"
fi

# Check RxNorm
if [ -f ~/lca_ontologies/rxnorm/rrf/RXNCONSO.RRF ]; then
    echo "✓ RxNorm found"
    LINES=$(wc -l < ~/lca_ontologies/rxnorm/rrf/RXNCONSO.RRF)
    echo "  Concepts: $LINES"
else
    echo "✗ RxNorm missing"
fi

# Check Neo4j
if docker ps | grep -q lca-neo4j; then
    echo "✓ Neo4j running"
elif systemctl is-active --quiet neo4j; then
    echo "✓ Neo4j service active"
else
    echo "✗ Neo4j not running"
fi

echo ""
echo "=== Environment Variables ==="
echo "SNOMED_CT_PATH: ${SNOMED_CT_PATH:-NOT SET}"
echo "LOINC_TABLE_PATH: ${LOINC_TABLE_PATH:-NOT SET}"
echo "RXNORM_PATH: ${RXNORM_PATH:-NOT SET}"
echo "NEO4J_URI: ${NEO4J_URI:-NOT SET}"

echo ""
echo "=== Python Package Verification ==="
python3 -c "
try:
    from owlready2 import *
    print('✓ owlready2 installed')
except ImportError:
    print('✗ owlready2 not installed')

try:
    from neo4j import GraphDatabase
    print('✓ neo4j driver installed')
except ImportError:
    print('✗ neo4j driver not installed')

try:
    from sentence_transformers import SentenceTransformer
    print('✓ sentence-transformers installed')
except ImportError:
    print('✗ sentence-transformers not installed')
"
```

**Save as** `verify_setup.sh` and run:
```bash
chmod +x verify_setup.sh
./verify_setup.sh
```

---

## Troubleshooting

### Issue: SNOMED CT OWL file too large

**Problem**: SNOMED CT OWL is 5GB+, causing memory issues

**Solution**:
```python
# Use chunked loading with owlready2
from owlready2 import *

# Increase Java heap for reasoners
default_world.set_backend(filename="snomed_cache.sqlite3")

# Load incrementally
onto = get_ontology("http://snomed.info/sct/900000000000207008").load()
```

### Issue: LOINC CSV encoding errors

**Problem**: CSV contains special characters

**Solution**:
```python
import pandas as pd

# Load with explicit encoding
loinc_df = pd.read_csv(
    "Loinc.csv",
    encoding='utf-8',
    low_memory=False,
    on_bad_lines='skip'
)
```

### Issue: Neo4j out of memory

**Problem**: Neo4j crashes during large imports

**Solution**:
```bash
# Increase heap size in neo4j.conf
server.memory.heap.initial_size=2G
server.memory.heap.max_size=4G
server.memory.pagecache.size=1G

# Or in Docker:
docker run \
    -e NEO4J_server_memory_heap_initial__size=2G \
    -e NEO4J_server_memory_heap_max__size=4G \
    ...
```

### Issue: RxNorm relationship loading slow

**Problem**: RXNREL.RRF has millions of relationships

**Solution**:
```python
# Use batch processing
import csv

def load_rxnorm_in_batches(filepath, batch_size=10000):
    relationships = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='|')
        batch = []
        for row in reader:
            batch.append(row)
            if len(batch) >= batch_size:
                process_batch(batch)
                batch = []
        if batch:
            process_batch(batch)
```

### Issue: UMLS license not approved

**Problem**: Waiting for UMLS account approval

**Workaround**: Use minimal SNOMED CT subset
```bash
# Download SNOMED CT US Edition (smaller)
# Or use sample data from:
# https://www.nlm.nih.gov/healthit/snomedct/us_edition.html
```

---

## Additional Resources

### Official Documentation

- **SNOMED CT**: https://confluence.ihtsdotools.org/
- **LOINC**: https://loinc.org/kb/
- **RxNorm**: https://www.nlm.nih.gov/research/umls/rxnorm/docs/index.html
- **Neo4j GDS**: https://neo4j.com/docs/graph-data-science/current/

### LCA-Specific Configuration

All ontology paths can be configured in `backend/config/ontology_config.yaml`:

```yaml
ontologies:
  snomed_ct:
    path: ~/lca_ontologies/snomed_ct/snomed_ct_optimized.owl
    version: "2025-01-31"
    load_on_startup: false  # Load on-demand for large ontologies

  loinc:
    path: ~/lca_ontologies/loinc/Loinc_2.78/LoincTable/Loinc.csv
    version: "2.78"
    load_on_startup: true  # CSV is faster to load

  rxnorm:
    path: ~/lca_ontologies/rxnorm/rrf/
    version: "2026-01"
    load_on_startup: true

neo4j:
  uri: "bolt://localhost:7687"
  username: "neo4j"
  password: "${NEO4J_PASSWORD}"  # From environment
  database: "neo4j"
  max_connection_lifetime: 3600
  connection_timeout: 30
```

---

## Support

If you encounter issues not covered here:

1. Check the [LCA GitHub Issues](https://github.com/your-org/lca/issues)
2. Consult the [LCA_Architecture.md](./LCA_Architecture.md) for system design
3. Review the [LCA_2026_Complete_Guide.md](./LCA_2026_Complete_Guide.md) for advanced features

---

**Last Updated**: January 2026
**LCA Version**: 3.0.0
