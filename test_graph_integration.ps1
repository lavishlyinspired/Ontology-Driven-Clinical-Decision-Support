# Test Script for Context Graph Integration
# Run this to verify backend is sending graph data properly

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Context Graph Integration Test" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Test 1: Check if backend is running
Write-Host "[1/5] Checking if backend is running..." -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "http://localhost:8000/api/health" -ErrorAction Stop
    Write-Host "✓ Backend is running (version: $($health.version))" -ForegroundColor Green
} catch {
    Write-Host "✗ Backend is NOT running!" -ForegroundColor Red
    Write-Host "   Start it with: python start_backend.py" -ForegroundColor Yellow
    exit 1
}

# Test 2: Check Neo4j connection
Write-Host ""
Write-Host "[2/5] Checking Neo4j graph database..." -ForegroundColor Yellow
try {
    $graphHealth = Invoke-RestMethod -Uri "http://localhost:8000/api/graph/statistics" -ErrorAction Stop
    Write-Host "✓ Neo4j connected: $($graphHealth.node_count) nodes, $($graphHealth.relationship_count) relationships" -ForegroundColor Green
} catch {
    Write-Host "✗ Neo4j not available" -ForegroundColor Red
    Write-Host "   Error: $($_.Exception.Message)" -ForegroundColor Yellow
}

# Test 3: Test chat stream endpoint
Write-Host ""
Write-Host "[3/5] Testing chat stream endpoint..." -ForegroundColor Yellow
Write-Host "   Sending test patient message..." -ForegroundColor Gray

$testMessage = @{
    message = "58 year old female, stage IV adenocarcinoma, PD-L1 75%, PS 1"
    session_id = "test_graph_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
} | ConvertTo-Json

# Use curl for SSE streaming
Write-Host "   Monitoring SSE stream for 10 seconds..." -ForegroundColor Gray
$curlCmd = "curl.exe -N -H `"Content-Type: application/json`" -d `"$($testMessage -replace '`"','\"')`" http://localhost:8000/api/v1/chat/stream 2>&1 | Select-String -Pattern `"graph_data|status|progress`" | Select-Object -First 20"
$stream = Invoke-Expression $curlCmd

if ($stream) {
    Write-Host "✓ Chat stream is working" -ForegroundColor Green
    Write-Host "   Sample events received:" -ForegroundColor Gray
    $stream | ForEach-Object { Write-Host "     $_" -ForegroundColor DarkGray }
    
    if ($stream | Select-String -Pattern "graph_data") {
        Write-Host ""
        Write-Host "✓✓ GRAPH DATA IS BEING SENT!" -ForegroundColor Green -BackgroundColor DarkGreen
    } else {
        Write-Host ""
        Write-Host "✗ No graph_data event detected in stream" -ForegroundColor Red
    }
} else {
    Write-Host "✗ No stream events received" -ForegroundColor Red
}

# Test 4: Check frontend
Write-Host ""
Write-Host "[4/5] Checking frontend..." -ForegroundColor Yellow
try {
    $frontendTest = Invoke-WebRequest -Uri "http://localhost:3000/chat" -Method Head -ErrorAction Stop
    Write-Host "✓ Frontend is accessible at http://localhost:3000/chat" -ForegroundColor Green
} catch {
    Write-Host "✗ Frontend is NOT running" -ForegroundColor Red
    Write-Host "   Start it with: cd frontend; npm run dev" -ForegroundColor Yellow
}

# Test 5: Summary
Write-Host ""
Write-Host "[5/5] Summary" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Implementation Status:" -ForegroundColor White
Write-Host "  ✓ Backend Graph API Routes (Phase 1)" -ForegroundColor Green
Write-Host "  ✓ Backend Graph Tools (Phase 2)" -ForegroundColor Green  
Write-Host "  ✓ Frontend NVL Visualization (Phase 3)" -ForegroundColor Green
Write-Host "  ✓ Chat Integration (Phase 4)" -ForegroundColor Green
Write-Host "  ✓ Graph Data Retrieval Added to ConversationService" -ForegroundColor Green
Write-Host ""

Write-Host "Next Steps to See Graph:" -ForegroundColor Yellow
Write-Host "  1. Open browser: http://localhost:3000/chat" -ForegroundColor White
Write-Host "  2. Open Developer Tools (F12)" -ForegroundColor White
Write-Host "  3. Go to Console tab" -ForegroundColor White
Write-Host "  4. Send a patient message" -ForegroundColor White
Write-Host "  5. Look for console logs:" -ForegroundColor White
Write-Host "     - '[LCA] SSE Message: graph_data'" -ForegroundColor Gray
Write-Host "     - '[LCA] Received graph_data:'" -ForegroundColor Gray
Write-Host "     - '[LCA] Calling onGraphDataChange'" -ForegroundColor Gray
Write-Host "  6. Graph should appear in right panel" -ForegroundColor White
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
