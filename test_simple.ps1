# Simple Graph Integration Test
Write-Host "Context Graph Integration Test" -ForegroundColor Cyan
Write-Host "==============================" -ForegroundColor Cyan

# Test backend
Write-Host "`n[1/3] Checking backend..." -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod http://localhost:8000/api/health
    Write-Host "Backend: RUNNING (v$($health.version))" -ForegroundColor Green
} catch {
    Write-Host "Backend: NOT RUNNING" -ForegroundColor Red
    exit 1
}

# Test Neo4j
Write-Host "`n[2/3] Checking Neo4j..." -ForegroundColor Yellow
try {
    $stats = Invoke-RestMethod http://localhost:8000/api/graph/statistics
    Write-Host "Neo4j: CONNECTED ($($stats.node_count) nodes)" -ForegroundColor Green
} catch {
    Write-Host "Neo4j: ERROR - $($_.Exception.Message)" -ForegroundColor Red
}

# Test frontend
Write-Host "`n[3/3] Checking frontend..." -ForegroundColor Yellow
try {
    $null = Invoke-WebRequest http://localhost:3000/chat -Method Head
    Write-Host "Frontend: RUNNING" -ForegroundColor Green
} catch {
    Write-Host "Frontend: NOT RUNNING" -ForegroundColor Red
}

Write-Host "`n==============================" -ForegroundColor Cyan
Write-Host "To test graph visualization:" -ForegroundColor Yellow
Write-Host "1. Go to http://localhost:3000/chat"
Write-Host "2. Open DevTools (F12) > Console"  
Write-Host "3. Send: '58F, stage IV adenocarcinoma, PD-L1 75%, PS 1'"
Write-Host "4. Watch console for '[LCA] Received graph_data'"
Write-Host "5. Graph should appear in right panel"
