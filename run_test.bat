@echo off
echo Starting LCA Test...
timeout /t 5 /nobreak >nul
echo 1
timeout /t 2 /nobreak >nul  
echo y
timeout /t 60 /nobreak >nul
) | python run_lca.py > test_result.log 2>&1
echo Test completed, check test_result.log
