param([int]$Cases = 20, [int]$RandomState = 2007525743)
$ErrorActionPreference = 'Stop'
$q4Runtime = Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
if (Test-Path -LiteralPath $q4Runtime) { $q4Python = $q4Runtime } else { $q4Python = 'python' }
& $q4Python (Join-Path $PSScriptRoot 'problem4_strategy.py') --cases $Cases --random-state $RandomState
exit $LASTEXITCODE
