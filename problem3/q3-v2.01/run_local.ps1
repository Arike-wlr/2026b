param(
    [int]$Cases = 20,
    [int]$RandomState = 1405468406
)
$ErrorActionPreference = 'Stop'
$q3BundledPython = Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
if (Test-Path -LiteralPath $q3BundledPython) {
    $q3Python = $q3BundledPython
} else {
    $q3Python = 'python'
}
& $q3Python (Join-Path $PSScriptRoot 'problem3_strategy.py') --cases $Cases --random-state $RandomState
exit $LASTEXITCODE
