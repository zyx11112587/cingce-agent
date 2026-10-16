param(
    [Parameter(Position = 0)]
    [ValidateSet("ingest", "search", "ask", "explain", "serve", "stats", "help")]
    [string]$Command = "help",

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $Root

$PythonCommand = Get-Command python -ErrorAction SilentlyContinue
if (-not $PythonCommand) {
    $PythonCommand = Get-Command py -ErrorAction SilentlyContinue
}
if (-not $PythonCommand) {
    Write-Error "Python was not found. Install Python 3.11+ first."
}

$ArgsToPass = @()
if ($Command -eq "help") {
    $ArgsToPass += "--help"
} else {
    $ArgsToPass += $Command
}
if ($RemainingArgs) {
    $ArgsToPass += $RemainingArgs
}

& $PythonCommand.Source -m exam_rag.cli @ArgsToPass
exit $LASTEXITCODE
