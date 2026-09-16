$ErrorActionPreference = "Stop"
$url = "https://data.nlr.gov/system/files/305/1765565264-combined_wind_experiments_0.csv"
$root = Split-Path -Parent $PSScriptRoot
$out = Join-Path $root "data\raw\combined_wind_experiments.csv"
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $out) | Out-Null
Invoke-WebRequest -Uri $url -OutFile $out
Write-Host "Downloaded $out"
